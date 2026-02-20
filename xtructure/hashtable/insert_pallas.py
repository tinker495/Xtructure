from __future__ import annotations

import os
from functools import lru_cache
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import jax.ops as jops
from jax.experimental import pallas as pl

from .hash_utils import _parse_bool_env, _parse_int_env


def pallas_insert_enabled() -> bool:
    return _parse_bool_env("XTRUCTURE_HASHTABLE_PALLAS_INSERT", True)


def _parse_max_probe_buckets(capacity: int) -> int:
    env_value = os.environ.get("XTRUCTURE_HASHTABLE_PALLAS_INSERT_MAX_PROBE_BUCKETS")
    if env_value is None or env_value.strip().lower() in {"", "none", "auto"}:
        return int(capacity)
    return _parse_int_env("XTRUCTURE_HASHTABLE_PALLAS_INSERT_MAX_PROBE_BUCKETS", int(capacity))


def _validate_bucket_size(bucket_size: int) -> None:
    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive.")
    if bucket_size > 32:
        raise ValueError("Pallas insert path currently supports bucket_size <= 32.")


def _full_mask_u32(bucket_size: int) -> int:
    if bucket_size >= 32:
        return 0xFFFFFFFF
    return (1 << bucket_size) - 1


def _make_bucket_owner_kernel(*, bucket_size: int, capacity: int):
    full_mask = _full_mask_u32(bucket_size)

    def bucket_owner_kernel(
        _fill_in_ref,
        _occ_in_ref,
        sorted_bucket_ref,
        group_start_ref,
        group_len_ref,
        _slot_in_ref,
        _inserted_in_ref,
        fill_ref,
        occ_ref,
        out_slot_ref,
        out_inserted_ref,
    ):
        pid = pl.program_id(axis=0)
        bucket_size_u32 = jnp.uint32(bucket_size)
        capacity_u32 = jnp.uint32(capacity)
        is_group_start = jnp.asarray(group_start_ref[pid], dtype=jnp.bool_)
        bucket = jnp.asarray(sorted_bucket_ref[pid], dtype=jnp.uint32)
        group_len = jnp.asarray(group_len_ref[pid], dtype=jnp.int32)

        valid_bucket = bucket < capacity_u32
        do_group = jnp.logical_and(is_group_start, valid_bucket)

        @pl.when(do_group)
        def _do():
            bucket_i32 = bucket.astype(jnp.int32)
            fill = jnp.asarray(fill_ref[bucket_i32], dtype=jnp.uint32)
            fill = jnp.minimum(fill, bucket_size_u32)

            for offset in range(bucket_size):
                offset_i32 = jnp.int32(offset)
                pos = pid + offset_i32
                in_group = offset_i32 < group_len
                can_insert = jnp.logical_and(in_group, fill < bucket_size_u32)

                @pl.when(can_insert)
                def _store():
                    out_slot_ref[pos] = fill
                    out_inserted_ref[pos] = jnp.bool_(True)

                fill = fill + can_insert.astype(jnp.uint32)

            is_full = fill >= bucket_size_u32
            occ = jax.lax.select(
                is_full,
                jnp.uint32(full_mask),
                jnp.left_shift(jnp.uint32(1), fill) - jnp.uint32(1),
            )
            fill_ref[bucket_i32] = fill
            occ_ref[bucket_i32] = occ

    return bucket_owner_kernel


@lru_cache(maxsize=None)
def _get_reserve_slots_fn(
    *,
    bucket_size: int,
    capacity: int,
    max_probe_buckets: int,
    backend: str,
):
    _validate_bucket_size(bucket_size)
    if capacity <= 0 or (capacity & (capacity - 1)) != 0:
        raise ValueError("capacity must be a positive power of two.")

    max_probe_buckets = min(int(max_probe_buckets), int(capacity))
    kernel = _make_bucket_owner_kernel(bucket_size=bucket_size, capacity=capacity)

    @jax.jit
    def _reserve_slots(
        bucket_fill_levels: chex.Array,
        bucket_occupancy: chex.Array,
        start_buckets: chex.Array,
        probe_steps: chex.Array,
        active: chex.Array,
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        batch = int(start_buckets.shape[0])
        if batch == 0:
            out_bucket = jnp.zeros((0,), dtype=jnp.uint32)
            out_slot = jnp.zeros((0,), dtype=jnp.uint32)
            inserted = jnp.zeros((0,), dtype=jnp.bool_)
            return bucket_fill_levels, bucket_occupancy, out_bucket, out_slot, inserted

        cap_mask = jnp.uint32(capacity - 1)
        sentinel_bucket = jnp.uint32(capacity)
        start_buckets = jnp.asarray(start_buckets, dtype=jnp.uint32).reshape(-1)
        probe_steps = jnp.asarray(probe_steps, dtype=jnp.uint32).reshape(-1)
        active = jnp.asarray(active, dtype=jnp.bool_).reshape(-1)

        result_bucket = start_buckets
        result_slot = jnp.zeros((batch,), dtype=jnp.uint32)
        inserted_any = jnp.zeros((batch,), dtype=jnp.bool_)
        pending = active
        cur_bucket = start_buckets

        def _cond(state):
            _fill, _occ, _cur_bucket, _res_b, _res_s, _ins, pending, rounds = state
            return jnp.logical_and(jnp.any(pending), rounds < max_probe_buckets)

        def _body(state):
            fill, occ, cur_bucket, res_b, res_s, ins, pending, rounds = state
            bucket_key = jnp.where(pending, cur_bucket, sentinel_bucket)
            batch_idx = jnp.arange(batch, dtype=jnp.uint32)

            sorted_bucket, sorted_batch_idx = jax.lax.sort(
                (bucket_key, batch_idx),
                dimension=0,
                is_stable=True,
                num_keys=1,
            )

            row_changed = sorted_bucket[1:] != sorted_bucket[:-1]
            group_start = jnp.concatenate([jnp.array([True]), row_changed], axis=0)
            group_id = jnp.cumsum(group_start.astype(jnp.int32)) - jnp.int32(1)
            group_counts = jops.segment_sum(
                jnp.ones((batch,), dtype=jnp.int32),
                group_id,
                num_segments=batch,
            )
            group_len = jnp.where(group_start, group_counts[group_id], jnp.int32(0))

            slot_init = jnp.zeros((batch,), dtype=jnp.uint32)
            ins_init = jnp.zeros((batch,), dtype=jnp.bool_)

            fill_shape = jax.ShapeDtypeStruct(bucket_fill_levels.shape, bucket_fill_levels.dtype)
            occ_shape = jax.ShapeDtypeStruct(bucket_occupancy.shape, bucket_occupancy.dtype)
            slot_shape = jax.ShapeDtypeStruct((batch,), jnp.uint32)
            ins_shape = jax.ShapeDtypeStruct((batch,), jnp.bool_)
            out_shape = (fill_shape, occ_shape, slot_shape, ins_shape)

            fill, occ, slot_sorted, ins_sorted = pl.pallas_call(
                kernel,
                grid=(batch,),
                out_shape=out_shape,
                backend=backend,
                input_output_aliases={0: 0, 1: 1, 5: 2, 6: 3},
            )(
                fill,
                occ,
                sorted_bucket,
                group_start,
                group_len,
                slot_init,
                ins_init,
            )

            inserted_round = (
                jnp.zeros((batch,), dtype=jnp.bool_).at[sorted_batch_idx].set(ins_sorted)
            )
            slot_round = jnp.zeros((batch,), dtype=jnp.uint32).at[sorted_batch_idx].set(slot_sorted)
            bucket_round = (
                jnp.zeros((batch,), dtype=jnp.uint32).at[sorted_batch_idx].set(sorted_bucket)
            )

            newly_inserted = jnp.logical_and(pending, inserted_round)
            res_b = jnp.where(newly_inserted, bucket_round, res_b)
            res_s = jnp.where(newly_inserted, slot_round, res_s)
            ins = jnp.logical_or(ins, newly_inserted)

            pending = jnp.logical_and(pending, jnp.logical_not(inserted_round))
            cur_bucket = jnp.where(
                pending, jnp.bitwise_and(cur_bucket + probe_steps, cap_mask), cur_bucket
            )
            rounds = rounds + 1
            return fill, occ, cur_bucket, res_b, res_s, ins, pending, rounds

        fill_out, occ_out, _, res_b, res_s, inserted, _, _ = jax.lax.while_loop(
            _cond,
            _body,
            (
                bucket_fill_levels,
                bucket_occupancy,
                cur_bucket,
                result_bucket,
                result_slot,
                inserted_any,
                pending,
                jnp.int32(0),
            ),
        )

        return fill_out, occ_out, res_b, res_s, inserted

    return _reserve_slots


def reserve_slots_pallas(
    bucket_fill_levels: chex.Array,
    bucket_occupancy: chex.Array,
    start_buckets: chex.Array,
    probe_steps: chex.Array,
    active: chex.Array,
    *,
    bucket_size: int,
    capacity: int,
    backend: str,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    cfg_max = _parse_max_probe_buckets(int(capacity))
    fn = _get_reserve_slots_fn(
        bucket_size=int(bucket_size),
        capacity=int(capacity),
        max_probe_buckets=int(cfg_max),
        backend=str(backend),
    )
    return fn(bucket_fill_levels, bucket_occupancy, start_buckets, probe_steps, active)
