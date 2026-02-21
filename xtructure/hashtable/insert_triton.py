from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pl_triton

from .hash_utils import _parse_bool_env, _parse_int_env


def triton_insert_enabled() -> bool:
    return _parse_bool_env("XTRUCTURE_HASHTABLE_TRITON_INSERT", True)


def _triton_insert_config() -> dict[str, Any]:
    return {
        "max_probe_buckets": _parse_int_env(
            "XTRUCTURE_HASHTABLE_TRITON_INSERT_MAX_PROBE_BUCKETS", 256
        ),
        "num_warps": os.environ.get("XTRUCTURE_HASHTABLE_TRITON_INSERT_NUM_WARPS"),
        "num_stages": os.environ.get("XTRUCTURE_HASHTABLE_TRITON_INSERT_NUM_STAGES"),
    }


def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip().lower()
    if value in {"", "none", "auto"}:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError("must be an integer.") from exc
    if parsed <= 0:
        raise ValueError("must be positive.")
    return parsed


def _validate_bucket_size(bucket_size: int) -> None:
    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive.")
    if bucket_size > 32:
        raise ValueError("Triton insert path currently supports bucket_size <= 32.")


def _full_mask_u32(bucket_size: int) -> int:
    if bucket_size >= 32:
        return 0xFFFFFFFF
    return (1 << bucket_size) - 1


def _make_reserve_slots_kernel(
    *,
    bucket_size: int,
    capacity_mask: int,
    max_probe_buckets: int,
):
    full_mask = _full_mask_u32(bucket_size)
    cap_mask = int(capacity_mask)

    def reserve_slots_kernel(
        _occ_in_ref,
        start_bucket_ref,
        probe_step_ref,
        active_ref,
        occ_ref,
        out_bucket_ref,
        out_slot_ref,
        out_inserted_ref,
    ):
        pid = pl.program_id(axis=0)
        is_active = jnp.asarray(active_ref[pid], dtype=jnp.bool_)
        start_bucket = jnp.asarray(start_bucket_ref[pid], dtype=jnp.uint32)
        step = jnp.asarray(probe_step_ref[pid], dtype=jnp.uint32)

        def _outer_cond(state):
            probe, _bucket, inserted, _slot = state
            return jnp.logical_and(
                jnp.logical_and(is_active, jnp.logical_not(inserted)),
                probe < max_probe_buckets,
            )

        def _outer_body(state):
            probe, bucket, inserted, slot = state
            bucket_i32 = bucket.astype(jnp.int32)

            occ0 = jnp.asarray(occ_ref[bucket_i32], dtype=jnp.uint32)

            def _inner_cond(inner_state):
                attempt, inserted, _slot, _occ = inner_state
                return jnp.logical_and(attempt < bucket_size, jnp.logical_not(inserted))

            def _inner_body(inner_state):
                attempt, inserted, slot, occ = inner_state
                free = jnp.bitwise_and(jnp.bitwise_not(occ), jnp.uint32(full_mask))
                has_free = free != 0
                neg_free = jnp.uint32(0) - free
                lsb = jnp.bitwise_and(free, neg_free)
                do_try = jnp.logical_and(has_free, jnp.logical_not(inserted))

                old = pl_triton.atomic_or(occ_ref, bucket_i32, lsb, mask=do_try)
                claimed = jnp.logical_and(do_try, jnp.bitwise_and(old, lsb) == 0)

                new_occ = jnp.bitwise_or(old, lsb)
                occ = jnp.asarray(
                    jax.lax.select(do_try, new_occ, occ), dtype=jnp.uint32
                )

                slot_candidate = jnp.uint32(31) - jax.lax.clz(lsb)
                slot = jnp.where(claimed, slot_candidate, slot)
                inserted = jnp.logical_or(inserted, claimed)

                attempt = jnp.where(has_free, attempt + 1, jnp.int32(bucket_size))
                return attempt, inserted, slot, occ

            _, inserted, slot, _ = jax.lax.while_loop(
                _inner_cond,
                _inner_body,
                (jnp.int32(0), inserted, slot, occ0),
            )

            next_bucket = jnp.bitwise_and(bucket + step, jnp.uint32(cap_mask))
            bucket = jnp.where(inserted, bucket, next_bucket)
            probe = probe + 1
            return probe, bucket, inserted, slot

        _, bucket, inserted, slot = jax.lax.while_loop(
            _outer_cond,
            _outer_body,
            (jnp.int32(0), start_bucket, jnp.bool_(False), jnp.uint32(0)),
        )

        out_bucket_ref[pid] = bucket
        out_slot_ref[pid] = slot
        out_inserted_ref[pid] = inserted

    return reserve_slots_kernel


@lru_cache(maxsize=None)
def _get_reserve_slots_fn(
    *,
    bucket_size: int,
    capacity: int,
    max_probe_buckets: int,
    num_warps: int | None,
    num_stages: int | None,
):
    _validate_bucket_size(bucket_size)
    if capacity <= 0 or (capacity & (capacity - 1)) != 0:
        raise ValueError("capacity must be a positive power of two.")

    max_probe_buckets = min(int(max_probe_buckets), int(capacity))
    capacity_mask = int(capacity - 1)
    kernel = _make_reserve_slots_kernel(
        bucket_size=bucket_size,
        capacity_mask=capacity_mask,
        max_probe_buckets=max_probe_buckets,
    )

    compiler_params = None
    if num_warps is not None or num_stages is not None:
        compiler_params = pl_triton.CompilerParams(
            num_warps=num_warps, num_stages=num_stages
        )

    @jax.jit
    def _reserve_slots(
        bucket_occupancy: chex.Array,
        start_buckets: chex.Array,
        probe_steps: chex.Array,
        active: chex.Array,
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        batch = int(start_buckets.shape[0])
        occ_shape = jax.ShapeDtypeStruct(bucket_occupancy.shape, bucket_occupancy.dtype)
        out_bucket_shape = jax.ShapeDtypeStruct((batch,), jnp.uint32)
        out_slot_shape = jax.ShapeDtypeStruct((batch,), jnp.uint32)
        out_ins_shape = jax.ShapeDtypeStruct((batch,), jnp.bool_)

        out_shape = (occ_shape, out_bucket_shape, out_slot_shape, out_ins_shape)

        occ_out, out_bucket, out_slot, inserted = pl.pallas_call(
            kernel,
            grid=(batch,),
            out_shape=out_shape,
            backend="triton",
            compiler_params=compiler_params,
            input_output_aliases={0: 0},
        )(
            bucket_occupancy,
            start_buckets,
            probe_steps,
            active,
        )
        return occ_out, out_bucket, out_slot, inserted

    return _reserve_slots


def reserve_slots_triton(
    bucket_occupancy: chex.Array,
    start_buckets: chex.Array,
    probe_steps: chex.Array,
    active: chex.Array,
    *,
    bucket_size: int,
    capacity: int,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    if jax.default_backend() != "gpu":
        raise ValueError("reserve_slots_triton requires a GPU backend.")

    cfg = _triton_insert_config()
    max_probe_env = os.environ.get(
        "XTRUCTURE_HASHTABLE_TRITON_INSERT_MAX_PROBE_BUCKETS"
    )
    if max_probe_env is None or max_probe_env.strip().lower() in {"", "none", "auto"}:
        max_probe_buckets = int(capacity)
    else:
        max_probe_buckets = _parse_int_env(
            "XTRUCTURE_HASHTABLE_TRITON_INSERT_MAX_PROBE_BUCKETS",
            default=int(capacity),
        )

    num_warps = _parse_optional_int(cfg["num_warps"])
    num_stages = _parse_optional_int(cfg["num_stages"])

    fn = _get_reserve_slots_fn(
        bucket_size=int(bucket_size),
        capacity=int(capacity),
        max_probe_buckets=max_probe_buckets,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return fn(bucket_occupancy, start_buckets, probe_steps, active)
