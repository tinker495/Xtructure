"""Insertion helpers for HashTable."""
from __future__ import annotations

from typing import Any, cast

import chex
import jax
import jax.numpy as jnp

from ..core import Xtructurable
from ..core.xtructure_numpy.array_ops import _where_no_broadcast
from .constants import SIZE_DTYPE, SLOT_IDX_DTYPE
from .hash_utils import _compute_unique_mask_from_hash_pairs, get_new_idx_byterized
from .insert_pallas import pallas_insert_enabled, reserve_slots_pallas
from .insert_triton import reserve_slots_triton, triton_insert_enabled
from .lookup import _hashtable_lookup_bucket_jit, _hashtable_lookup_parallel_internal
from .types import BucketIdx, HashIdx

BucketIdxCls = cast(Any, BucketIdx)
HashIdxCls = cast(Any, HashIdx)


def _scatter_safe_indices(
    indices: chex.Array,
    mask: chex.Array,
    *,
    out_of_bounds: int,
) -> chex.Array:
    indices_i32 = jnp.asarray(indices, dtype=jnp.int32).reshape(-1)
    mask = jnp.asarray(mask, dtype=jnp.bool_).reshape(-1)
    oob = jnp.broadcast_to(jnp.int32(int(out_of_bounds)), indices_i32.shape)
    return cast(jax.Array, jax.lax.select(mask, indices_i32, oob))


def _scatter_set_xtructurable_drop(
    original: Xtructurable,
    safe_indices_i32: chex.Array,
    values: Xtructurable,
) -> Xtructurable:
    safe_indices_i32 = jnp.asarray(safe_indices_i32, dtype=jnp.int32).reshape(-1)
    return jax.tree_util.tree_map(
        lambda field, value: field.at[safe_indices_i32].set(value, mode="drop"),
        original,
        values,
    )


def _resolve_slot_conflicts(flat_indices: chex.Array, active: chex.Array) -> chex.Array:
    active = jnp.asarray(active, dtype=jnp.bool_).reshape(-1)
    flat_indices = jnp.asarray(flat_indices, dtype=jnp.uint32).reshape(-1)
    batch_size = int(flat_indices.shape[0])

    batch_idx = jnp.arange(batch_size, dtype=jnp.uint32)
    sentinel = jnp.uint32(0xFFFFFFFF)
    keys = cast(jax.Array, jnp.where(active, flat_indices, sentinel))

    operand = cast(Any, (keys, batch_idx))
    sorted_keys, sorted_batch_idx = cast(
        tuple[jax.Array, jax.Array],
        jax.lax.sort(operand, dimension=0, is_stable=True),
    )

    is_first = jnp.concatenate(
        [jnp.array([True]), sorted_keys[1:] != sorted_keys[:-1]],
        axis=0,
    )
    is_valid = sorted_keys != sentinel
    winners_in_sorted = jnp.logical_and(is_first, is_valid)

    winners = jnp.zeros((batch_size,), dtype=jnp.bool_)
    winners = winners.at[sorted_batch_idx].set(winners_in_sorted)
    return winners


@jax.jit
def _hashtable_insert_jit(table: Any, input: Xtructurable) -> tuple[Any, chex.Array, Any]:
    def _update_table(
        table: Any,
        input: Xtructurable,
        idx: Any,
        fingerprint: chex.Array,
    ):
        table = table.replace(
            table=table.table.at[idx.index * table.bucket_size + idx.slot_index].set(input),
            fingerprints=table.fingerprints.at[idx.index * table.bucket_size + idx.slot_index].set(
                fingerprint
            ),
            bucket_fill_levels=table.bucket_fill_levels.at[idx.index].add(1),
            size=table.size + 1,
        )

        if table.bucket_size <= 32:
            bucket_i32 = jnp.asarray(idx.index, dtype=jnp.int32)
            slot_u32 = jnp.asarray(idx.slot_index, dtype=jnp.uint32)
            bit = jnp.left_shift(jnp.uint32(1), slot_u32)
            table = table.replace(
                bucket_occupancy=table.bucket_occupancy.at[bucket_i32].set(
                    jnp.bitwise_or(table.bucket_occupancy[bucket_i32], bit)
                )
            )
        return table

    idx, found, fingerprint = _hashtable_lookup_bucket_jit(table, input)

    is_empty = idx.slot_index >= table.bucket_fill_levels[idx.index]
    can_insert = jnp.logical_and(jnp.logical_not(found), is_empty)

    def _no_insert():
        return table

    def _do_insert():
        return _update_table(table, input, idx, fingerprint)

    table = jax.lax.cond(can_insert, _do_insert, _no_insert)
    inserted = can_insert
    bucket_size_u32 = SIZE_DTYPE(table.bucket_size)
    return table, inserted, HashIdxCls(index=idx.index * bucket_size_u32 + idx.slot_index)


def _hashtable_parallel_insert_internal(
    table: Any,
    inputs: Xtructurable,
    probe_steps: chex.Array,
    index: Any,
    updatable: chex.Array,
    fingerprints: chex.Array,
) -> tuple[Any, Any, chex.Array]:
    capacity = jnp.asarray(table._capacity, dtype=SIZE_DTYPE)
    probe_steps = jnp.asarray(probe_steps, dtype=SIZE_DTYPE)

    def _advance(idx: Any, step: chex.Array) -> Any:
        next_bucket = idx.slot_index >= (table.bucket_size - 1)

        def _next_bucket() -> Any:
            mask = capacity - SIZE_DTYPE(1)
            next_index = (idx.index + step) & mask
            bucket_fill = table.bucket_fill_levels[next_index]
            return BucketIdxCls(
                index=SIZE_DTYPE(next_index),
                slot_index=SLOT_IDX_DTYPE(bucket_fill),
            )

        def _same_bucket() -> Any:
            return BucketIdxCls(
                index=idx.index,
                slot_index=SLOT_IDX_DTYPE(idx.slot_index + 1),
            )

        return jax.lax.cond(next_bucket, _next_bucket, _same_bucket)

    def _next_idx(idxs: Any, unupdateds: chex.Array) -> Any:
        return jax.vmap(
            lambda active, current_idx, step: jax.lax.cond(
                active,
                lambda: _advance(current_idx, step),
                lambda: current_idx,
            )
        )(unupdateds, idxs, probe_steps)

    valid_initial = index.slot_index < table.bucket_size
    flat_initial_slots = index.index * SIZE_DTYPE(table.bucket_size) + index.slot_index.astype(
        SIZE_DTYPE
    )
    initial_candidates = jnp.logical_and(updatable, valid_initial)
    initial_unique_mask = _resolve_slot_conflicts(flat_initial_slots, initial_candidates)
    pending = jnp.logical_or(
        jnp.logical_and(updatable, jnp.logical_not(initial_unique_mask)),
        jnp.logical_and(updatable, jnp.logical_not(valid_initial)),
    )

    def _cond(val: tuple[Any, chex.Array, chex.Array]) -> chex.Array:
        _, pending, probes = val
        return jnp.logical_and(jnp.any(pending), probes < table.max_probes)

    def _body(val: tuple[Any, chex.Array, chex.Array]) -> tuple[Any, chex.Array, chex.Array]:
        idxs, pending, probes = val
        updated_idxs = _next_idx(idxs, pending)

        valid = updated_idxs.slot_index < table.bucket_size
        flat_updated_slots = updated_idxs.index * SIZE_DTYPE(
            table.bucket_size
        ) + updated_idxs.slot_index.astype(SIZE_DTYPE)

        active_for_unique = jnp.logical_and(updatable, valid)
        unique_mask = _resolve_slot_conflicts(flat_updated_slots, active_for_unique)

        next_pending = jnp.logical_or(
            jnp.logical_and(updatable, jnp.logical_not(unique_mask)),
            jnp.logical_and(updatable, jnp.logical_not(valid)),
        )
        return updated_idxs, next_pending, probes + 1

    index, pending, _ = jax.lax.while_loop(_cond, _body, (index, pending, jnp.uint32(0)))

    successful = jnp.logical_and(updatable, jnp.logical_not(pending))
    successful = jnp.logical_and(successful, index.slot_index < table.bucket_size)
    flat_indices = index.index * SIZE_DTYPE(table.bucket_size) + index.slot_index.astype(SIZE_DTYPE)

    safe_flat_indices = _scatter_safe_indices(
        flat_indices,
        successful,
        out_of_bounds=int(table.fingerprints.shape[0]),
    )
    new_table = _scatter_set_xtructurable_drop(table.table, safe_flat_indices, inputs)
    new_fingerprints = table.fingerprints.at[safe_flat_indices].set(
        jnp.asarray(fingerprints, dtype=jnp.uint32),
        mode="drop",
    )
    new_bucket_fill_levels = table.bucket_fill_levels.at[index.index].add(successful)
    new_size = table.size + jnp.sum(successful, dtype=SIZE_DTYPE)

    table = table.replace(
        table=new_table,
        fingerprints=new_fingerprints,
        bucket_fill_levels=new_bucket_fill_levels,
        size=new_size,
    )

    if table.bucket_size <= 32:
        bucket_i32 = jnp.asarray(index.index, dtype=jnp.int32)
        slot_u32 = jnp.asarray(index.slot_index, dtype=jnp.uint32)
        safe_slot_u32 = jnp.asarray(
            jax.lax.select(successful, slot_u32, jnp.uint32(0)),
            dtype=jnp.uint32,
        )
        bit = jnp.left_shift(jnp.uint32(1), safe_slot_u32)
        bit = jnp.asarray(jax.lax.select(successful, bit, jnp.uint32(0)), dtype=jnp.uint32)
        table = table.replace(bucket_occupancy=table.bucket_occupancy.at[bucket_i32].add(bit))
    return table, index, successful


def _hashtable_parallel_insert_internal_triton(
    table: Any,
    inputs: Xtructurable,
    probe_steps: chex.Array,
    start_bucket_idx: chex.Array,
    updatable: chex.Array,
    fingerprints: chex.Array,
) -> tuple[Any, Any, chex.Array]:
    bucket_size = int(table.bucket_size)
    capacity = int(table._capacity)

    occ, out_bucket, out_slot, inserted = reserve_slots_triton(
        table.bucket_occupancy,
        start_bucket_idx,
        probe_steps,
        updatable,
        bucket_size=bucket_size,
        capacity=capacity,
    )

    occ: jax.Array = cast(jax.Array, occ)
    out_bucket: jax.Array = cast(jax.Array, out_bucket)

    bucket_size_u32 = SIZE_DTYPE(bucket_size)
    flat_indices = out_bucket.astype(SIZE_DTYPE) * bucket_size_u32 + out_slot.astype(SIZE_DTYPE)

    safe_flat_indices = _scatter_safe_indices(
        flat_indices,
        inserted,
        out_of_bounds=int(table.fingerprints.shape[0]),
    )
    new_table = _scatter_set_xtructurable_drop(table.table, safe_flat_indices, inputs)
    new_fingerprints = table.fingerprints.at[safe_flat_indices].set(
        jnp.asarray(fingerprints, dtype=jnp.uint32),
        mode="drop",
    )
    new_size = table.size + jnp.sum(inserted, dtype=SIZE_DTYPE)

    full_mask = jnp.uint32(0xFFFFFFFF if bucket_size >= 32 else (1 << bucket_size) - 1)
    occ_rows = occ[out_bucket.astype(jnp.int32)]
    fill_rows = jax.lax.population_count(jnp.bitwise_and(occ_rows, full_mask)).astype(SIZE_DTYPE)
    safe_buckets = _scatter_safe_indices(
        out_bucket,
        inserted,
        out_of_bounds=int(table.bucket_fill_levels.shape[0]),
    )
    new_bucket_fill_levels = table.bucket_fill_levels.at[safe_buckets].set(fill_rows, mode="drop")

    table = table.replace(
        table=new_table,
        fingerprints=new_fingerprints,
        bucket_fill_levels=new_bucket_fill_levels,
        bucket_occupancy=occ,
        size=new_size,
    )
    inserted_idx = BucketIdxCls(
        index=out_bucket.astype(SIZE_DTYPE),
        slot_index=out_slot.astype(SLOT_IDX_DTYPE),
    )
    return table, inserted_idx, inserted


def _hashtable_parallel_insert_internal_pallas(
    table: Any,
    inputs: Xtructurable,
    probe_steps: chex.Array,
    start_bucket_idx: chex.Array,
    updatable: chex.Array,
    fingerprints: chex.Array,
) -> tuple[Any, Any, chex.Array]:
    bucket_size = int(table.bucket_size)
    capacity = int(table._capacity)

    fill, occ, out_bucket, out_slot, inserted = reserve_slots_pallas(
        table.bucket_fill_levels,
        table.bucket_occupancy,
        start_bucket_idx,
        probe_steps,
        updatable,
        bucket_size=bucket_size,
        capacity=capacity,
        backend="mosaic_tpu",
    )

    bucket_size_u32 = SIZE_DTYPE(bucket_size)
    flat_indices = out_bucket.astype(SIZE_DTYPE) * bucket_size_u32 + out_slot.astype(SIZE_DTYPE)

    safe_flat_indices = _scatter_safe_indices(
        flat_indices,
        inserted,
        out_of_bounds=int(table.fingerprints.shape[0]),
    )
    new_table = _scatter_set_xtructurable_drop(table.table, safe_flat_indices, inputs)
    new_fingerprints = table.fingerprints.at[safe_flat_indices].set(
        jnp.asarray(fingerprints, dtype=jnp.uint32),
        mode="drop",
    )
    new_size = table.size + jnp.sum(inserted, dtype=SIZE_DTYPE)

    table = table.replace(
        table=new_table,
        fingerprints=new_fingerprints,
        bucket_fill_levels=fill,
        bucket_occupancy=occ,
        size=new_size,
    )
    inserted_idx = BucketIdxCls(
        index=out_bucket.astype(SIZE_DTYPE),
        slot_index=out_slot.astype(SLOT_IDX_DTYPE),
    )
    return table, inserted_idx, inserted


@jax.jit
def _hashtable_parallel_insert_jit(
    table: Any,
    inputs: Xtructurable,
    filled: chex.Array | bool | None = None,
    unique_key: chex.Array | None = None,
):
    if filled is None:
        filled = jnp.ones(inputs.shape.batch, dtype=jnp.bool_)

    filled = jnp.asarray(filled)
    batch_len = inputs.shape.batch
    bucket_size_u32 = SIZE_DTYPE(table.bucket_size)

    def _process_insert(filled_mask):
        initial_idx, steps, uint32eds, fingerprints, primary_hashes, secondary_hashes = cast(
            tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
            jax.vmap(get_new_idx_byterized, in_axes=(0, None, None))(
                inputs, table._capacity, table.seed
            ),
        )

        unique_filled, representative_indices = _compute_unique_mask_from_hash_pairs(
            primary_hashes,
            secondary_hashes,
            filled_mask,
            unique_key,
            uint32eds=uint32eds,
        )

        idx = BucketIdxCls(index=initial_idx, slot_index=jnp.zeros(batch_len, dtype=SLOT_IDX_DTYPE))

        initial_found = jnp.logical_not(unique_filled)
        idx, found = _hashtable_lookup_parallel_internal(
            table,
            inputs,
            idx,
            steps,
            fingerprints,
            initial_found,
            filled_mask,
        )

        updatable = jnp.logical_and(jnp.logical_not(found), unique_filled)

        use_triton = (
            jax.default_backend() == "gpu" and table.bucket_size <= 32 and triton_insert_enabled()
        )

        use_pallas = (
            jax.default_backend() == "tpu" and table.bucket_size <= 32 and pallas_insert_enabled()
        )

        if use_triton:
            updated_table, inserted_idx, inserted = _hashtable_parallel_insert_internal_triton(
                table,
                inputs,
                steps,
                idx.index,
                updatable,
                fingerprints,
            )
        elif use_pallas:
            updated_table, inserted_idx, inserted = _hashtable_parallel_insert_internal_pallas(
                table,
                inputs,
                steps,
                idx.index,
                updatable,
                fingerprints,
            )
        else:
            updated_table, inserted_idx, inserted = _hashtable_parallel_insert_internal(
                table,
                inputs,
                steps,
                idx,
                updatable,
                fingerprints,
            )

        cond_found = jnp.asarray(found, dtype=jnp.bool_)
        cond_keep_current = jnp.logical_or(cond_found, jnp.logical_not(inserted))

        inserted_index = jnp.asarray(inserted_idx.index, dtype=idx.index.dtype)
        inserted_slot_index = jnp.asarray(inserted_idx.slot_index, dtype=idx.slot_index.dtype)
        current_index = jnp.asarray(idx.index)
        current_slot_index = jnp.asarray(idx.slot_index)

        provisional_index = _where_no_broadcast(
            cond_keep_current,
            current_index,
            inserted_index,
        )
        provisional_slot_index = _where_no_broadcast(
            cond_keep_current,
            current_slot_index,
            inserted_slot_index,
        )
        provisional_idx = BucketIdxCls(index=provisional_index, slot_index=provisional_slot_index)

        representative_indices = jnp.asarray(representative_indices, dtype=jnp.int32)
        final_idx = BucketIdxCls(
            index=provisional_idx.index[representative_indices],
            slot_index=provisional_idx.slot_index[representative_indices],
        )

        return (
            updated_table,
            inserted,
            unique_filled,
            HashIdxCls(index=final_idx.index * bucket_size_u32 + final_idx.slot_index),
        )

    def _empty_insert(_):
        return (
            table,
            jnp.zeros(batch_len, dtype=jnp.bool_),
            jnp.zeros(batch_len, dtype=jnp.bool_),
            HashIdxCls(index=jnp.zeros(batch_len, dtype=SIZE_DTYPE)),
        )

    if filled.ndim == 0:
        return jax.lax.cond(
            filled,
            lambda: _process_insert(jnp.ones(batch_len, dtype=jnp.bool_)),
            lambda: _empty_insert(None),
        )
    else:
        return _process_insert(filled)
