"""Insertion helpers for HashTable."""
from __future__ import annotations

from typing import TYPE_CHECKING

import chex
import jax
import jax.numpy as jnp

from ..core import Xtructurable
from ..core.xtructure_numpy.array_ops import (
    _update_array_on_condition,
    _where_no_broadcast,
)
from .constants import SIZE_DTYPE, SLOT_IDX_DTYPE
from .hash_utils import _compute_unique_mask_from_uint32eds, get_new_idx_byterized
from .lookup import _hashtable_lookup_bucket_jit, _hashtable_lookup_parallel_internal
from .types import BucketIdx, HashIdx

if TYPE_CHECKING:
    from .table import HashTable


def _resolve_slot_conflicts(flat_indices: chex.Array, active: chex.Array) -> chex.Array:
    active = jnp.asarray(active, dtype=jnp.bool_)
    batch_size = flat_indices.shape[0]
    flat_indices = jnp.asarray(flat_indices, dtype=jnp.uint32)

    batch_idx = jnp.arange(batch_size, dtype=jnp.uint32)
    sentinel = jnp.uint32(0xFFFFFFFF)
    keys = jnp.where(active, flat_indices, sentinel)

    sorted_keys, sorted_batch_idx = jax.lax.sort((keys, batch_idx), dimension=0, is_stable=True)

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
def _hashtable_insert_jit(
    table: "HashTable", input: Xtructurable
) -> tuple["HashTable", bool, HashIdx]:
    def _update_table(
        table: "HashTable", input: Xtructurable, idx: BucketIdx, fingerprint: chex.Array
    ):
        table = table.replace(
            table=table.table.at[idx.index * table.bucket_size + idx.slot_index].set(input),
            fingerprints=table.fingerprints.at[idx.index * table.bucket_size + idx.slot_index].set(
                fingerprint
            ),
            bucket_fill_levels=table.bucket_fill_levels.at[idx.index].add(1),
            size=table.size + 1,
        )
        return table

    idx, found, fingerprint = _hashtable_lookup_bucket_jit(table, input)

    is_empty = idx.slot_index >= table.bucket_fill_levels[idx.index]
    can_insert = jnp.logical_and(~found, is_empty)

    def _no_insert():
        return table

    def _do_insert():
        return _update_table(table, input, idx, fingerprint)

    table = jax.lax.cond(can_insert, _do_insert, _no_insert)
    inserted = can_insert
    bucket_size_u32 = SIZE_DTYPE(table.bucket_size)
    return table, inserted, HashIdx(index=idx.index * bucket_size_u32 + idx.slot_index)


def _hashtable_parallel_insert_internal(
    table: "HashTable",
    inputs: Xtructurable,
    inputs_uint32ed: chex.Array,
    probe_steps: chex.Array,
    index: BucketIdx,
    updatable: chex.Array,
    fingerprints: chex.Array,
) -> tuple["HashTable", BucketIdx]:
    capacity = jnp.asarray(table._capacity, dtype=SIZE_DTYPE)
    probe_steps = jnp.asarray(probe_steps, dtype=SIZE_DTYPE)

    def _advance(idx: BucketIdx, step: chex.Array) -> BucketIdx:
        next_bucket = idx.slot_index >= (table.bucket_size - 1)

        def _next_bucket() -> BucketIdx:
            mask = capacity - SIZE_DTYPE(1)
            next_index = (idx.index + step) & mask
            bucket_fill = table.bucket_fill_levels[next_index]
            return BucketIdx(
                index=SIZE_DTYPE(next_index),
                slot_index=SLOT_IDX_DTYPE(bucket_fill),
            )

        def _same_bucket() -> BucketIdx:
            return BucketIdx(
                index=idx.index,
                slot_index=SLOT_IDX_DTYPE(idx.slot_index + 1),
            )

        return jax.lax.cond(next_bucket, _next_bucket, _same_bucket)

    def _next_idx(idxs: BucketIdx, unupdateds: chex.Array) -> BucketIdx:
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

    def _cond(val: tuple[BucketIdx, chex.Array, chex.Array]) -> bool:
        _, pending, probes = val
        return jnp.logical_and(jnp.any(pending), probes < table.max_probes)

    def _body(
        val: tuple[BucketIdx, chex.Array, chex.Array]
    ) -> tuple[BucketIdx, chex.Array, chex.Array]:
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

    new_table = table.table.at[flat_indices].set_as_condition(successful, inputs)

    new_fingerprints = _update_array_on_condition(
        table.fingerprints,
        flat_indices,
        successful,
        fingerprints.astype(jnp.uint32),
    )
    new_bucket_fill_levels = table.bucket_fill_levels.at[index.index].add(successful)
    new_size = table.size + jnp.sum(successful, dtype=SIZE_DTYPE)

    table = table.replace(
        table=new_table,
        fingerprints=new_fingerprints,
        bucket_fill_levels=new_bucket_fill_levels,
        size=new_size,
    )
    return table, index


@jax.jit
def _hashtable_parallel_insert_jit(
    table: "HashTable",
    inputs: Xtructurable,
    filled: chex.Array | bool = None,
    unique_key: chex.Array = None,
):
    if filled is None:
        filled = jnp.ones(inputs.shape.batch, dtype=jnp.bool_)

    filled = jnp.asarray(filled)
    batch_len = inputs.shape.batch
    bucket_size_u32 = SIZE_DTYPE(table.bucket_size)

    def _process_insert(filled_mask):
        initial_idx, steps, uint32eds, fingerprints = jax.vmap(
            get_new_idx_byterized, in_axes=(0, None, None)
        )(inputs, table._capacity, table.seed)

        unique_filled, representative_indices = _compute_unique_mask_from_uint32eds(
            uint32eds=uint32eds,
            filled=filled_mask,
            unique_key=unique_key,
        )

        idx = BucketIdx(index=initial_idx, slot_index=jnp.zeros(batch_len, dtype=SLOT_IDX_DTYPE))

        initial_found = jnp.logical_not(unique_filled)
        idx, found = _hashtable_lookup_parallel_internal(
            table,
            inputs,
            uint32eds,
            idx,
            steps,
            fingerprints,
            initial_found,
            filled_mask,
        )

        updatable = jnp.logical_and(~found, unique_filled)

        updated_table, inserted_idx = _hashtable_parallel_insert_internal(
            table, inputs, uint32eds, steps, idx, updatable, fingerprints
        )

        cond_found = jnp.asarray(found, dtype=jnp.bool_)

        inserted_index = jnp.asarray(inserted_idx.index, dtype=idx.index.dtype)
        inserted_slot_index = jnp.asarray(inserted_idx.slot_index, dtype=idx.slot_index.dtype)
        current_index = jnp.asarray(idx.index)
        current_slot_index = jnp.asarray(idx.slot_index)

        provisional_index = _where_no_broadcast(
            cond_found,
            current_index,
            inserted_index,
        )
        provisional_slot_index = _where_no_broadcast(
            cond_found,
            current_slot_index,
            inserted_slot_index,
        )
        provisional_idx = BucketIdx(index=provisional_index, slot_index=provisional_slot_index)

        representative_indices = jnp.asarray(representative_indices, dtype=jnp.int32)
        final_idx = BucketIdx(
            index=provisional_idx.index[representative_indices],
            slot_index=provisional_idx.slot_index[representative_indices],
        )

        return (
            updated_table,
            updatable,
            unique_filled,
            HashIdx(index=final_idx.index * bucket_size_u32 + final_idx.slot_index),
        )

    def _empty_insert(_):
        return (
            table,
            jnp.zeros(batch_len, dtype=jnp.bool_),
            jnp.zeros(batch_len, dtype=jnp.bool_),
            HashIdx(index=jnp.zeros(batch_len, dtype=SIZE_DTYPE)),
        )

    if filled.ndim == 0:
        return jax.lax.cond(
            filled,
            lambda: _process_insert(jnp.ones(batch_len, dtype=jnp.bool_)),
            lambda: _empty_insert(None),
        )
    else:
        return _process_insert(filled)
