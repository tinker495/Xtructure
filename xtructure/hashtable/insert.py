"""Insertion helpers for HashTable."""

from __future__ import annotations

from typing import Any

import chex
import jax
import jax.numpy as jnp

from ..core.dtype_facts import SIZE_DTYPE
from ..core.protocol import Xtructurable
from ..core.xtructure_numpy.array_ops import (
    _update_array_on_condition,
    _where_no_broadcast,
)
from .constants import SLOT_IDX_DTYPE
from .hash_utils import (
    _compute_unique_mask_from_uint32eds,
    get_new_idx_byterized_batched,
)
from .lookup import _hashtable_lookup_bucket_jit, _hashtable_lookup_parallel_internal
from .types import BucketIdx, HashIdx, HashTableProbe


def _check_probe_matches_inputs(probe: HashTableProbe, inputs: Xtructurable) -> None:
    """Fail fast when a supplied probe does not match the insert batch.

    The probe must be exactly what :func:`get_new_idx_byterized_batched` would
    have produced for ``inputs``; a mismatch means the caller threaded intermediates
    from a different batch / table, which would silently corrupt the table. We
    reject it at trace time rather than recomputing behind the caller's back.
    """
    batch_shape = tuple(inputs.shape.batch)
    expected = {
        "index": (batch_shape, SIZE_DTYPE),
        "step": (batch_shape, SIZE_DTYPE),
        "fingerprint": (batch_shape, jnp.uint32),
    }
    for name, (shape, dtype) in expected.items():
        arr = getattr(probe, name)
        if arr.shape != shape:
            raise ValueError(
                f"HashTableProbe.{name} shape {arr.shape} does not match insert batch {shape}"
            )
        if arr.dtype != dtype:
            raise ValueError(
                f"HashTableProbe.{name} dtype {arr.dtype} does not match expected {dtype}"
            )
    if probe.uint32ed.ndim != 2 or probe.uint32ed.shape[:1] != batch_shape:
        raise ValueError(
            f"HashTableProbe.uint32ed shape {probe.uint32ed.shape} is not "
            f"{batch_shape + ('lanes',)} for the insert batch"
        )
    if probe.uint32ed.dtype != jnp.uint32:
        raise ValueError(
            f"HashTableProbe.uint32ed dtype {probe.uint32ed.dtype} does not match expected uint32"
        )


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


def _allocate_initial_bucket_slots(
    table: Any,
    index: BucketIdx,
    updatable: chex.Array,
) -> tuple[chex.Array, chex.Array]:
    """Allocate every candidate that fits in its lookup-selected bucket.

    Parallel lookup leaves each missing value at the first free slot of a
    non-full bucket.  The legacy insert loop then elects one writer for that
    slot, advances the losers by one slot, sorts the entire batch again, and
    repeats.  When every bucket has enough remaining slots, those repeated
    rounds are exactly equivalent to assigning candidates their stable rank
    within the bucket in one sort.

    Returns the ranked slots (batch order, SIZE_DTYPE — may exceed
    bucket_size for overflowing candidates) and a scalar indicating whether
    all active candidates fit. Callers route overflow batches to the
    bucket-level probe allocator, which warm-starts from these slots.
    """
    batch_size = index.index.shape[0]
    batch_idx = jnp.arange(batch_size, dtype=jnp.uint32)
    sentinel = jnp.uint32(0xFFFFFFFF)
    bucket_keys = jnp.where(updatable, index.index.astype(jnp.uint32), sentinel)

    sorted_buckets, sorted_batch_idx = jax.lax.sort(
        (bucket_keys, batch_idx), dimension=0, is_stable=True
    )
    valid_sorted = sorted_buckets != sentinel
    group_starts = jnp.concatenate(
        [jnp.array([True]), sorted_buckets[1:] != sorted_buckets[:-1]], axis=0
    )
    sorted_positions = jnp.arange(batch_size, dtype=SIZE_DTYPE)
    start_positions = jnp.where(group_starts, sorted_positions, SIZE_DTYPE(0))
    group_start_positions = jax.lax.cummax(start_positions, axis=0)
    rank_in_bucket = sorted_positions - group_start_positions

    safe_buckets = jnp.where(valid_sorted, sorted_buckets, jnp.uint32(table._capacity))
    first_free = table.bucket_fill_levels[safe_buckets].astype(SIZE_DTYPE)
    ranked_slots = first_free + rank_in_bucket
    fits_sorted = jnp.logical_or(~valid_sorted, ranked_slots < table.bucket_size)
    all_fit = jnp.all(fits_sorted)

    slots = jnp.zeros((batch_size,), dtype=SIZE_DTYPE)
    slots = slots.at[sorted_batch_idx].set(ranked_slots)
    return slots, all_fit


def _allocate_bucket_slots(
    table: Any,
    index: BucketIdx,
    probe_steps: chex.Array,
    updatable: chex.Array,
    initial_slots: chex.Array,
) -> tuple[BucketIdx, chex.Array]:
    """Allocate stable per-bucket ranks and probe only bucket overflows.

    Warm-starts from the ranked slots of :func:`_allocate_initial_bucket_slots`:
    the prologue below must stay bit-equivalent to the first `_body` round with
    every candidate pending, so the while loop only pays full-batch sorts for
    genuine bucket overflows.
    """
    batch_size = index.index.shape[0]
    batch_idx = jnp.arange(batch_size, dtype=jnp.uint32)
    sentinel = jnp.uint32(0xFFFFFFFF)
    capacity = SIZE_DTYPE(table._capacity)
    capacity_mask = capacity - SIZE_DTYPE(1)
    bucket_size = SIZE_DTYPE(table.bucket_size)
    max_probes = SIZE_DTYPE(table.max_probes)
    probe_steps = jnp.asarray(probe_steps, dtype=SIZE_DTYPE)

    pending = jnp.asarray(updatable, dtype=jnp.bool_)
    first_free = table.bucket_fill_levels[index.index].astype(SIZE_DTYPE)
    rank_in_bucket = initial_slots - first_free
    accepted = jnp.logical_and(
        pending,
        jnp.logical_and(initial_slots < bucket_size, rank_in_bucket <= max_probes),
    )
    slots = jnp.where(accepted, initial_slots, index.slot_index.astype(SIZE_DTYPE))
    steps_to_next = jnp.maximum(SIZE_DTYPE(1), bucket_size - first_free)
    overflow = pending & ~accepted & (initial_slots >= bucket_size) & (steps_to_next <= max_probes)
    buckets = jnp.where(overflow, (index.index + probe_steps) & capacity_mask, index.index)
    probe_count = jnp.where(overflow, steps_to_next, SIZE_DTYPE(0))
    pending = overflow
    assigned = accepted

    def _cond(
        carry: tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array],
    ) -> chex.Array:
        _, _, pending, _, _ = carry
        return jnp.any(pending)

    def _body(
        carry: tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array],
    ) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        buckets, slots, pending, assigned, probe_count = carry
        bucket_keys = jnp.where(updatable, buckets, sentinel)
        sorted_buckets, perm = jax.lax.sort((bucket_keys, batch_idx), dimension=0, is_stable=True)
        sorted_pending = pending[perm]
        sorted_assigned = assigned[perm]
        valid_sorted = sorted_buckets != sentinel

        group_starts = jnp.concatenate(
            [jnp.array([True]), sorted_buckets[1:] != sorted_buckets[:-1]], axis=0
        )
        group_id = jnp.cumsum(group_starts.astype(jnp.int32)) - 1
        assigned_per_group = (
            jnp.zeros((batch_size,), dtype=SIZE_DTYPE)
            .at[group_id]
            .add(sorted_assigned.astype(SIZE_DTYPE))
        )

        pending_prefix = jnp.cumsum(sorted_pending.astype(SIZE_DTYPE))
        prefix_before = pending_prefix - sorted_pending.astype(SIZE_DTYPE)
        group_base_marks = jnp.where(group_starts, prefix_before, SIZE_DTYPE(0))
        group_base = jax.lax.associative_scan(jnp.maximum, group_base_marks)
        pending_rank = pending_prefix - group_base - SIZE_DTYPE(1)

        safe_buckets = jnp.where(valid_sorted, sorted_buckets, capacity)
        first_free = table.bucket_fill_levels[safe_buckets].astype(SIZE_DTYPE)
        rank_offset = assigned_per_group[group_id] + pending_rank
        proposed_slots = first_free + rank_offset
        sorted_probe_count = probe_count[perm]
        accepted_sorted = (
            sorted_pending
            & (proposed_slots < bucket_size)
            & (sorted_probe_count + rank_offset <= max_probes)
        )

        accepted = jnp.zeros_like(pending).at[perm].set(accepted_sorted)
        proposed = jnp.zeros((batch_size,), dtype=SIZE_DTYPE).at[perm].set(proposed_slots)
        slots = jnp.where(accepted, proposed, slots)
        assigned = assigned | accepted

        current_fill = table.bucket_fill_levels[buckets].astype(SIZE_DTYPE)
        steps_to_next = jnp.maximum(SIZE_DTYPE(1), bucket_size - current_fill)
        overflow = (
            pending
            & ~accepted
            & (proposed >= bucket_size)
            & (probe_count + steps_to_next <= max_probes)
        )
        buckets = jnp.where(overflow, (buckets + probe_steps) & capacity_mask, buckets)
        probe_count = jnp.where(overflow, probe_count + steps_to_next, probe_count)
        return buckets, slots, overflow, assigned, probe_count

    buckets, slots, _, assigned, _ = jax.lax.while_loop(
        _cond,
        _body,
        (buckets, slots, pending, assigned, probe_count),
    )
    return (
        BucketIdx(index=buckets, slot_index=slots.astype(SLOT_IDX_DTYPE)),
        jnp.logical_and(updatable, ~assigned),
    )


@jax.jit
def _hashtable_insert_jit(table: Any, input: Xtructurable) -> tuple[Any, bool, HashIdx]:
    def _update_table(table: Any, input: Xtructurable, idx: BucketIdx, fingerprint: chex.Array):
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
    table: Any,
    inputs: Xtructurable,
    probe_steps: chex.Array,
    index: BucketIdx,
    updatable: chex.Array,
    fingerprints: chex.Array,
) -> tuple[Any, BucketIdx]:
    initial_slots, all_fit = _allocate_initial_bucket_slots(table, index, updatable)
    index, pending = jax.lax.cond(
        all_fit,
        lambda: (
            BucketIdx(index=index.index, slot_index=initial_slots.astype(SLOT_IDX_DTYPE)),
            jnp.zeros_like(updatable, dtype=jnp.bool_),
        ),
        lambda: _allocate_bucket_slots(table, index, probe_steps, updatable, initial_slots),
    )

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
    table: Any,
    inputs: Xtructurable,
    filled: chex.Array | bool = None,
    unique_key: chex.Array = None,
    probe: HashTableProbe = None,
):
    if filled is None:
        filled = jnp.ones(inputs.shape.batch, dtype=jnp.bool_)

    filled = jnp.asarray(filled)
    batch_len = inputs.shape.batch
    bucket_size_u32 = SIZE_DTYPE(table.bucket_size)

    if probe is not None:
        _check_probe_matches_inputs(probe, inputs)

    def _process_insert(filled_mask):
        local_probe = (
            probe
            if probe is not None
            else get_new_idx_byterized_batched(inputs, table._capacity, table.seed)
        )
        initial_idx = local_probe.index
        steps = local_probe.step
        uint32eds = local_probe.uint32ed
        fingerprints = local_probe.fingerprint

        unique_filled, representative_indices = _compute_unique_mask_from_uint32eds(
            uint32eds=uint32eds,
            filled=filled_mask,
            unique_key=unique_key,
        )

        idx = BucketIdx(index=initial_idx, slot_index=jnp.zeros(batch_len, dtype=SLOT_IDX_DTYPE))

        initial_found = jnp.logical_not(unique_filled)
        idx, found = jax.lax.cond(
            table.size == 0,
            lambda: (idx, initial_found),
            lambda: _hashtable_lookup_parallel_internal(
                table,
                inputs,
                idx,
                steps,
                fingerprints,
                initial_found,
                filled_mask,
            ),
        )

        updatable = jnp.logical_and(~found, unique_filled)

        updated_table, inserted_idx = _hashtable_parallel_insert_internal(
            table, inputs, steps, idx, updatable, fingerprints
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
