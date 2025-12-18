"""Bucketed open-addressing hash table for Xtructurables.

This implementation uses double hashing to probe buckets, with a small fixed number
of slots per bucket (`slots_per_bucket`). A 32-bit fingerprint is stored per-slot to
avoid loading/comparing full values on most probes.
"""

import dataclasses
from functools import partial
from typing import TypeVar

import chex
import jax
import jax.numpy as jnp

from ..core import FieldDescriptor, Xtructurable, xtructure_dataclass
from ..core.xtructure_decorators.hash import uint32ed_to_hash
from ..core.xtructure_numpy.array_ops import (
    _update_array_on_condition,
    _where_no_broadcast,
)

SIZE_DTYPE = jnp.uint32
HASH_TABLE_IDX_DTYPE = jnp.uint8
DOUBLE_HASH_SECONDARY_DELTA = jnp.uint32(0x9E3779B1)
FINGERPRINT_MIX_CONSTANT_A = jnp.uint32(0x85EBCA6B)
FINGERPRINT_MIX_CONSTANT_B = jnp.uint32(0xC2B2AE35)

T = TypeVar("T")


@xtructure_dataclass
class TableIdx:
    index: FieldDescriptor.scalar(dtype=SIZE_DTYPE)
    table_index: FieldDescriptor.scalar(dtype=HASH_TABLE_IDX_DTYPE)


@xtructure_dataclass
class HashIdx:
    index: FieldDescriptor.scalar(dtype=SIZE_DTYPE)


def _mix_fingerprint(primary: chex.Array, secondary: chex.Array, length: chex.Array) -> chex.Array:
    mix = jnp.asarray(primary, dtype=jnp.uint32)
    secondary = jnp.asarray(secondary, dtype=jnp.uint32)
    length = jnp.asarray(length, dtype=jnp.uint32)

    mix ^= jnp.uint32(0x9E3779B9)
    mix = jnp.uint32(
        mix + secondary * FINGERPRINT_MIX_CONSTANT_A + length * FINGERPRINT_MIX_CONSTANT_B
    )
    mix ^= mix >> jnp.uint32(16)
    mix *= jnp.uint32(0x7FEB352D)
    mix ^= mix >> jnp.uint32(15)
    return mix


def _first_occurrence_mask(
    values: chex.Array, active: chex.Array, sentinel: chex.Array
) -> chex.Array:
    active = jnp.asarray(active, dtype=jnp.bool_)
    values = jnp.asarray(values, dtype=jnp.uint32)
    sentinel = jnp.asarray(sentinel, dtype=jnp.uint32)

    safe_values = jnp.where(active, values, sentinel)
    _, unique_indices = jnp.unique(
        safe_values,
        size=values.shape[0],
        return_index=True,
        return_inverse=False,
        fill_value=sentinel,
    )
    mask = jnp.zeros_like(active, dtype=jnp.bool_).at[unique_indices].set(True)
    return jnp.logical_and(mask, active)


def _compute_unique_mask_from_uint32eds(
    uint32eds: chex.Array,
    filled: chex.Array,
    unique_key: chex.Array | None,
) -> tuple[chex.Array, chex.Array]:
    filled = jnp.asarray(filled, dtype=jnp.bool_)
    batch_len = filled.shape[0]

    if uint32eds.ndim == 1:
        uint32eds = uint32eds[:, None]

    sentinel_row = jnp.full_like(uint32eds, jnp.uint32(0xFFFFFFFF))
    safe_uint32eds = jnp.where(filled[:, None], uint32eds, sentinel_row)
    fill_row = jnp.full((uint32eds.shape[1],), jnp.uint32(0xFFFFFFFF))
    _, unique_indices, inverse = jnp.unique(
        safe_uint32eds,
        axis=0,
        size=batch_len,
        fill_value=fill_row,
        return_index=True,
        return_inverse=True,
    )

    indices = jnp.arange(batch_len, dtype=jnp.int32)

    if unique_key is not None:
        masked_key = jnp.where(filled, unique_key, jnp.inf)
        min_keys = (
            jnp.full((batch_len,), jnp.inf, dtype=masked_key.dtype).at[inverse].min(masked_key)
        )
        candidate_indices = jnp.where(masked_key == min_keys[inverse], indices, batch_len)
    else:
        candidate_indices = jnp.where(filled, indices, batch_len)

    representative_per_group = (
        jnp.full((batch_len,), batch_len, dtype=jnp.int32).at[inverse].min(candidate_indices)
    )
    representative_per_group = jnp.where(
        representative_per_group == batch_len, 0, representative_per_group
    )

    representative_indices = representative_per_group[inverse]
    representative_indices = jnp.where(filled, representative_indices, 0)

    unique_mask = jnp.logical_and(filled, indices == representative_indices)
    return unique_mask, representative_indices


def _normalize_probe_step(step: chex.Array, modulus: int) -> chex.Array:
    step = jnp.asarray(step, dtype=SIZE_DTYPE)
    modulus = jnp.asarray(modulus, dtype=SIZE_DTYPE)
    # Since modulus is power-of-two, bitwise AND is equivalent to modulo
    step = step & (modulus - 1)
    # Ensure step is non-zero and odd to guarantee full cycle in power-of-two table
    step = jnp.where(step == 0, SIZE_DTYPE(1), step)
    step = jnp.bitwise_or(step, SIZE_DTYPE(1))
    return step


def get_new_idx_from_uint32ed(
    input_uint32ed: chex.Array,
    modulus: int,
    seed: int,
) -> tuple[chex.Array, chex.Array, chex.Array]:
    """
    Calculate new index for input state using the hash function from its uint32ed representation
    and reduce it modulo the provided table capacity.
    """
    seed_u32 = jnp.asarray(seed, dtype=jnp.uint32)
    primary_hash = uint32ed_to_hash(input_uint32ed, seed_u32)
    secondary_seed = jnp.bitwise_xor(seed_u32, DOUBLE_HASH_SECONDARY_DELTA)
    secondary_hash = uint32ed_to_hash(input_uint32ed, secondary_seed)
    # Use bitwise AND for power-of-two capacity
    index = primary_hash & (modulus - 1)
    step = _normalize_probe_step(secondary_hash, modulus)
    length = jnp.uint32(input_uint32ed.size)
    fingerprint = _mix_fingerprint(primary_hash, secondary_hash, length)
    return index, step, fingerprint


def get_new_idx_byterized(
    input: Xtructurable,
    modulus: int,
    seed: int,
) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """
    Calculate new index and return uint32ed representation of input state.
    Similar to get_new_idx but also returns the uint32ed representation for
    equality comparison.
    """
    hash_value, uint32ed = input.hash_with_uint32ed(seed)
    seed_u32 = jnp.asarray(seed, dtype=jnp.uint32)
    secondary_seed = jnp.bitwise_xor(seed_u32, DOUBLE_HASH_SECONDARY_DELTA)
    secondary_hash = uint32ed_to_hash(uint32ed, secondary_seed)
    # Use bitwise AND for power-of-two capacity
    idx = hash_value & (modulus - 1)
    step = _normalize_probe_step(secondary_hash, modulus)
    length = jnp.uint32(uint32ed.size)
    fingerprint = _mix_fingerprint(hash_value, secondary_hash, length)
    return idx, step, uint32ed, fingerprint


def _hashtable_build(
    dataclass: Xtructurable,
    seed: int,
    capacity: int,
    slots_per_bucket: int = 32,
    hash_size_multiplier: int = 2,
    max_probes: int | None = None,
) -> "HashTable":
    # Calculate target internal capacity and round up to power of two
    target_capacity = int(hash_size_multiplier * capacity / slots_per_bucket)
    _capacity = 1 << (target_capacity - 1).bit_length()
    _capacity = max(_capacity, 2)  # Ensure at least 2 buckets

    if max_probes is None:
        max_probes = _capacity * slots_per_bucket
    max_probes = max(1, int(max_probes))
    size = SIZE_DTYPE(0)
    # Initialize table with default states. No need for +1 as power-of-two
    # with bitmasking handles bounds naturally.
    # Note: we use jax.jit here only for the array creation to keep it on device

    @jax.jit
    def _create_arrays():
        table = dataclass.default((_capacity * slots_per_bucket,))
        table_idx = jnp.zeros((_capacity,), dtype=HASH_TABLE_IDX_DTYPE)
        fingerprints = jnp.zeros((_capacity * slots_per_bucket,), dtype=jnp.uint32)
        return table, table_idx, fingerprints

    table, table_idx, fingerprints = _create_arrays()

    return HashTable(
        seed=int(seed),
        capacity=int(capacity),
        _capacity=int(_capacity),
        slots_per_bucket=int(slots_per_bucket),
        max_probes=int(max_probes),
        size=size,
        table=table,
        table_idx=table_idx,
        fingerprints=fingerprints,
    )


def _hashtable_lookup_internal(
    table: "HashTable",
    slots_per_bucket: int,
    input: Xtructurable,
    input_uint32ed: chex.Array,
    idx: TableIdx,
    probe_step: chex.Array,
    input_fingerprint: chex.Array,
    found: bool,
) -> tuple[TableIdx, bool]:
    del input_uint32ed
    probe_step = jnp.asarray(probe_step, dtype=SIZE_DTYPE)
    capacity = jnp.asarray(table._capacity, dtype=SIZE_DTYPE)
    max_probes = jnp.asarray(table.max_probes, dtype=SIZE_DTYPE)

    def _cond(val: tuple[TableIdx, bool, chex.Array]) -> bool:
        idx, found, probes = val
        within_budget = probes < max_probes
        return jnp.logical_and(within_budget, ~found)

    def _body(val: tuple[TableIdx, bool, chex.Array]) -> tuple[TableIdx, bool, chex.Array]:
        idx, found, probes = val
        bucket_idx = idx.index
        filled_limit = table.table_idx[bucket_idx]

        # Scan the entire bucket (Horizontal Scan) using static slots_per_bucket
        slot_offsets = jnp.arange(slots_per_bucket, dtype=HASH_TABLE_IDX_DTYPE)
        flat_indices = bucket_idx * slots_per_bucket + slot_offsets
        stored_fps = table.fingerprints[flat_indices]

        # Fingerprint match within the filled part of the bucket
        fp_matches = (stored_fps == input_fingerprint) & (slot_offsets < filled_limit)
        has_fp_match = jnp.any(fp_matches)
        first_match_slot = jnp.argmax(fp_matches).astype(HASH_TABLE_IDX_DTYPE)
        candidate_flat = bucket_idx * slots_per_bucket + first_match_slot

        # Fast path: check only the first fingerprint match.
        candidate_equal = jax.lax.cond(
            has_fp_match,
            lambda: jnp.asarray(table.table[candidate_flat] == input, dtype=jnp.bool_),
            lambda: jnp.bool_(False),
        )

        found_fast = jnp.logical_and(has_fp_match, candidate_equal)

        # Rare path: fingerprint collision (or multiple fp matches). Only then scan other fp matches.
        other_fp_matches = fp_matches & (slot_offsets != first_match_slot)
        need_fallback = jnp.logical_and(~found_fast, jnp.any(other_fp_matches))

        def _fallback_scan():
            def _check_slot(slot_idx):
                return jax.lax.cond(
                    other_fp_matches[slot_idx],
                    lambda: table.table[flat_indices[slot_idx]] == input,
                    lambda: jnp.bool_(False),
                )

            slot_equals = jax.vmap(_check_slot)(slot_offsets)
            found_in_bucket = jnp.any(slot_equals)
            match_table_idx = jnp.argmax(slot_equals).astype(HASH_TABLE_IDX_DTYPE)
            return found_in_bucket, match_table_idx

        found_fb, match_fb = jax.lax.cond(
            need_fallback,
            lambda: _fallback_scan(),
            lambda: (jnp.bool_(False), HASH_TABLE_IDX_DTYPE(0)),
        )

        new_found = jnp.logical_or(found_fast, found_fb)
        match_table_idx = jnp.where(found_fast, first_match_slot, match_fb).astype(
            HASH_TABLE_IDX_DTYPE
        )

        bucket_full = filled_limit == slots_per_bucket
        should_stop = new_found | ~bucket_full

        next_bucket_idx = (bucket_idx + probe_step) & (capacity - 1)
        next_idx = TableIdx(
            index=jnp.where(should_stop, bucket_idx, next_bucket_idx),
            table_index=jnp.where(new_found, match_table_idx, filled_limit),
        )

        return (
            next_idx,
            new_found,
            jnp.where(should_stop, max_probes, probes + SIZE_DTYPE(slots_per_bucket)),
        )

    idx, found, _ = jax.lax.while_loop(_cond, _body, (idx, found, SIZE_DTYPE(0)))
    return idx, found


@partial(jax.jit, static_argnums=(1,))
def _hashtable_lookup_slot_jit(
    table: "HashTable", slots_per_bucket: int, input: Xtructurable
) -> tuple[TableIdx, bool, chex.Array]:
    index, step, input_uint32ed, fingerprint = get_new_idx_byterized(
        input, table._capacity, table.seed
    )
    idx = TableIdx(index=index, table_index=HASH_TABLE_IDX_DTYPE(0))
    idx, found = _hashtable_lookup_internal(
        table, slots_per_bucket, input, input_uint32ed, idx, step, fingerprint, False
    )
    return idx, found, fingerprint


@partial(jax.jit, static_argnums=(1,))
def _hashtable_lookup_jit(
    table: "HashTable", slots_per_bucket: int, input: Xtructurable
) -> tuple[HashIdx, bool]:
    idx, found, _ = _hashtable_lookup_slot_jit(table, slots_per_bucket, input)
    return HashIdx(index=idx.index * slots_per_bucket + idx.table_index), found


def _hashtable_lookup_parallel_internal(
    table: "HashTable",
    slots_per_bucket: int,
    inputs: Xtructurable,
    input_uint32eds: chex.Array,
    idxs: TableIdx,
    probe_steps: chex.Array,
    query_fingerprints: chex.Array,
    founds: chex.Array,
    active: chex.Array | None = None,
) -> tuple[TableIdx, chex.Array]:
    del input_uint32eds
    batch_size = inputs.shape.batch
    if active is None:
        active = jnp.ones(batch_size, dtype=jnp.bool_)

    capacity = jnp.asarray(table._capacity, dtype=SIZE_DTYPE)
    max_probes = jnp.asarray(table.max_probes, dtype=SIZE_DTYPE)
    slot_offsets = jnp.arange(slots_per_bucket, dtype=HASH_TABLE_IDX_DTYPE)

    def _cond(val: tuple[TableIdx, chex.Array, chex.Array, chex.Array]) -> bool:
        _, _, _, active = val
        return jnp.any(active)

    def _body(
        val: tuple[TableIdx, chex.Array, chex.Array, chex.Array]
    ) -> tuple[TableIdx, chex.Array, chex.Array, chex.Array]:
        idxs, founds, probes, active = val
        bucket_indices = idxs.index

        # Horizontal Scan: Calculate flat indices for all slots in the current buckets
        flat_indices = bucket_indices[:, None] * slots_per_bucket + slot_offsets
        # [batch_size, slots_per_bucket]
        stored_fps = table.fingerprints[flat_indices]

        # Fingerprint match within the filled part of each bucket
        filled_limits = table.table_idx[bucket_indices]
        is_filled = slot_offsets < filled_limits[:, None]
        fp_matches = (stored_fps == query_fingerprints[:, None]) & is_filled & active[:, None]
        has_fp_match_row = jnp.any(fp_matches, axis=1)
        first_match_slot = jnp.argmax(fp_matches, axis=1).astype(HASH_TABLE_IDX_DTYPE)  # 0 if none
        candidate_flat = bucket_indices * slots_per_bucket + first_match_slot

        # Fast path: compare only the first fp match per row (batch x 1 gather).
        candidate_states = table.table[candidate_flat]
        candidate_equal = jax.vmap(lambda s, x: s == x)(candidate_states, inputs)
        found_fast = has_fp_match_row & candidate_equal

        # Rare path: fp collision (or multiple matches) -> scan remaining fp matches.
        other_fp_matches = fp_matches & (slot_offsets[None, :] != first_match_slot[:, None])
        need_fallback = (~found_fast) & jnp.any(other_fp_matches, axis=1)
        any_need_fallback = jnp.any(need_fallback)

        def _fallback_scan():
            def _compare_single_input(bucket_data, input_val):
                return jax.vmap(lambda slot_data: slot_data == input_val)(bucket_data)

            value_equals = jax.vmap(_compare_single_input)(table.table[flat_indices], inputs)
            match_found = value_equals & other_fp_matches & need_fallback[:, None]
            found_fb = jnp.any(match_found, axis=1)
            idx_fb = jnp.argmax(match_found, axis=1).astype(HASH_TABLE_IDX_DTYPE)
            return found_fb, idx_fb

        found_fb, idx_fb = jax.lax.cond(
            any_need_fallback,
            _fallback_scan,
            lambda: (
                jnp.zeros_like(need_fallback, dtype=jnp.bool_),
                jnp.zeros_like(first_match_slot, dtype=HASH_TABLE_IDX_DTYPE),
            ),
        )

        new_founds_in_bucket = found_fast | found_fb
        new_founds = jnp.logical_or(founds, new_founds_in_bucket)

        match_table_indices = jnp.where(found_fast, first_match_slot, idx_fb).astype(
            HASH_TABLE_IDX_DTYPE
        )

        # Update probes
        new_probes = probes + active.astype(SIZE_DTYPE) * SIZE_DTYPE(slots_per_bucket)

        # Determine who should still be active for NEXT iteration
        bucket_full = filled_limits == slots_per_bucket
        still_active = active & ~new_founds & (new_probes < max_probes) & bucket_full

        # Advance bucket_indices for those still active
        next_bucket_indices = (bucket_indices + probe_steps) & (capacity - 1)

        updated_index = jnp.where(still_active, next_bucket_indices, bucket_indices)
        updated_table_index = jnp.where(new_founds_in_bucket, match_table_indices, filled_limits)
        updated_idxs = TableIdx(index=updated_index, table_index=updated_table_index)

        return updated_idxs, new_founds, new_probes, still_active

    # Initial activity check
    initial_active = active & ~founds
    probes = jnp.zeros(batch_size, dtype=SIZE_DTYPE)

    idxs, founds, _, _ = jax.lax.while_loop(_cond, _body, (idxs, founds, probes, initial_active))
    return idxs, founds


@partial(jax.jit, static_argnums=(1,))
def _hashtable_lookup_parallel_jit(
    table: "HashTable",
    slots_per_bucket: int,
    inputs: Xtructurable,
    filled: chex.Array | bool = True,
) -> tuple[HashIdx, chex.Array]:
    filled = jnp.asarray(filled)
    batch_size = inputs.shape.batch

    def _process_batch(filled_mask):
        initial_idx, steps, input_uint32eds, fingerprints = jax.vmap(
            get_new_idx_byterized, in_axes=(0, None, None)
        )(inputs, table._capacity, table.seed)

        idxs = TableIdx(
            index=initial_idx, table_index=jnp.zeros(batch_size, dtype=HASH_TABLE_IDX_DTYPE)
        )
        founds = jnp.zeros(batch_size, dtype=jnp.bool_)

        idx, found = _hashtable_lookup_parallel_internal(
            table,
            slots_per_bucket,
            inputs,
            input_uint32eds,
            idxs,
            steps,
            fingerprints,
            founds,
            filled_mask,
        )
        return HashIdx(index=idx.index * slots_per_bucket + idx.table_index), found

    def _empty_result(_):
        return (
            HashIdx(index=jnp.zeros(batch_size, dtype=SIZE_DTYPE)),
            jnp.zeros(batch_size, dtype=jnp.bool_),
        )

    if filled.ndim == 0:
        return jax.lax.cond(
            filled,
            lambda: _process_batch(jnp.ones(batch_size, dtype=jnp.bool_)),
            lambda: _empty_result(None),
        )
    else:
        return _process_batch(filled)


@partial(jax.jit, static_argnums=(1,))
def _hashtable_insert_jit(
    table: "HashTable", slots_per_bucket: int, input: Xtructurable
) -> tuple["HashTable", bool, HashIdx]:
    def _update_table(
        table: "HashTable", input: Xtructurable, idx: TableIdx, fingerprint: chex.Array
    ):
        flat_index = idx.index * slots_per_bucket + idx.table_index
        table = table.replace(
            table=table.table.at[flat_index].set(input),
            fingerprints=table.fingerprints.at[flat_index].set(fingerprint),
            table_idx=table.table_idx.at[idx.index].add(1),
            size=table.size + SIZE_DTYPE(1),
        )
        return table

    idx, found, fingerprint = _hashtable_lookup_slot_jit(table, slots_per_bucket, input)
    empty_slot = idx.table_index >= table.table_idx[idx.index]
    can_insert = jnp.logical_and(~found, empty_slot)

    def _no_insert():
        return table

    def _do_insert():
        return _update_table(table, input, idx, fingerprint)

    table = jax.lax.cond(can_insert, _do_insert, _no_insert)
    return table, can_insert, HashIdx(index=idx.index * slots_per_bucket + idx.table_index)


def _resolve_slot_conflicts(flat_indices: chex.Array, active: chex.Array) -> chex.Array:
    """Resolve slot contention within a batch using sorting.

    Returns a boolean mask where True indicates the input won the slot.
    This version avoids allocating table-sized buffers and works with non-static capacity.
    """
    batch_size = flat_indices.shape[0]
    # Use a large value for inactive elements to push them to the end during sort
    sentinel = jnp.uint32(0xFFFFFFFF)
    keys = jnp.where(active, flat_indices, sentinel)
    indices = jnp.arange(batch_size, dtype=jnp.uint32)

    # Sort indices by their target flat_index
    sorted_keys, perm = jax.lax.sort_key_val(keys, indices)

    # A slot is won by the first occurrence in the sorted list
    is_first = jnp.concatenate([jnp.array([True]), sorted_keys[1:] != sorted_keys[:-1]], axis=0)
    is_valid = sorted_keys != sentinel
    winners_in_sorted = is_first & is_valid

    # Scatter the results back to original batch order
    return jnp.zeros((batch_size,), dtype=jnp.bool_).at[perm].set(winners_in_sorted)


def _hashtable_parallel_insert_internal(
    table: "HashTable",
    slots_per_bucket: int,
    inputs: Xtructurable,
    inputs_uint32ed: chex.Array,
    probe_steps: chex.Array,
    index: TableIdx,
    updatable: chex.Array,
    fingerprints: chex.Array,
) -> tuple["HashTable", TableIdx, chex.Array]:
    del inputs_uint32ed
    capacity = jnp.asarray(table._capacity, dtype=SIZE_DTYPE)
    probe_steps = jnp.asarray(probe_steps, dtype=SIZE_DTYPE)
    max_probes = jnp.asarray(table.max_probes, dtype=SIZE_DTYPE)

    def _advance(idx: TableIdx, step: chex.Array) -> TableIdx:
        next_table = idx.table_index >= (slots_per_bucket - 1)

        def _next_bucket() -> TableIdx:
            next_index = (idx.index + step) & (capacity - 1)
            bucket_fill = table.table_idx[next_index]
            return TableIdx(
                index=SIZE_DTYPE(next_index),
                table_index=HASH_TABLE_IDX_DTYPE(bucket_fill),
            )

        def _same_bucket() -> TableIdx:
            return TableIdx(
                index=idx.index,
                table_index=HASH_TABLE_IDX_DTYPE(idx.table_index + 1),
            )

        return jax.lax.cond(next_table, _next_bucket, _same_bucket)

    def _next_idx(idxs: TableIdx, should_advance: chex.Array) -> TableIdx:
        return jax.vmap(
            lambda active, current_idx, step: jax.lax.cond(
                active,
                lambda: _advance(current_idx, step),
                lambda: current_idx,
            )
        )(should_advance, idxs, probe_steps)

    flat_initial_slots = index.index * slots_per_bucket + index.table_index
    initial_unique_mask = _resolve_slot_conflicts(flat_initial_slots, updatable)
    pending = jnp.logical_and(updatable, jnp.logical_not(initial_unique_mask))
    probe_counts = jnp.zeros_like(updatable, dtype=SIZE_DTYPE)
    failed = jnp.zeros_like(updatable, dtype=jnp.bool_)

    def _cond(val: tuple[TableIdx, chex.Array, chex.Array, chex.Array]) -> bool:
        _, pending, probe_counts, failed = val
        active_pending = jnp.logical_and(
            pending, jnp.logical_and(~failed, probe_counts < max_probes)
        )
        return jnp.any(active_pending)

    def _body(
        val: tuple[TableIdx, chex.Array, chex.Array, chex.Array]
    ) -> tuple[TableIdx, chex.Array, chex.Array, chex.Array]:
        idxs, pending, probe_counts, failed = val
        active_pending = jnp.logical_and(
            pending, jnp.logical_and(~failed, probe_counts < max_probes)
        )
        updated_idxs = _next_idx(idxs, active_pending)
        new_probe_counts = probe_counts + active_pending.astype(SIZE_DTYPE)

        flat_updated_slots = updated_idxs.index * slots_per_bucket + updated_idxs.table_index
        valid_candidate = updated_idxs.table_index < slots_per_bucket
        active_for_unique = jnp.logical_and(updatable, jnp.logical_and(~failed, valid_candidate))
        updated_unique_mask = _resolve_slot_conflicts(flat_updated_slots, active_for_unique)

        tentative_pending = jnp.logical_and(
            updatable, jnp.logical_and(~failed, ~updated_unique_mask)
        )
        over_budget = new_probe_counts >= max_probes
        exhausted = jnp.logical_and(tentative_pending, over_budget)
        next_failed = jnp.logical_or(failed, exhausted)
        next_pending = jnp.logical_and(tentative_pending, ~over_budget)
        return updated_idxs, next_pending, new_probe_counts, next_failed

    index, pending, _, failed = jax.lax.while_loop(
        _cond, _body, (index, pending, probe_counts, failed)
    )
    failed = jnp.logical_or(failed, pending)
    successful = jnp.logical_and(
        updatable, jnp.logical_and(~failed, index.table_index < slots_per_bucket)
    )
    flat_indices = index.index * slots_per_bucket + index.table_index

    new_table = table.table.at[flat_indices].set_as_condition(successful, inputs)

    new_fingerprints = _update_array_on_condition(
        table.fingerprints,
        flat_indices,
        successful,
        fingerprints.astype(jnp.uint32),
    )
    new_table_idx = table.table_idx.at[index.index].add(successful)
    new_size = table.size + jnp.sum(successful, dtype=SIZE_DTYPE)

    table = table.replace(
        table=new_table, fingerprints=new_fingerprints, table_idx=new_table_idx, size=new_size
    )
    return table, index, successful


@partial(jax.jit, static_argnums=(1,))
def _hashtable_parallel_insert_jit(
    table: "HashTable",
    slots_per_bucket: int,
    inputs: Xtructurable,
    filled: chex.Array | bool = None,
    unique_key: chex.Array = None,
):
    if filled is None:
        filled = jnp.ones(inputs.shape.batch, dtype=jnp.bool_)

    filled = jnp.asarray(filled)
    batch_len = inputs.shape.batch

    def _process_insert(filled_mask):
        # Get initial indices, probe steps, and byte representations
        initial_idx, steps, uint32eds, fingerprints = jax.vmap(
            get_new_idx_byterized, in_axes=(0, None, None)
        )(inputs, table._capacity, table.seed)

        unique_filled, representative_indices = _compute_unique_mask_from_uint32eds(
            uint32eds=uint32eds,
            filled=filled_mask,
            unique_key=unique_key,
        )

        # Look up each state
        idx = TableIdx(
            index=initial_idx, table_index=jnp.zeros(batch_len, dtype=HASH_TABLE_IDX_DTYPE)
        )

        initial_found = jnp.logical_not(unique_filled)
        idx, found = _hashtable_lookup_parallel_internal(
            table, slots_per_bucket, inputs, uint32eds, idx, steps, fingerprints, initial_found
        )

        empty_slot = idx.table_index >= table.table_idx[idx.index]
        updatable = jnp.logical_and(~found, jnp.logical_and(unique_filled, empty_slot))

        # Perform parallel insertion
        updated_table, inserted_idx, inserted_mask = _hashtable_parallel_insert_internal(
            table, slots_per_bucket, inputs, uint32eds, steps, idx, updatable, fingerprints
        )

        # Provisional index selection
        cond_found = jnp.asarray(found, dtype=jnp.bool_)

        inserted_index = jnp.asarray(inserted_idx.index, dtype=idx.index.dtype)
        inserted_table_index = jnp.asarray(inserted_idx.table_index, dtype=idx.table_index.dtype)
        current_index = jnp.asarray(idx.index)
        current_table_index = jnp.asarray(idx.table_index)

        provisional_index = _where_no_broadcast(
            cond_found,
            current_index,
            inserted_index,
        )
        provisional_table_index = _where_no_broadcast(
            cond_found,
            current_table_index,
            inserted_table_index,
        )
        provisional_idx = TableIdx(index=provisional_index, table_index=provisional_table_index)

        representative_indices = jnp.asarray(representative_indices, dtype=jnp.int32)
        final_idx = TableIdx(
            index=provisional_idx.index[representative_indices],
            table_index=provisional_idx.table_index[representative_indices],
        )

        return (
            updated_table,
            inserted_mask,
            unique_filled,
            HashIdx(index=final_idx.index * slots_per_bucket + final_idx.table_index),
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


@jax.jit
def _hashtable_getitem_jit(table, idx: HashIdx) -> Xtructurable:
    return table.table[idx.index]


@dataclasses.dataclass(frozen=True)
class HashTable:
    """
    Bucketed open-addressing hash table.

    Attributes:
        seed: Initial seed for hash functions
        capacity: User-specified capacity
        _capacity: Actual internal capacity (larger than specified to handle collisions)
        slots_per_bucket: Number of slots in each bucket (Horizontal scan size)
        max_probes: Per-lookup probe upper bound
        size: Current number of items in table
        table: The actual storage for states
        table_idx: Per-bucket fill count (0..slots_per_bucket)
        fingerprints: Per-slot 32-bit fingerprints
    """

    seed: int
    capacity: int
    _capacity: int
    slots_per_bucket: int
    max_probes: int
    size: int
    table: Xtructurable  # shape = State("args" = (capacity, slots_per_bucket, ...), ...)
    table_idx: chex.Array  # shape = (capacity, ) is the index of the table in the bucket table.
    fingerprints: chex.Array  # shape = (_capacity * slots_per_bucket,)

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

    @staticmethod
    def build(
        dataclass: Xtructurable,
        seed: int,
        capacity: int,
        slots_per_bucket: int = 32,
        hash_size_multiplier: int = 2,
        max_probes: int | None = None,
    ) -> "HashTable":
        """
        Initialize a new hash table with specified parameters.

        Args:
            dataclass: Example Xtructurable to determine the structure
            seed: Initial seed for hash functions
            capacity: Desired capacity of the table

        Returns:
            Initialized HashTable instance
        """
        return _hashtable_build(
            dataclass,
            seed,
            capacity,
            slots_per_bucket,
            hash_size_multiplier,
            max_probes,
        )

    def lookup_slot(self, input: Xtructurable) -> tuple[TableIdx, bool, chex.Array]:
        """
        Finds the state in the hash table using bucketed probing.

        Args:
            input: The Xtructurable state to look up.

        Returns:
            A tuple (idx, found, fingerprint):
            - idx (TableIdx): Index information for the slot examined.
            - found (bool): True if the state was found, False otherwise.
            - fingerprint (uint32): Hash fingerprint of the probed state (internal use).
            If not found, idx indicates the first empty slot encountered
            during the probe path where an insertion could occur.
        """
        return _hashtable_lookup_slot_jit(self, self.slots_per_bucket, input)

    def lookup(self, input: Xtructurable) -> tuple[HashIdx, bool]:
        """
        Find a state in the hash table.

        Returns a tuple of `(HashIdx, found)` where `HashIdx.index` is the flat
        index into `table.table`, and `found` indicates existence.
        """
        return _hashtable_lookup_jit(self, self.slots_per_bucket, input)

    def lookup_parallel(
        self, inputs: Xtructurable, filled: chex.Array | bool = True
    ) -> tuple[HashIdx, chex.Array]:
        """
        Finds the state in the hash table using bucketed probing.

        Returns `(HashIdx, found_mask)` per input.
        """
        return _hashtable_lookup_parallel_jit(self, self.slots_per_bucket, inputs, filled)

    def insert(self, input: Xtructurable) -> tuple["HashTable", bool, HashIdx]:
        """
        insert the state in the table

        Returns `(table, inserted?, flat_idx)`.
        """
        return _hashtable_insert_jit(self, self.slots_per_bucket, input)

    def parallel_insert(
        self,
        inputs: Xtructurable,
        filled: chex.Array | bool = None,
        unique_key: chex.Array = None,
    ):
        """
        Parallel insertion of multiple states into the hash table.

        Args:
            inputs: States to insert
            filled: Boolean array indicating which inputs are valid
            unique_key: Optional key array for determining priority among duplicate states.
                       When provided, among duplicate states, only the one with the smallest
                       key value will be marked as unique in unique_filled mask.

        Returns:
            Tuple of (updated_table, updatable, unique_filled, idx)
        """
        return _hashtable_parallel_insert_jit(
            self, self.slots_per_bucket, inputs, filled, unique_key
        )

    def __getitem__(self, idx: HashIdx) -> Xtructurable:
        return _hashtable_getitem_jit(self, idx)


def _hashtable_flatten(table: HashTable):
    # Static metadata goes to aux_data, JAX arrays go to children
    children = (table.size, table.table, table.table_idx, table.fingerprints)
    aux_data = (
        table.seed,
        table.capacity,
        table._capacity,
        table.slots_per_bucket,
        table.max_probes,
    )
    return children, aux_data


def _hashtable_unflatten(aux_data, children):
    seed, capacity, _capacity, slots_per_bucket, max_probes = aux_data
    size, table, table_idx, fingerprints = children
    return HashTable(
        seed=seed,
        capacity=capacity,
        _capacity=_capacity,
        slots_per_bucket=slots_per_bucket,
        max_probes=max_probes,
        size=size,
        table=table,
        table_idx=table_idx,
        fingerprints=fingerprints,
    )


# Override the default registration from base_dataclass to treat metadata as static
jax.tree_util.register_pytree_node(HashTable, _hashtable_flatten, _hashtable_unflatten)
