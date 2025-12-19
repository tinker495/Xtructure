"""
Hash table implementation using Double Hashing with bucketed storage for efficient state storage and lookup.
This module provides functionality for hashing Xtructurables and managing collisions.
"""
from functools import partial
from typing import TypeVar

import chex
import jax
import jax.numpy as jnp

from ..core import FieldDescriptor, Xtructurable, base_dataclass, xtructure_dataclass
from ..core.xtructure_decorators.hash import uint32ed_to_hash
from ..core.xtructure_numpy.array_ops import (
    _update_array_on_condition,
    _where_no_broadcast,
)

SIZE_DTYPE = jnp.uint32
SLOT_IDX_DTYPE = jnp.uint8
DOUBLE_HASH_SECONDARY_DELTA = jnp.uint32(0x9E3779B1)
FINGERPRINT_MIX_CONSTANT_A = jnp.uint32(0x85EBCA6B)
FINGERPRINT_MIX_CONSTANT_B = jnp.uint32(0xC2B2AE35)

T = TypeVar("T")


@xtructure_dataclass
class BucketIdx:
    index: FieldDescriptor.scalar(dtype=SIZE_DTYPE)
    slot_index: FieldDescriptor.scalar(dtype=SLOT_IDX_DTYPE)


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
    modulus_u32 = jnp.asarray(modulus, dtype=SIZE_DTYPE)
    modulus_u32 = jnp.maximum(modulus_u32, SIZE_DTYPE(1))
    mask = modulus_u32 - SIZE_DTYPE(1)
    is_pow2 = jnp.logical_and(modulus_u32 > 0, (modulus_u32 & mask) == 0)
    # If modulus is power-of-two, bitmask is equivalent to modulo and typically faster.
    step = jax.lax.select(is_pow2, step & mask, step % modulus_u32)
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
    modulus_u32 = jnp.asarray(modulus, dtype=SIZE_DTYPE)
    modulus_u32 = jnp.maximum(modulus_u32, SIZE_DTYPE(1))
    mask = modulus_u32 - SIZE_DTYPE(1)
    is_pow2 = jnp.logical_and(modulus_u32 > 0, (modulus_u32 & mask) == 0)
    index = jax.lax.select(
        is_pow2, jnp.asarray(primary_hash, dtype=SIZE_DTYPE) & mask, primary_hash % modulus_u32
    )
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
    modulus_u32 = jnp.asarray(modulus, dtype=SIZE_DTYPE)
    modulus_u32 = jnp.maximum(modulus_u32, SIZE_DTYPE(1))
    mask = modulus_u32 - SIZE_DTYPE(1)
    is_pow2 = jnp.logical_and(modulus_u32 > 0, (modulus_u32 & mask) == 0)
    idx = jax.lax.select(
        is_pow2, jnp.asarray(hash_value, dtype=SIZE_DTYPE) & mask, hash_value % modulus_u32
    )
    step = _normalize_probe_step(secondary_hash, modulus)
    length = jnp.uint32(uint32ed.size)
    fingerprint = _mix_fingerprint(hash_value, secondary_hash, length)
    return idx, step, uint32ed, fingerprint


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5))
def _hashtable_build_jit(
    dataclass: Xtructurable,
    seed: int,
    capacity: int,
    bucket_size: int = 2,
    hash_size_multiplier: int = 2,
    max_probes: int | None = None,
) -> "HashTable":
    # Ensure _capacity is power-of-two for better performance and cycle guarantee
    _target_cap = int(hash_size_multiplier * capacity / bucket_size)
    # Calculate next power of two using bit_length() to ensure a concrete Python int for array shapes
    if _target_cap <= 1:
        _capacity = 1
    else:
        _capacity = 1 << (_target_cap - 1).bit_length()

    size = SIZE_DTYPE(0)
    # Default max_probes to a full-table slot scan if not provided
    if max_probes is None:
        max_probes = _capacity * bucket_size

    # Initialize table with default states
    table = dataclass.default(((_capacity + 1) * bucket_size,))
    bucket_fill_levels = jnp.zeros((_capacity + 1), dtype=SLOT_IDX_DTYPE)
    fingerprints = jnp.zeros(((_capacity + 1) * bucket_size,), dtype=jnp.uint32)
    return HashTable(
        seed=seed,
        capacity=capacity,
        _capacity=_capacity,
        bucket_size=bucket_size,
        size=size,
        table=table,
        bucket_fill_levels=bucket_fill_levels,
        fingerprints=fingerprints,
        max_probes=int(max_probes),
    )


def _hashtable_lookup_internal(
    table: "HashTable",
    input: Xtructurable,
    input_uint32ed: chex.Array,
    idx: BucketIdx,
    probe_step: chex.Array,
    input_fingerprint: chex.Array,
    found: bool,
) -> tuple[BucketIdx, bool]:
    probe_step = jnp.asarray(probe_step, dtype=SIZE_DTYPE)
    bucket_size = int(table.bucket_size)
    max_probes_u32 = SIZE_DTYPE(int(table.max_probes))
    capacity = jnp.asarray(table._capacity, dtype=SIZE_DTYPE)
    mask = capacity - SIZE_DTYPE(1)
    del input_uint32ed

    slot_offsets = jnp.arange(bucket_size, dtype=SLOT_IDX_DTYPE)
    slot_offsets_u32 = slot_offsets.astype(SIZE_DTYPE)
    bucket_size_u32 = SIZE_DTYPE(bucket_size)

    def _cond(val: tuple[BucketIdx, bool, chex.Array]) -> bool:
        _, found, probes = val
        return jnp.logical_and(~found, probes < max_probes_u32)

    def _body(val: tuple[BucketIdx, bool, chex.Array]) -> tuple[BucketIdx, bool, chex.Array]:
        idx, found, probes = val
        bucket_idx = jnp.asarray(idx.index, dtype=SIZE_DTYPE)
        filled_limit = table.bucket_fill_levels[bucket_idx].astype(SLOT_IDX_DTYPE)
        filled_limit_u32 = filled_limit.astype(SIZE_DTYPE)

        flat_indices = bucket_idx * bucket_size_u32 + slot_offsets_u32
        stored_fps = table.fingerprints[flat_indices]

        is_filled = slot_offsets < filled_limit
        fp_matches = jnp.logical_and(is_filled, stored_fps == input_fingerprint)
        has_fp_match = jnp.any(fp_matches)
        first_match_slot = jnp.argmax(fp_matches).astype(SLOT_IDX_DTYPE)  # 0 if none
        candidate_flat = bucket_idx * bucket_size_u32 + first_match_slot.astype(SIZE_DTYPE)

        def _compare_first() -> chex.Array:
            return jnp.asarray(table.table[candidate_flat] == input, dtype=jnp.bool_)

        found_fast = jnp.logical_and(
            has_fp_match, jax.lax.cond(has_fp_match, _compare_first, lambda: jnp.bool_(False))
        )

        other_fp_matches = jnp.logical_and(fp_matches, slot_offsets != first_match_slot)
        need_fallback = jnp.logical_and(~found_fast, jnp.any(other_fp_matches))

        def _fallback_scan() -> tuple[chex.Array, chex.Array]:
            def _check_slot(off: chex.Array, match: chex.Array) -> chex.Array:
                flat = bucket_idx * bucket_size_u32 + off.astype(SIZE_DTYPE)
                return jax.lax.cond(
                    match,
                    lambda: jnp.asarray(table.table[flat] == input, dtype=jnp.bool_),
                    lambda: jnp.bool_(False),
                )

            slot_equals = jax.vmap(_check_slot)(slot_offsets, other_fp_matches)
            found_in_bucket = jnp.any(slot_equals)
            match_slot = jnp.argmax(slot_equals).astype(SLOT_IDX_DTYPE)
            return found_in_bucket, match_slot

        found_fb, match_fb = jax.lax.cond(
            need_fallback,
            _fallback_scan,
            lambda: (jnp.bool_(False), SLOT_IDX_DTYPE(0)),
        )

        new_found = jnp.logical_or(found_fast, found_fb)
        match_slot = jnp.where(found_fast, first_match_slot, match_fb).astype(SLOT_IDX_DTYPE)

        bucket_full = filled_limit_u32 == bucket_size_u32
        should_stop = jnp.logical_or(new_found, jnp.logical_not(bucket_full))

        out_slot = jnp.where(new_found, match_slot, filled_limit).astype(SLOT_IDX_DTYPE)
        next_bucket = (bucket_idx + probe_step) & mask

        updated_idx = BucketIdx(
            index=jnp.where(should_stop, bucket_idx, next_bucket).astype(SIZE_DTYPE),
            slot_index=jnp.where(should_stop, out_slot, SLOT_IDX_DTYPE(0)),
        )

        probes = probes + bucket_size_u32
        probes = jnp.where(should_stop, max_probes_u32, probes)
        return updated_idx, new_found, probes

    # Start loop with current idx. Initial probes set to 0.
    idx, found, _ = jax.lax.while_loop(_cond, _body, (idx, found, jnp.uint32(0)))
    return idx, found


@jax.jit
def _hashtable_lookup_bucket_jit(
    table: "HashTable", input: Xtructurable
) -> tuple[BucketIdx, bool, chex.Array]:
    index, step, input_uint32ed, fingerprint = get_new_idx_byterized(
        input, table._capacity, table.seed
    )
    idx = BucketIdx(index=index, slot_index=SLOT_IDX_DTYPE(0))
    idx, found = _hashtable_lookup_internal(
        table, input, input_uint32ed, idx, step, fingerprint, False
    )
    return idx, found, fingerprint


@jax.jit
def _hashtable_lookup_jit(table: "HashTable", input: Xtructurable) -> tuple[HashIdx, bool]:
    idx, found, _ = _hashtable_lookup_bucket_jit(table, input)
    bucket_size_u32 = SIZE_DTYPE(table.bucket_size)
    return HashIdx(index=idx.index * bucket_size_u32 + idx.slot_index), found


def _hashtable_lookup_parallel_internal(
    table: "HashTable",
    inputs: Xtructurable,
    input_uint32eds: chex.Array,
    idxs: BucketIdx,
    probe_steps: chex.Array,
    fingerprints: chex.Array,
    founds: chex.Array,
    active: chex.Array | None = None,
) -> tuple[BucketIdx, chex.Array]:
    # Lockstep batch probing (single while-loop over vector state), with horizontal bucket scan.
    if active is None:
        active = jnp.ones(inputs.shape.batch, dtype=jnp.bool_)

    del input_uint32eds

    bucket_size = int(table.bucket_size)
    max_probes_u32 = SIZE_DTYPE(int(table.max_probes))
    capacity = jnp.asarray(table._capacity, dtype=SIZE_DTYPE)
    mask = capacity - SIZE_DTYPE(1)
    probe_steps = jnp.asarray(probe_steps, dtype=SIZE_DTYPE)
    bucket_size_u32 = SIZE_DTYPE(bucket_size)
    slot_offsets = jnp.arange(bucket_size, dtype=SLOT_IDX_DTYPE)
    slot_offsets_u32 = slot_offsets.astype(SIZE_DTYPE)

    probes = jnp.zeros(inputs.shape.batch, dtype=SIZE_DTYPE)

    def _cond(val: tuple[BucketIdx, chex.Array, chex.Array, chex.Array]) -> chex.Array:
        _, _, _, active = val
        return jnp.any(active)

    def _body(
        val: tuple[BucketIdx, chex.Array, chex.Array, chex.Array]
    ) -> tuple[BucketIdx, chex.Array, chex.Array, chex.Array]:
        idxs, founds, probes, active = val

        bucket_indices = jnp.asarray(idxs.index, dtype=SIZE_DTYPE)
        filled_limits = table.bucket_fill_levels[bucket_indices].astype(SLOT_IDX_DTYPE)
        filled_limits_u32 = filled_limits.astype(SIZE_DTYPE)
        under_limit = probes < max_probes_u32

        # All active rows scan one bucket per iteration.
        step_active = jnp.logical_and(active, jnp.logical_and(~founds, under_limit))

        # Horizontal scan: gather all fingerprints in current buckets.
        flat_indices = bucket_indices[:, None] * bucket_size_u32 + slot_offsets_u32[None, :]
        stored_fps = table.fingerprints[flat_indices]

        is_filled = slot_offsets[None, :] < filled_limits[:, None]
        fp_matches = jnp.logical_and(
            jnp.logical_and(is_filled, stored_fps == fingerprints[:, None]),
            step_active[:, None],
        )

        has_fp_match_row = jnp.any(fp_matches, axis=1)
        first_match_slot = jnp.argmax(fp_matches, axis=1).astype(SLOT_IDX_DTYPE)  # 0 if none
        candidate_flat = bucket_indices * bucket_size_u32 + first_match_slot.astype(SIZE_DTYPE)

        def _maybe_compare(
            has_match: chex.Array, flat: chex.Array, inp: Xtructurable
        ) -> chex.Array:
            return jax.lax.cond(
                has_match,
                lambda: jnp.asarray(table.table[flat] == inp, dtype=jnp.bool_),
                lambda: jnp.bool_(False),
            )

        found_fast = jax.vmap(_maybe_compare)(
            jnp.logical_and(step_active, has_fp_match_row), candidate_flat, inputs
        )

        other_fp_matches = jnp.logical_and(
            fp_matches,
            slot_offsets[None, :] != first_match_slot[:, None],
        )
        need_fallback = jnp.logical_and(~found_fast, jnp.any(other_fp_matches, axis=1))
        any_need_fallback = jnp.any(need_fallback)

        def _fallback_scan():
            # Rare path: gather full bucket values only if any row needs it.
            bucket_vals = table.table[flat_indices]
            value_equals = jax.vmap(lambda bv, inp: jax.vmap(lambda s: s == inp)(bv))(
                bucket_vals, inputs
            )
            match_found = jnp.logical_and(value_equals, other_fp_matches)
            match_found = jnp.logical_and(match_found, need_fallback[:, None])
            found_fb = jnp.any(match_found, axis=1)
            idx_fb = jnp.argmax(match_found, axis=1).astype(SLOT_IDX_DTYPE)
            return found_fb, idx_fb

        found_fb, idx_fb = jax.lax.cond(
            any_need_fallback,
            _fallback_scan,
            lambda: (
                jnp.zeros_like(need_fallback, dtype=jnp.bool_),
                jnp.zeros_like(first_match_slot, dtype=SLOT_IDX_DTYPE),
            ),
        )

        new_founds_in_bucket = jnp.logical_or(found_fast, found_fb)
        founds = jnp.logical_or(founds, new_founds_in_bucket)

        match_slot = jnp.where(found_fast, first_match_slot, idx_fb).astype(SLOT_IDX_DTYPE)

        bucket_full = filled_limits_u32 == bucket_size_u32

        out_slot = jnp.where(new_founds_in_bucket, match_slot, filled_limits).astype(SLOT_IDX_DTYPE)

        # Budget/progress
        probes = probes + step_active.astype(SIZE_DTYPE) * bucket_size_u32
        under_limit_next = probes < max_probes_u32

        # Continue only if still not found, bucket full, and under budget.
        continue_mask = jnp.logical_and(
            step_active,
            jnp.logical_and(~new_founds_in_bucket, jnp.logical_and(bucket_full, under_limit_next)),
        )

        next_bucket_indices = (bucket_indices + probe_steps) & mask
        updated_idxs = BucketIdx(
            index=jnp.where(continue_mask, next_bucket_indices, bucket_indices).astype(SIZE_DTYPE),
            slot_index=jnp.where(continue_mask, SLOT_IDX_DTYPE(0), out_slot),
        )
        # IMPORTANT: only update indices for rows that actually participated this iteration.
        # Otherwise, already-found rows can get overwritten while other rows continue looping.
        idxs = BucketIdx(
            index=jnp.where(step_active, updated_idxs.index, idxs.index),
            slot_index=jnp.where(step_active, updated_idxs.slot_index, idxs.slot_index),
        )

        # Update active: remain active only for those continuing.
        active = continue_mask
        return idxs, founds, probes, active

    idxs, founds, _, _ = jax.lax.while_loop(_cond, _body, (idxs, founds, probes, active))
    return idxs, founds


def _resolve_slot_conflicts(flat_indices: chex.Array, active: chex.Array) -> chex.Array:
    """Pick at most one 'winner' per flat slot among a batch.

    Deterministic: winner is the smallest batch index among contenders for the same slot.
    Uses sorting (no table-sized buffers, avoids jnp.unique in inner loops).
    """
    active = jnp.asarray(active, dtype=jnp.bool_)
    batch_size = flat_indices.shape[0]
    flat_indices = jnp.asarray(flat_indices, dtype=jnp.uint32)

    # We intentionally avoid uint64 here because many JAX installs default to x64 disabled.
    # Stable sort by flat_index, using original batch order as deterministic tie-breaker.
    batch_idx = jnp.arange(batch_size, dtype=jnp.uint32)
    sentinel = jnp.uint32(0xFFFFFFFF)
    keys = jnp.where(active, flat_indices, sentinel)

    # Stable sort guarantees that for equal keys, earlier batch_idx stays earlier -> min batch wins.
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
def _hashtable_lookup_parallel_jit(
    table: "HashTable", inputs: Xtructurable, filled: chex.Array | bool = True
) -> tuple[HashIdx, chex.Array]:
    filled = jnp.asarray(filled)
    batch_size = inputs.shape.batch

    def _process_batch(filled_mask):
        initial_idx, steps, input_uint32eds, fingerprints = jax.vmap(
            get_new_idx_byterized, in_axes=(0, None, None)
        )(inputs, table._capacity, table.seed)

        idxs = BucketIdx(index=initial_idx, slot_index=jnp.zeros(batch_size, dtype=SLOT_IDX_DTYPE))
        founds = jnp.zeros(batch_size, dtype=jnp.bool_)

        idx, found = _hashtable_lookup_parallel_internal(
            table, inputs, input_uint32eds, idxs, steps, fingerprints, founds, filled_mask
        )
        bucket_size_u32 = SIZE_DTYPE(table.bucket_size)
        return HashIdx(index=idx.index * bucket_size_u32 + idx.slot_index), found

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

    # Only insert if not found AND we actually landed on an empty slot (didn't hit max_probes)
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

        # Perform parallel insertion
        updated_table, inserted_idx = _hashtable_parallel_insert_internal(
            table, inputs, uint32eds, steps, idx, updatable, fingerprints
        )

        # Provisional index selection
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


@jax.jit
def _hashtable_getitem_jit(table, idx: HashIdx) -> Xtructurable:
    return table.table[idx.index]


@base_dataclass(
    frozen=True, static_fields=("seed", "capacity", "_capacity", "bucket_size", "max_probes")
)
class HashTable:
    """
    Bucketed Double Hash Table Implementation

    This implementation uses Double Hashing with bucketed storage to resolve collisions.
    Each item is hashed to a bucket, and if the bucket is full, a secondary hash
    is used to determine the step size for probing the next bucket.

    Attributes:
        seed: Initial seed for hash functions
        capacity: User-specified capacity
        _capacity: Actual internal bucket capacity
        bucket_size: Number of slots per bucket
        size: Current number of items in table
        table: The actual storage for states
        bucket_fill_levels: Number of filled slots in each bucket
        fingerprints: Hash fingerprints for fast equality checks
    """

    seed: int
    capacity: int
    _capacity: int
    bucket_size: int
    size: int
    table: Xtructurable  # shape = State("args" = (_capacity * bucket_size, ...), ...)
    bucket_fill_levels: chex.Array  # shape = (_capacity, ) Number of filled slots per bucket
    fingerprints: chex.Array  # shape = ((_capacity + 1) * bucket_size,)
    max_probes: int

    @staticmethod
    def build(
        dataclass: Xtructurable,
        seed: int,
        capacity: int,
        bucket_size: int = 8,
        hash_size_multiplier: int = 2,
        max_probes: int | None = None,
    ) -> "HashTable":
        """
        Initialize a new hash table with specified parameters.

        Args:
            dataclass: Example Xtructurable to determine the structure
            seed: Initial seed for hash functions
            capacity: Desired capacity of the table
            bucket_size: Number of slots per bucket
            hash_size_multiplier: Multiplier for internal table size
            max_probes: Maximum number of probes for lookup/insert

        Returns:
            Initialized HashTable instance
        """
        return _hashtable_build_jit(
            dataclass, seed, capacity, bucket_size, hash_size_multiplier, max_probes
        )

    def lookup_bucket(self, input: Xtructurable) -> tuple[BucketIdx, bool, chex.Array]:
        """
        Finds the state in the hash table using bucketed double hashing.

        Args:
            input: The Xtructurable state to look up.

        Returns:
            A tuple (idx, found, fingerprint):
            - idx (BucketIdx): Index information (bucket and slot) for the slot examined.
            - found (bool): True if the state was found, False otherwise.
            - fingerprint (uint32): Hash fingerprint of the probed state.
            If not found, idx indicates the first empty slot encountered.
        """
        return _hashtable_lookup_bucket_jit(self, input)

    def lookup(self, input: Xtructurable) -> tuple[HashIdx, bool]:
        """
        Find a state in the hash table.

        Returns a tuple of `(HashIdx, found)` where `HashIdx.index` is the flat
        index into `table.table`, and `found` indicates existence.
        """
        return _hashtable_lookup_jit(self, input)

    def lookup_parallel(
        self, inputs: Xtructurable, filled: chex.Array | bool = True
    ) -> tuple[HashIdx, chex.Array]:
        """
        Finds states in the hash table in parallel.

        Returns `(HashIdx, found_mask)` per input.
        """
        return _hashtable_lookup_parallel_jit(self, inputs, filled)

    def insert(self, input: Xtructurable) -> tuple["HashTable", bool, HashIdx]:
        """
        Insert a state into the table.

        Returns `(table, inserted?, flat_idx)`.
        """
        return _hashtable_insert_jit(self, input)

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
        return _hashtable_parallel_insert_jit(self, inputs, filled, unique_key)

    def __getitem__(self, idx: HashIdx) -> Xtructurable:
        return _hashtable_getitem_jit(self, idx)
