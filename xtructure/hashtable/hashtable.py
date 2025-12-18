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
    modulus = jnp.asarray(modulus, dtype=SIZE_DTYPE)
    step = step % modulus
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
    index = primary_hash % modulus
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
    idx = hash_value % modulus
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
    # Default max_probes to _capacity if not provided
    if max_probes is None:
        max_probes = _capacity

    _max_probes = jnp.array(max_probes, dtype=SIZE_DTYPE)

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
        max_probes=_max_probes,
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
    capacity = jnp.asarray(table._capacity, dtype=SIZE_DTYPE)

    def _advance(idx: BucketIdx) -> BucketIdx:
        next_bucket = idx.slot_index >= (table.bucket_size - 1)

        def _next_bucket():
            next_index = jnp.mod(idx.index + probe_step, capacity)
            return BucketIdx(
                index=SIZE_DTYPE(next_index),
                slot_index=SLOT_IDX_DTYPE(0),
            )

        def _same_bucket():
            return BucketIdx(
                index=idx.index,
                slot_index=SLOT_IDX_DTYPE(idx.slot_index + 1),
            )

        return jax.lax.cond(next_bucket, _next_bucket, _same_bucket)

    def _cond(val: tuple[BucketIdx, bool, chex.Array]) -> bool:
        idx, found, probes = val
        filled_slots = table.bucket_fill_levels[idx.index]
        in_empty = idx.slot_index >= filled_slots
        # Stop if found, or encountered empty slot, or exceeded max_probes
        return jnp.logical_and(~found, jnp.logical_and(~in_empty, probes < table.max_probes))

    def _body(val: tuple[BucketIdx, bool, chex.Array]) -> tuple[BucketIdx, bool, chex.Array]:
        idx, found, probes = val
        flat_index = idx.index * table.bucket_size + idx.slot_index

        # Fingerprint-first check: only load state if fingerprint matches
        stored_fp = table.fingerprints[flat_index]
        is_filled = idx.slot_index < table.bucket_fill_levels[idx.index]
        fingerprints_match = jnp.logical_and(is_filled, stored_fp == input_fingerprint)

        def _compare_value(_: None) -> jnp.bool_:
            state = table.table[flat_index]
            return jnp.asarray(state == input, dtype=jnp.bool_)

        matched = jax.lax.cond(
            fingerprints_match, _compare_value, lambda _: jnp.bool_(False), operand=None
        )

        new_found = jnp.logical_or(found, matched)
        next_idx = _advance(idx)

        updated_idx = BucketIdx(
            index=jnp.where(new_found, idx.index, next_idx.index),
            slot_index=jnp.where(new_found, idx.slot_index, next_idx.slot_index),
        )
        return updated_idx, new_found, probes + 1

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
    return HashIdx(index=idx.index * table.bucket_size + idx.slot_index), found


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
    if active is None:
        active = jnp.ones(inputs.shape.batch, dtype=jnp.bool_)

    def _lu(
        table: "HashTable",
        input: Xtructurable,
        input_uint32ed: chex.Array,
        idx: BucketIdx,
        probe_step: chex.Array,
        fingerprint: chex.Array,
        found: bool,
        active: bool,
    ) -> tuple[BucketIdx, bool]:
        probe_step = jnp.asarray(probe_step, dtype=SIZE_DTYPE)
        capacity = jnp.asarray(table._capacity, dtype=SIZE_DTYPE)

        def _advance(idx: BucketIdx) -> BucketIdx:
            next_bucket = idx.slot_index >= (table.bucket_size - 1)

            def _next_bucket():
                next_index = jnp.mod(idx.index + probe_step, capacity)
                return BucketIdx(
                    index=SIZE_DTYPE(next_index),
                    slot_index=SLOT_IDX_DTYPE(0),
                )

            def _same_bucket():
                return BucketIdx(
                    index=idx.index,
                    slot_index=SLOT_IDX_DTYPE(idx.slot_index + 1),
                )

            return jax.lax.cond(next_bucket, _next_bucket, _same_bucket)

        def _cond(val: tuple[BucketIdx, bool, chex.Array]) -> bool:
            idx, found, probes = val
            filled_slots = table.bucket_fill_levels[idx.index]
            in_empty = idx.slot_index >= filled_slots
            under_limit = probes < table.max_probes
            return jnp.logical_and(
                active, jnp.logical_and(~found, jnp.logical_and(~in_empty, under_limit))
            )

        def _body(val: tuple[BucketIdx, bool, chex.Array]) -> tuple[BucketIdx, bool, chex.Array]:
            idx, found, probes = val
            flat_index = idx.index * table.bucket_size + idx.slot_index

            # Fingerprint-first check
            stored_fp = table.fingerprints[flat_index]
            is_filled = idx.slot_index < table.bucket_fill_levels[idx.index]
            fingerprints_match = jnp.logical_and(is_filled, stored_fp == fingerprint)

            def _compare_value(_: None) -> jnp.bool_:
                state = table.table[flat_index]
                return jnp.asarray(state == input, dtype=jnp.bool_)

            matched = jax.lax.cond(
                fingerprints_match, _compare_value, lambda _: jnp.bool_(False), operand=None
            )

            new_found = jnp.logical_or(found, matched)
            next_idx = _advance(idx)

            updated_idx = BucketIdx(
                index=jnp.where(new_found, idx.index, next_idx.index),
                slot_index=jnp.where(new_found, idx.slot_index, next_idx.slot_index),
            )
            return updated_idx, new_found, probes + 1

        # Start loop with current idx. Initial probes set to 0.
        idx, found, _ = jax.lax.while_loop(
            _cond, _body, (idx, jnp.logical_and(found, active), jnp.uint32(0))
        )
        return idx, found

    table_in_axes = jax.tree_util.tree_map(lambda _: None, table)
    idxs, founds = jax.vmap(_lu, in_axes=(table_in_axes, 0, 0, 0, 0, 0, 0, 0))(
        table, inputs, input_uint32eds, idxs, probe_steps, fingerprints, founds, active
    )
    return idxs, founds


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
        return HashIdx(index=idx.index * table.bucket_size + idx.slot_index), found

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
    return table, inserted, HashIdx(index=idx.index * table.bucket_size + idx.slot_index)


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
            next_index = jnp.mod(idx.index + step, capacity)
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

    flat_initial_slots = index.index * table.bucket_size + index.slot_index
    sentinel_slot = SIZE_DTYPE(table._capacity * table.bucket_size + 1)
    initial_unique_mask = _first_occurrence_mask(flat_initial_slots, updatable, sentinel_slot)
    unupdated = jnp.logical_and(updatable, jnp.logical_not(initial_unique_mask))

    def _cond(val: tuple[BucketIdx, chex.Array, chex.Array]) -> bool:
        _, pending, probes = val
        return jnp.logical_and(jnp.any(pending), probes < table.max_probes)

    def _body(
        val: tuple[BucketIdx, chex.Array, chex.Array]
    ) -> tuple[BucketIdx, chex.Array, chex.Array]:
        idxs, pending, probes = val
        updated_idxs = _next_idx(idxs, pending)
        overflowed = jnp.logical_and(updated_idxs.slot_index >= table.bucket_size, pending)
        flat_updated_slots = updated_idxs.index * table.bucket_size + updated_idxs.slot_index
        updated_unique_mask = _first_occurrence_mask(flat_updated_slots, updatable, sentinel_slot)
        not_uniques = jnp.logical_not(updated_unique_mask)
        next_pending = jnp.logical_and(updatable, not_uniques)
        next_pending = jnp.logical_or(next_pending, overflowed)
        return updated_idxs, next_pending, probes + 1

    index, pending, _ = jax.lax.while_loop(_cond, _body, (index, unupdated, jnp.uint32(0)))

    successful = jnp.logical_and(updatable, jnp.logical_not(pending))
    flat_indices = index.index * table.bucket_size + index.slot_index

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
            table, inputs, uint32eds, idx, steps, fingerprints, initial_found
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
            HashIdx(index=final_idx.index * table.bucket_size + final_idx.slot_index),
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


@base_dataclass
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
    max_probes: chex.Array  # scalar uint32

    @staticmethod
    def build(
        dataclass: Xtructurable,
        seed: int,
        capacity: int,
        bucket_size: int = 2,
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
