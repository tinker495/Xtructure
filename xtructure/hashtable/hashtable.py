"""
Hash table implementation using Cuckoo hashing technique for efficient state storage and lookup.
This module provides functionality for hashing Xtructurables and managing collisions.
"""

from functools import partial
from typing import TypeVar

import chex
import jax
import jax.numpy as jnp

from ..core import FieldDescriptor, Xtructurable, base_dataclass, xtructure_dataclass
from ..core import xtructure_numpy as xnp
from ..core.xtructure_decorators.hash import uint32ed_to_hash
from ..core.xtructure_numpy.array_ops import _where_no_broadcast

SIZE_DTYPE = jnp.uint32
HASH_TABLE_IDX_DTYPE = jnp.uint8
DOUBLE_HASH_SECONDARY_DELTA = jnp.uint32(0x9E3779B1)

T = TypeVar("T")


@xtructure_dataclass
class CuckooIdx:
    index: FieldDescriptor[SIZE_DTYPE]
    table_index: FieldDescriptor[HASH_TABLE_IDX_DTYPE]


@xtructure_dataclass
class HashIdx:
    index: FieldDescriptor[SIZE_DTYPE]


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
    fingerprint = jnp.asarray(primary_hash, dtype=jnp.uint32)
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
    fingerprint = jnp.asarray(hash_value, dtype=jnp.uint32)
    return idx, step, uint32ed, fingerprint


@base_dataclass
class HashTable:
    """
    Cuckoo Hash Table Implementation

    This implementation uses multiple hash functions (specified by n_table)
    to resolve collisions. Each item can be stored in one of n_table possible positions.

    Attributes:
        seed: Initial seed for hash functions
        capacity: User-specified capacity
        _capacity: Actual internal capacity (larger than specified to handle collisions)
        size: Current number of items in table
        table: The actual storage for states
        table_idx: Indices tracking which hash function was used for each entry
    """

    seed: int
    capacity: int
    _capacity: int
    cuckoo_table_n: int
    size: int
    table: Xtructurable  # shape = State("args" = (capacity, cuckoo_len, ...), ...)
    table_idx: chex.Array  # shape = (capacity, ) is the index of the table in the cuckoo table.
    fingerprints: chex.Array  # shape = ((capacity + 1) * cuckoo_len,)

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
    def build(
        dataclass: Xtructurable,
        seed: int,
        capacity: int,
        cuckoo_table_n: int = 2,
        hash_size_multiplier: int = 2,
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
        _capacity = int(
            hash_size_multiplier * capacity / cuckoo_table_n
        )  # Convert to concrete integer
        size = SIZE_DTYPE(0)
        # Initialize table with default states
        table = dataclass.default(((_capacity + 1) * cuckoo_table_n,))
        table_idx = jnp.zeros((_capacity + 1), dtype=HASH_TABLE_IDX_DTYPE)
        fingerprints = jnp.zeros(((_capacity + 1) * cuckoo_table_n,), dtype=jnp.uint32)
        return HashTable(
            seed=seed,
            capacity=capacity,
            _capacity=_capacity,
            cuckoo_table_n=cuckoo_table_n,
            size=size,
            table=table,
            table_idx=table_idx,
            fingerprints=fingerprints,
        )

    @staticmethod
    def _lookup(
        table: "HashTable",
        input: Xtructurable,
        input_uint32ed: chex.Array,
        idx: CuckooIdx,
        probe_step: chex.Array,
        input_fingerprint: chex.Array,
        found: bool,
    ) -> tuple[CuckooIdx, bool]:
        """
        Internal lookup method that searches for a state in the table.
        Uses cuckoo hashing technique to check multiple possible locations.

        Args:
            table: Hash table instance
            input: State to look up
            input_uint32ed: uint32ed representation of the state to look up
            idx: Initial index to check
            probe_step: Increment used for double hashing
            input_fingerprint: 32-bit fingerprint of the input state
            found: Whether the state has been found

        Returns:
            Tuple of (idx, found)
        """

        probe_step = jnp.asarray(probe_step, dtype=SIZE_DTYPE)
        capacity = jnp.asarray(table._capacity, dtype=SIZE_DTYPE)

        def _advance(idx: CuckooIdx) -> CuckooIdx:
            next_table = idx.table_index >= (table.cuckoo_table_n - 1)

            def _next_bucket():
                next_index = jnp.mod(idx.index + probe_step, capacity)
                return CuckooIdx(
                    index=SIZE_DTYPE(next_index),
                    table_index=HASH_TABLE_IDX_DTYPE(0),
                )

            def _same_bucket():
                return CuckooIdx(
                    index=idx.index,
                    table_index=HASH_TABLE_IDX_DTYPE(idx.table_index + 1),
                )

            return jax.lax.cond(next_table, _next_bucket, _same_bucket)

        def _cond(val: tuple[CuckooIdx, bool]) -> bool:
            idx, found = val
            filled_idx = table.table_idx[idx.index]
            in_empty = idx.table_index >= filled_idx
            return jnp.logical_and(~found, ~in_empty)

        def _body(val: tuple[CuckooIdx, bool]) -> tuple[CuckooIdx, bool]:
            idx, found = val
            flat_index = idx.index * table.cuckoo_table_n + idx.table_index
            state = table.table[flat_index]
            filled_limit = table.table_idx[idx.index]
            is_filled = idx.table_index < filled_limit
            stored_fp = table.fingerprints[flat_index]
            fingerprints_match = jnp.logical_and(is_filled, stored_fp == input_fingerprint)

            def _compare(_: None) -> jnp.bool_:
                return jnp.asarray(state == input, dtype=jnp.bool_)

            value_equal = jax.lax.cond(
                fingerprints_match,
                _compare,
                lambda _: jnp.bool_(False),
                operand=None,
            )

            matched = jnp.logical_and(is_filled, value_equal)
            new_found = jnp.logical_or(found, matched)
            next_idx = _advance(idx)
            updated_index = jnp.where(new_found, idx.index, next_idx.index)
            updated_table_index = jnp.where(new_found, idx.table_index, next_idx.table_index)
            updated_idx = CuckooIdx(
                index=updated_index,
                table_index=updated_table_index,
            )
            return updated_idx, new_found

        flat_index = idx.index * table.cuckoo_table_n + idx.table_index
        state = table.table[flat_index]
        is_filled = idx.table_index < table.table_idx[idx.index]
        stored_fp = table.fingerprints[flat_index]
        fingerprints_match = jnp.logical_and(is_filled, stored_fp == input_fingerprint)

        def _compare_initial(_: None) -> jnp.bool_:
            return jnp.asarray(state == input, dtype=jnp.bool_)

        initial_equal = jax.lax.cond(
            fingerprints_match,
            _compare_initial,
            lambda _: jnp.bool_(False),
            operand=None,
        )

        found = jnp.logical_or(found, initial_equal)
        idx, found = jax.lax.while_loop(_cond, _body, (idx, found))
        return idx, found

    @jax.jit
    def lookup_cuckoo(
        table: "HashTable", input: Xtructurable
    ) -> tuple[CuckooIdx, bool, chex.Array]:
        """
        Finds the state in the hash table using Cuckoo hashing.

        Args:
            table: The HashTable instance.
            input: The Xtructurable state to look up.

        Returns:
            A tuple (idx, found, fingerprint):
            - idx (CuckooIdx): Index information for the slot examined.
            - found (bool): True if the state was found, False otherwise.
            - fingerprint (uint32): Hash fingerprint of the probed state (internal use).
            If not found, idx and table_idx indicate the first empty slot encountered
            during the Cuckoo search path where an insertion could occur.
        """
        index, step, input_uint32ed, fingerprint = get_new_idx_byterized(
            input, table._capacity, table.seed
        )
        idx = CuckooIdx(index=index, table_index=HASH_TABLE_IDX_DTYPE(0))
        idx, found = HashTable._lookup(table, input, input_uint32ed, idx, step, fingerprint, False)
        return idx, found, fingerprint

    @jax.jit
    def lookup(table: "HashTable", input: Xtructurable) -> tuple[HashIdx, bool]:
        """
        Find a state in the hash table.

        Returns a tuple of `(HashIdx, found)` where `HashIdx.index` is the flat
        index into `table.table`, and `found` indicates existence.
        """
        idx, found, _ = HashTable.lookup_cuckoo(table, input)
        return HashIdx(index=idx.index * table.cuckoo_table_n + idx.table_index), found

    @staticmethod
    def _lookup_parallel(
        table: "HashTable",
        inputs: Xtructurable,
        input_uint32eds: chex.Array,
        idxs: CuckooIdx,
        probe_steps: chex.Array,
        fingerprints: chex.Array,
        founds: chex.Array,
    ) -> tuple[CuckooIdx, chex.Array]:
        """
        Internal lookup method that searches for states in the table in parallel.
        Uses cuckoo hashing technique to check multiple possible locations.
        """

        def _lu(
            input: Xtructurable,
            input_uint32ed: chex.Array,
            idx: CuckooIdx,
            probe_step: chex.Array,
            fingerprint: chex.Array,
            found: bool,
        ) -> tuple[CuckooIdx, bool]:
            probe_step = jnp.asarray(probe_step, dtype=SIZE_DTYPE)
            capacity = jnp.asarray(table._capacity, dtype=SIZE_DTYPE)

            def _advance(idx: CuckooIdx) -> CuckooIdx:
                next_table = idx.table_index >= (table.cuckoo_table_n - 1)

                def _next_bucket():
                    next_index = jnp.mod(idx.index + probe_step, capacity)
                    return CuckooIdx(
                        index=SIZE_DTYPE(next_index),
                        table_index=HASH_TABLE_IDX_DTYPE(0),
                    )

                def _same_bucket():
                    return CuckooIdx(
                        index=idx.index,
                        table_index=HASH_TABLE_IDX_DTYPE(idx.table_index + 1),
                    )

                return jax.lax.cond(next_table, _next_bucket, _same_bucket)

            def _cond(val: tuple[CuckooIdx, bool]) -> bool:
                idx, found = val
                filled_idx = table.table_idx[idx.index]
                in_empty = idx.table_index >= filled_idx
                return jnp.logical_and(~found, ~in_empty)

            def _body(val: tuple[CuckooIdx, bool]) -> tuple[CuckooIdx, bool]:
                idx, found = val
                flat_index = idx.index * table.cuckoo_table_n + idx.table_index
                state = table.table[flat_index]
                filled_limit = table.table_idx[idx.index]
                is_filled = idx.table_index < filled_limit
                stored_fp = table.fingerprints[flat_index]
                fingerprints_match = jnp.logical_and(is_filled, stored_fp == fingerprint)

                def _compare(_: None) -> jnp.bool_:
                    return jnp.asarray(state == input, dtype=jnp.bool_)

                value_equal = jax.lax.cond(
                    fingerprints_match,
                    _compare,
                    lambda _: jnp.bool_(False),
                    operand=None,
                )

                matched = jnp.logical_and(is_filled, value_equal)
                new_found = jnp.logical_or(found, matched)
                next_idx = _advance(idx)
                updated_index = jnp.where(new_found, idx.index, next_idx.index)
                updated_table_index = jnp.where(new_found, idx.table_index, next_idx.table_index)
                updated_idx = CuckooIdx(
                    index=updated_index,
                    table_index=updated_table_index,
                )
                return updated_idx, new_found

            flat_index = idx.index * table.cuckoo_table_n + idx.table_index
            state = table.table[flat_index]
            is_filled = idx.table_index < table.table_idx[idx.index]
            stored_fp = table.fingerprints[flat_index]
            fingerprints_match = jnp.logical_and(is_filled, stored_fp == fingerprint)

            def _compare_initial(_: None) -> jnp.bool_:
                return jnp.asarray(state == input, dtype=jnp.bool_)

            initial_equal = jax.lax.cond(
                fingerprints_match,
                _compare_initial,
                lambda _: jnp.bool_(False),
                operand=None,
            )

            found = jnp.logical_or(found, initial_equal)
            idx, found = jax.lax.while_loop(_cond, _body, (idx, found))
            return idx, found

        idxs, founds = jax.vmap(_lu, in_axes=(0, 0, 0, 0, 0, 0))(
            inputs, input_uint32eds, idxs, probe_steps, fingerprints, founds
        )
        return idxs, founds

    @jax.jit
    def lookup_parallel(table: "HashTable", inputs: Xtructurable) -> tuple[HashIdx, chex.Array]:
        """
        Finds the state in the hash table using Cuckoo hashing.

        Returns `(HashIdx, found_mask)` per input.
        """
        initial_idx, steps, input_uint32eds, fingerprints = jax.vmap(
            get_new_idx_byterized, in_axes=(0, None, None)
        )(inputs, table._capacity, table.seed)

        batch_size = inputs.shape.batch

        idxs = CuckooIdx(
            index=initial_idx, table_index=jnp.zeros(batch_size, dtype=HASH_TABLE_IDX_DTYPE)
        )
        founds = jnp.zeros(batch_size, dtype=jnp.bool_)

        idx, found = HashTable._lookup_parallel(
            table, inputs, input_uint32eds, idxs, steps, fingerprints, founds
        )
        return HashIdx(index=idx.index * table.cuckoo_table_n + idx.table_index), found

    @jax.jit
    def insert(table: "HashTable", input: Xtructurable) -> tuple["HashTable", bool, HashIdx]:
        """
        insert the state in the table

        Returns `(table, inserted?, flat_idx)`.
        """

        def _update_table(
            table: "HashTable", input: Xtructurable, idx: CuckooIdx, fingerprint: chex.Array
        ):
            """
            insert the state in the table
            """
            table.table = table.table.at[idx.index * table.cuckoo_table_n + idx.table_index].set(
                input
            )
            flat_index = idx.index * table.cuckoo_table_n + idx.table_index
            table.fingerprints = table.fingerprints.at[flat_index].set(fingerprint)
            table.table_idx = table.table_idx.at[idx.index].add(1)
            return table

        idx, found, fingerprint = HashTable.lookup_cuckoo(table, input)

        def _no_insert():
            return table

        def _do_insert():
            return _update_table(table, input, idx, fingerprint)

        table = jax.lax.cond(found, _no_insert, _do_insert)
        inserted = ~found
        return table, inserted, HashIdx(index=idx.index * table.cuckoo_table_n + idx.table_index)

    @staticmethod
    def _parallel_insert(
        table: "HashTable",
        inputs: Xtructurable,
        inputs_uint32ed: chex.Array,
        probe_steps: chex.Array,
        index: CuckooIdx,
        updatable: chex.Array,
        fingerprints: chex.Array,
    ) -> tuple["HashTable", CuckooIdx]:
        capacity = jnp.asarray(table._capacity, dtype=SIZE_DTYPE)
        probe_steps = jnp.asarray(probe_steps, dtype=SIZE_DTYPE)

        def _advance(idx: CuckooIdx, step: chex.Array) -> CuckooIdx:
            next_table = idx.table_index >= (table.cuckoo_table_n - 1)

            def _next_bucket() -> CuckooIdx:
                next_index = jnp.mod(idx.index + step, capacity)
                bucket_fill = table.table_idx[next_index]
                return CuckooIdx(
                    index=SIZE_DTYPE(next_index),
                    table_index=HASH_TABLE_IDX_DTYPE(bucket_fill),
                )

            def _same_bucket() -> CuckooIdx:
                return CuckooIdx(
                    index=idx.index,
                    table_index=HASH_TABLE_IDX_DTYPE(idx.table_index + 1),
                )

            return jax.lax.cond(next_table, _next_bucket, _same_bucket)

        def _next_idx(idxs: CuckooIdx, unupdateds: chex.Array) -> CuckooIdx:
            return jax.vmap(
                lambda active, current_idx, step: jax.lax.cond(
                    active,
                    lambda: _advance(current_idx, step),
                    lambda: current_idx,
                )
            )(unupdateds, idxs, probe_steps)

        initial_not_uniques = jnp.logical_not(xnp.unique_mask(index, filled=updatable))
        unupdated = jnp.logical_and(updatable, initial_not_uniques)

        def _cond(val: tuple[CuckooIdx, chex.Array]) -> bool:
            _, pending = val
            return jnp.any(pending)

        def _body(val: tuple[CuckooIdx, chex.Array]) -> tuple[CuckooIdx, chex.Array]:
            idxs, pending = val
            updated_idxs = _next_idx(idxs, pending)
            overflowed = jnp.logical_and(updated_idxs.table_index >= table.cuckoo_table_n, pending)
            not_uniques = jnp.logical_not(xnp.unique_mask(updated_idxs, filled=updatable))
            next_pending = jnp.logical_and(updatable, not_uniques)
            next_pending = jnp.logical_or(next_pending, overflowed)
            return updated_idxs, next_pending

        index, _ = jax.lax.while_loop(_cond, _body, (index, unupdated))

        successful = updatable
        flat_indices = index.index * table.cuckoo_table_n + index.table_index
        table.table = table.table.at[flat_indices].set_as_condition(successful, inputs)

        current_fp = table.fingerprints[flat_indices]
        delta_fp = jnp.where(
            successful, fingerprints.astype(jnp.uint32) - current_fp, jnp.uint32(0)
        )
        table.fingerprints = table.fingerprints.at[flat_indices].add(delta_fp)
        table.table_idx = table.table_idx.at[index.index].add(successful)
        table.size += jnp.sum(successful, dtype=SIZE_DTYPE)
        return table, index

    @jax.jit
    def parallel_insert(
        table: "HashTable",
        inputs: Xtructurable,
        filled: chex.Array = None,
        unique_key: chex.Array = None,
    ):
        """
        Parallel insertion of multiple states into the hash table.

        Args:
            table: Hash table instance
            inputs: States to insert
            filled: Boolean array indicating which inputs are valid
            unique_key: Optional key array for determining priority among duplicate states.
                       When provided, among duplicate states, only the one with the smallest
                       key value will be marked as unique in unique_filled mask.

        Returns:
            Tuple of (updated_table, updatable, unique_filled, idx)
        """
        if filled is None:
            filled = jnp.ones((len(inputs),), dtype=jnp.bool_)

        # Get initial indices, probe steps, and byte representations
        initial_idx, steps, uint32eds, fingerprints = jax.vmap(
            get_new_idx_byterized, in_axes=(0, None, None)
        )(inputs, table._capacity, table.seed)

        batch_len = filled.shape[0]

        # Find unique states to avoid duplicates using enhanced unique_mask with filled optimization
        unique_filled, unique_uint32eds_idx, inverse_indices = xnp.unique_mask(
            val=inputs,
            key=unique_key,
            filled=filled,
            batch_len=batch_len,
            return_index=True,
            return_inverse=True,
        )

        # Look up each state
        idx = CuckooIdx(
            index=initial_idx, table_index=jnp.zeros((batch_len,), dtype=HASH_TABLE_IDX_DTYPE)
        )

        initial_found = jnp.logical_not(unique_filled)
        idx, found = HashTable._lookup_parallel(
            table, inputs, uint32eds, idx, steps, fingerprints, initial_found
        )

        updatable = jnp.logical_and(~found, unique_filled)

        # Perform parallel insertion
        table, inserted_idx = HashTable._parallel_insert(
            table, inputs, uint32eds, steps, idx, updatable, fingerprints
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
        provisional_idx = CuckooIdx(index=provisional_index, table_index=provisional_table_index)

        # Only keep indices for unique elements
        correct_indices_for_uniques = CuckooIdx(
            index=provisional_idx.index[unique_uint32eds_idx],
            table_index=provisional_idx.table_index[unique_uint32eds_idx],
        )
        # Broadcast to all batch elements using inverse_indices
        final_idx = CuckooIdx(
            index=correct_indices_for_uniques.index[inverse_indices],
            table_index=correct_indices_for_uniques.table_index[inverse_indices],
        )

        return (
            table,
            updatable,
            unique_filled,
            HashIdx(index=final_idx.index * table.cuckoo_table_n + final_idx.table_index),
        )

    @jax.jit
    def __getitem__(self, idx: HashIdx) -> Xtructurable:
        return self.table[idx.index]
