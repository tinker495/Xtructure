"""
Hash table implementation using Cuckoo hashing technique for efficient state storage and lookup.
This module provides functionality for hashing Xtructurables and managing collisions.
"""

from functools import partial
from typing import TypeVar

import chex
import jax
import jax.numpy as jnp

from ..core import FieldDescriptor, Xtructurable, xtructure_dataclass
from ..core.xtructure_decorators.hash import uint32ed_to_hash

SIZE_DTYPE = jnp.uint32
HASH_TABLE_IDX_DTYPE = jnp.uint8

T = TypeVar("T")


@xtructure_dataclass
class CuckooIdx:
    index: FieldDescriptor[SIZE_DTYPE]
    table_index: FieldDescriptor[HASH_TABLE_IDX_DTYPE]


@xtructure_dataclass
class HashIdx:
    index: FieldDescriptor[SIZE_DTYPE]


def get_new_idx_from_uint32ed(
    input_uint32ed: chex.Array,
    capacity: int,
    seed: int,
) -> int:
    """
    Calculate new index for input state using the hash function from its uint32ed representation.
    """
    hash_value = uint32ed_to_hash(input_uint32ed, seed)
    idx = hash_value % capacity
    return idx


def get_new_idx_byterized(
    input: Xtructurable,
    capacity: int,
    seed: int,
) -> tuple[int, chex.Array]:
    """
    Calculate new index and return uint32ed representation of input state.
    Similar to get_new_idx but also returns the uint32ed representation for
    equality comparison.
    """
    hash_value, uint32ed = input.hash_with_uint32ed(seed)
    idx = hash_value % capacity
    return idx, uint32ed


@chex.dataclass
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
        return HashTable(
            seed=seed,
            capacity=capacity,
            _capacity=_capacity,
            cuckoo_table_n=cuckoo_table_n,
            size=size,
            table=table,
            table_idx=table_idx,
        )

    @staticmethod
    def _lookup(
        table: "HashTable",
        input: Xtructurable,
        input_uint32ed: chex.Array,
        idx: CuckooIdx,
        seed: int,
        found: bool,
    ) -> tuple[int, CuckooIdx, bool]:
        """
        Internal lookup method that searches for a state in the table.
        Uses cuckoo hashing technique to check multiple possible locations.

        Args:
            table: Hash table instance
            input: State to look up
            input_uint32ed: uint32ed representation of the state to look up
            idx: Initial index to check
            seed: Initial seed
            found: Whether the state has been found

        Returns:
            Tuple of (seed, idx, table_idx, found)
        """

        def _cond(val: tuple[int, CuckooIdx, bool]):
            seed, idx, found = val
            filled_idx = table.table_idx[idx.index]
            in_empty = idx.table_index >= filled_idx
            return jnp.logical_and(~found, ~in_empty)

        def _while(val: tuple[int, CuckooIdx, bool]):
            seed, idx, found = val

            def get_new_idx_and_table_idx(seed, idx):
                next_table = idx.table_index >= (table.cuckoo_table_n - 1)
                seed, idx = jax.lax.cond(
                    next_table,
                    lambda _: (
                        seed + 1,
                        CuckooIdx(
                            index=get_new_idx_from_uint32ed(
                                input_uint32ed, table.capacity, seed + 1
                            ),
                            table_index=HASH_TABLE_IDX_DTYPE(0),
                        ),
                    ),
                    lambda _: (
                        seed,
                        CuckooIdx(
                            index=idx.index,
                            table_index=idx.table_index + 1,
                        ),
                    ),
                    None,
                )
                return seed, idx

            state = table.table[
                idx.index * table.cuckoo_table_n + idx.table_index
            ]  # Modified indexing
            found = state == input
            seed, idx = jax.lax.cond(
                found,
                lambda _: (seed, idx),
                lambda _: get_new_idx_and_table_idx(seed, idx),
                None,
            )
            return seed, idx, found

        state = table.table[idx.index * table.cuckoo_table_n + idx.table_index]  # Modified indexing
        found = jnp.logical_or(found, state == input)
        update_seed, idx, found = jax.lax.while_loop(_cond, _while, (seed, idx, found))
        return update_seed, idx, found

    @jax.jit
    def lookup_cuckoo(table: "HashTable", input: Xtructurable) -> tuple[CuckooIdx, bool]:
        """
        Finds the state in the hash table using Cuckoo hashing.

        Args:
            table: The HashTable instance.
            input: The Xtructurable state to look up.

        Returns:
            A tuple (idx, table_idx, found):
            - idx (int): The primary hash index in the table.
            - table_idx (int): The cuckoo table index (which hash function/slot was used or probed).
            - found (bool): True if the state was found, False otherwise.
            If not found, idx and table_idx indicate the first empty slot encountered
            during the Cuckoo search path where an insertion could occur.
        """
        index, input_uint32ed = get_new_idx_byterized(input, table.capacity, table.seed)
        idx = CuckooIdx(index=index, table_index=HASH_TABLE_IDX_DTYPE(0))
        _, idx, found = HashTable._lookup(table, input, input_uint32ed, idx, table.seed, False)
        return idx, found

    @jax.jit
    def lookup(table: "HashTable", input: Xtructurable) -> tuple[HashIdx, bool]:
        """
        Finds the state in the hash table using Cuckoo hashing.

        Args:
            table: The HashTable instance.
            input: The Xtructurable state to look up.

        Returns:
            A tuple (idx, table_idx, found):
            - idx (int): The primary hash index in the table.
            - table_idx (int): The cuckoo table index (which hash function/slot was used or probed).
            - found (bool): True if the state was found, False otherwise.
            If not found, idx and table_idx indicate the first empty slot encountered
            during the Cuckoo search path where an insertion could occur.
        """
        idx, found = HashTable.lookup_cuckoo(table, input)
        return HashIdx(index=idx.index * table.cuckoo_table_n + idx.table_index), found

    @staticmethod
    def _lookup_parallel(
        table: "HashTable",
        inputs: Xtructurable,
        input_uint32eds: chex.Array,
        idxs: CuckooIdx,
        seeds: chex.Array,
        founds: chex.Array,
    ) -> tuple[chex.Array, CuckooIdx, chex.Array]:
        """
        Internal lookup method that searches for states in the table in parallel.
        Uses cuckoo hashing technique to check multiple possible locations.
        """

        def _lu(
            input: Xtructurable,
            input_uint32ed: chex.Array,
            idx: CuckooIdx,
            seed: int,
            found: bool,
        ) -> tuple[int, CuckooIdx, bool]:
            # lookup but not copying the table
            def _cond(val: tuple[int, CuckooIdx, bool]):
                seed, idx, found = val
                filled_idx = table.table_idx[idx.index]
                in_empty = idx.table_index >= filled_idx
                return jnp.logical_and(~found, ~in_empty)

            def _while(val: tuple[int, CuckooIdx, bool]):
                seed, idx, found = val

                def get_new_idx_and_table_idx(seed, idx):
                    next_table = idx.table_index >= (table.cuckoo_table_n - 1)
                    seed, idx = jax.lax.cond(
                        next_table,
                        lambda _: (
                            seed + 1,
                            CuckooIdx(
                                index=get_new_idx_from_uint32ed(
                                    input_uint32ed, table.capacity, seed + 1
                                ),
                                table_index=HASH_TABLE_IDX_DTYPE(0),
                            ),
                        ),
                        lambda _: (
                            seed,
                            CuckooIdx(
                                index=idx.index,
                                table_index=idx.table_index + 1,
                            ),
                        ),
                        None,
                    )
                    return seed, idx

                state = table.table[
                    idx.index * table.cuckoo_table_n + idx.table_index
                ]  # Modified indexing
                found = state == input
                seed, idx = jax.lax.cond(
                    found,
                    lambda _: (seed, idx),
                    lambda _: get_new_idx_and_table_idx(seed, idx),
                    None,
                )
                return seed, idx, found

            state = table.table[
                idx.index * table.cuckoo_table_n + idx.table_index
            ]  # Modified indexing
            found = jnp.logical_or(found, state == input)
            update_seed, idx, found = jax.lax.while_loop(_cond, _while, (seed, idx, found))
            return update_seed, idx, found

        seeds, idxs, founds = jax.vmap(_lu, in_axes=(0, 0, 0, 0, 0))(
            inputs, input_uint32eds, idxs, seeds, founds
        )
        return seeds, idxs, founds

    @jax.jit
    def lookup_parallel(table: "HashTable", inputs: Xtructurable) -> tuple[HashIdx, bool]:
        """
        Finds the state in the hash table using Cuckoo hashing.
        """
        initial_idx, input_uint32eds = jax.vmap(get_new_idx_byterized, in_axes=(0, None, None))(
            inputs, table.capacity, table.seed
        )

        batch_size = inputs.shape.batch

        idxs = CuckooIdx(
            index=initial_idx, table_index=jnp.zeros(batch_size, dtype=HASH_TABLE_IDX_DTYPE)
        )
        seeds = jnp.full(batch_size, table.seed, dtype=jnp.uint32)
        founds = jnp.zeros(batch_size, dtype=jnp.bool_)

        _, idx, found = HashTable._lookup_parallel(
            table, inputs, input_uint32eds, idxs, seeds, founds
        )
        return HashIdx(index=idx.index * table.cuckoo_table_n + idx.table_index), found

    @jax.jit
    def insert(table: "HashTable", input: Xtructurable) -> tuple["HashTable", bool, HashIdx]:
        """
        insert the state in the table
        """

        def _update_table(table: "HashTable", input: Xtructurable, idx: CuckooIdx):
            """
            insert the state in the table
            """
            table.table = table.table.at[idx.index * table.cuckoo_table_n + idx.table_index].set(
                input
            )
            table.table_idx = table.table_idx.at[idx.index].add(1)
            return table

        idx, found = HashTable.lookup_cuckoo(table, input)
        table = jax.lax.cond(
            found, lambda _: table, lambda _: _update_table(table, input, idx), None
        )
        return table, ~found, HashIdx(index=idx.index * table.cuckoo_table_n + idx.table_index)

    @staticmethod
    def _parallel_insert(
        table: "HashTable",
        inputs: Xtructurable,
        inputs_uint32ed: chex.Array,
        seeds: chex.Array,
        index: CuckooIdx,
        updatable: chex.Array,
        batch_len: int,
    ):
        def _next_idx(seeds: chex.Array, idxs: CuckooIdx, unupdateds: chex.Array):
            def get_new_idx_and_table_idx(seed, idx, state_uint32ed):
                next_table = idx.table_index >= (table.cuckoo_table_n - 1)

                def next_table_fn(seed, table):
                    next_idx = get_new_idx_from_uint32ed(state_uint32ed, table.capacity, seed + 1)
                    seed = seed + 1
                    return seed, CuckooIdx(index=next_idx, table_index=table.table_idx[next_idx])

                seed, idx = jax.lax.cond(
                    next_table,
                    next_table_fn,
                    lambda seed, _: (
                        seed,
                        CuckooIdx(index=idx.index, table_index=idx.table_index + 1),
                    ),
                    seed,
                    table,
                )
                return seed, idx

            seeds, idxs = jax.vmap(
                lambda unupdated, seed, idx, state_uint32ed: jax.lax.cond(
                    unupdated,
                    lambda _: get_new_idx_and_table_idx(seed, idx, state_uint32ed),
                    lambda _: (seed, idx),
                    None,
                )
            )(unupdateds, seeds, idxs, inputs_uint32ed)
            return seeds, idxs

        def _cond(val: tuple[chex.Array, CuckooIdx, chex.Array]):
            _, _, unupdated = val
            return jnp.any(unupdated)

        def _while(val: tuple[chex.Array, CuckooIdx, chex.Array]):
            seeds, idxs, unupdated = val
            seeds, idxs = _next_idx(seeds, idxs, unupdated)

            overflowed = jnp.logical_and(
                idxs.table_index >= table.cuckoo_table_n, unupdated
            )  # Overflowed index must be updated
            idxs = jax.tree_util.tree_map(lambda x: jnp.where(updatable, x, -1), idxs)
            idx_bytes = jax.vmap(lambda x: x.uint32ed)(idxs)
            unique_idxs = jnp.unique(idx_bytes, axis=0, size=batch_len, return_index=True)[
                1
            ]  # val = (unique_len, 2), unique_idxs = (unique_len,)
            not_uniques = (
                jnp.ones((batch_len,), dtype=jnp.bool_).at[unique_idxs].set(False)
            )  # set the unique index to True

            unupdated = jnp.logical_and(updatable, not_uniques)
            unupdated = jnp.logical_or(unupdated, overflowed)
            return seeds, idxs, unupdated

        index = jax.tree_util.tree_map(lambda x: jnp.where(updatable, x, -1), index)
        # Use index_bytes for unique calculation instead of _idxs
        hash_idx_bytes = jax.vmap(lambda x: x.uint32ed)(index)
        unique_idxs = jnp.unique(hash_idx_bytes, axis=0, size=batch_len, return_index=True)[1]
        not_uniques = (
            jnp.ones((batch_len,), dtype=jnp.bool_).at[unique_idxs].set(False)
        )  # set the unique index to False (i.e., mark uniques as False)
        unupdated = jnp.logical_and(
            updatable, not_uniques
        )  # remove the unique index from the unupdated index

        seeds, index, _ = jax.lax.while_loop(_cond, _while, (seeds, index, unupdated))

        flat_indices = index.index * table.cuckoo_table_n + index.table_index
        table.table = table.table.at[flat_indices].set_as_condition(updatable, inputs)
        table.table_idx = table.table_idx.at[index.index].add(updatable)
        table.size += jnp.sum(updatable, dtype=SIZE_DTYPE)
        return table, index

    @jax.jit
    def parallel_insert(table: "HashTable", inputs: Xtructurable, filled: chex.Array = None):
        """
        Parallel insertion of multiple states into the hash table.

        Args:
            table: Hash table instance
            inputs: States to insert
            filled: Boolean array indicating which inputs are valid

        Returns:
            Tuple of (updated_table, updatable, unique_filled, idx, table_idx)
        """
        if filled is None:
            filled = jnp.ones((len(inputs),), dtype=jnp.bool_)

        # Get initial indices and byte representations
        initial_idx, uint32eds = jax.vmap(get_new_idx_byterized, in_axes=(0, None, None))(
            inputs, table.capacity, table.seed
        )

        batch_len = filled.shape[0]

        # Find unique states to avoid duplicates
        unique_uint32eds_idx = jnp.unique(uint32eds, axis=0, size=batch_len, return_index=True)[1]
        unique = jnp.zeros((batch_len,), dtype=jnp.bool_).at[unique_uint32eds_idx].set(True)
        unique_filled = jnp.logical_and(filled, unique)

        # Look up each state
        idx = CuckooIdx(
            index=initial_idx, table_index=jnp.zeros((batch_len,), dtype=HASH_TABLE_IDX_DTYPE)
        )

        seeds = jnp.full((batch_len,), table.seed, dtype=jnp.uint32)

        seeds, idx, found = HashTable._lookup_parallel(
            table, inputs, uint32eds, idx, seeds, ~unique_filled
        )

        updatable = jnp.logical_and(~found, unique_filled)

        # Perform parallel insertion
        table, idx = HashTable._parallel_insert(
            table, inputs, uint32eds, seeds, idx, updatable, batch_len
        )

        # Get final indices
        idx = CuckooIdx(
            index=initial_idx, table_index=jnp.zeros((batch_len,), dtype=HASH_TABLE_IDX_DTYPE)
        )

        seeds = jnp.full((batch_len,), table.seed, dtype=jnp.uint32)
        _, idx, _ = HashTable._lookup_parallel(table, inputs, uint32eds, idx, seeds, ~filled)

        return (
            table,
            updatable,
            unique_filled,
            HashIdx(index=idx.index * table.cuckoo_table_n + idx.table_index),
        )

    @jax.jit
    def get(self, idx: HashIdx) -> Xtructurable:
        return self.table[idx.index]
