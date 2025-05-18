"""
Hash table implementation using Cuckoo hashing technique for efficient state storage and lookup.
This module provides functionality for hashing Xtructurables and managing collisions.
"""

from functools import partial
from typing import TypeVar

import chex
import jax
import jax.numpy as jnp

from ..core import Xtructurable

SIZE_DTYPE = jnp.uint32
HASH_TABLE_IDX_DTYPE = jnp.uint8

T = TypeVar("T")


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
    ):
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
        table = dataclass.default((_capacity + 1, cuckoo_table_n))
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
    def get_new_idx(
        table: "HashTable",
        input: Xtructurable,
        seed: int,
    ):
        """
        Calculate new index for input state using the hash function.

        Args:
            table: Hash table instance
            input: State to hash
            seed: Seed for hash function

        Returns:
            Index in the table for the input state
        """
        hash_value, _ = input.hash(seed)
        idx = hash_value % table._capacity
        return idx

    @staticmethod
    def get_new_idx_byterized(
        table: "HashTable",
        input: Xtructurable,
        seed: int,
    ):
        """
        Calculate new index and return byte representation of input state.
        Similar to get_new_idx but also returns the byte representation for
        equality comparison.
        """
        hash_value, bytes = input.hash(seed)
        idx = hash_value % table._capacity
        return idx, bytes

    @staticmethod
    def _lookup(
        table: "HashTable",
        input: Xtructurable,
        idx: int,
        table_idx: int,
        seed: int,
        found: bool,
    ):
        """
        Internal lookup method that searches for a state in the table.
        Uses cuckoo hashing technique to check multiple possible locations.

        Args:
            table: Hash table instance
            input: State to look up
            idx: Initial index to check
            table_idx: Which hash function to start with
            seed: Initial seed
            found: Whether the state has been found

        Returns:
            Tuple of (seed, idx, table_idx, found)
        """

        def _cond(val):
            seed, idx, table_idx, found = val
            filled_idx = table.table_idx[idx]
            in_empty = table_idx >= filled_idx
            return jnp.logical_and(~found, ~in_empty)

        def _while(val):
            seed, idx, table_idx, found = val

            def get_new_idx_and_table_idx(seed, idx, table_idx):
                next_table = table_idx >= (table.cuckoo_table_n - 1)
                seed, idx, table_idx = jax.lax.cond(
                    next_table,
                    lambda _: (
                        seed + 1,
                        HashTable.get_new_idx(table, input, seed + 1),
                        HASH_TABLE_IDX_DTYPE(0),
                    ),
                    lambda _: (seed, idx, HASH_TABLE_IDX_DTYPE(table_idx + 1)),
                    None,
                )
                return seed, idx, table_idx

            state = table.table[idx, table_idx]
            found = state == input
            seed, idx, table_idx = jax.lax.cond(
                found,
                lambda _: (seed, idx, table_idx),
                lambda _: get_new_idx_and_table_idx(seed, idx, table_idx),
                None,
            )
            return seed, idx, table_idx, found

        state = table.table[idx, table_idx]
        found = jnp.logical_or(found, state == input)
        update_seed, idx, table_idx, found = jax.lax.while_loop(
            _cond, _while, (seed, idx, table_idx, found)
        )
        return update_seed, idx, table_idx, found

    def lookup(table: "HashTable", input: Xtructurable):
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
        index = HashTable.get_new_idx(table, input, table.seed)
        _, idx, table_idx, found = HashTable._lookup(
            table, input, index, HASH_TABLE_IDX_DTYPE(0), table.seed, False
        )
        return idx, table_idx, found

    def insert(table: "HashTable", input: Xtructurable):
        """
        insert the state in the table
        """

        def _update_table(table: "HashTable", input: Xtructurable, idx: int, table_idx: int):
            """
            insert the state in the table
            """
            table.table = table.table.at[idx, table_idx].set(input)
            table.table_idx = table.table_idx.at[idx].add(1)
            return table

        idx, table_idx, found = HashTable.lookup(table, input)
        return (
            jax.lax.cond(
                found, lambda _: table, lambda _: _update_table(table, input, idx, table_idx), None
            ),
            ~found,
        )

    @staticmethod
    @partial(
        jax.jit,
        static_argnums=(
            0,
            2,
        ),
    )
    def make_batched(statecls: Xtructurable, inputs: Xtructurable, batch_size: int):
        """
        make a batched version of the inputs
        """
        count = len(inputs)
        batched = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y]),
            inputs,
            statecls.default((batch_size - count,)),
        )
        filled = jnp.concatenate([jnp.ones(count), jnp.zeros(batch_size - count)], dtype=jnp.bool_)
        return batched, filled

    @staticmethod
    def _parallel_insert(
        table: "HashTable",
        inputs: Xtructurable,
        seeds: chex.Array,
        index: chex.Array,
        updatable: chex.Array,
        batch_len: int,
    ):
        def _next_idx(seeds, _idxs, unupdateds):
            def get_new_idx_and_table_idx(seed, idx, table_idx, state):
                next_table = table_idx >= (table.cuckoo_table_n - 1)

                def next_table_fn(seed, table):
                    next_idx = HashTable.get_new_idx(table, state, seed)
                    seed = seed + 1
                    return seed, next_idx, table.table_idx[next_idx].astype(jnp.uint32)

                seed, idx, table_idx = jax.lax.cond(
                    next_table,
                    next_table_fn,
                    lambda seed, _: (seed, idx, table_idx + 1),
                    seed,
                    table,
                )
                return seed, idx, table_idx

            idxs = _idxs[:, 0]
            table_idxs = _idxs[:, 1]
            seeds, idxs, table_idxs = jax.vmap(
                lambda unupdated, seed, idx, table_idx, state: jax.lax.cond(
                    unupdated,
                    lambda _: get_new_idx_and_table_idx(seed, idx, table_idx, state),
                    lambda _: (seed, idx, table_idx),
                    None,
                )
            )(unupdateds, seeds, idxs, table_idxs, inputs)
            _idxs = jnp.stack((idxs, table_idxs), axis=1)
            return seeds, _idxs

        def _cond(val):
            _, _, unupdated = val
            return jnp.any(unupdated)

        def _while(val):
            seeds, _idxs, unupdated = val
            seeds, _idxs = _next_idx(seeds, _idxs, unupdated)

            overflowed = jnp.logical_and(
                _idxs[:, 1] >= table.cuckoo_table_n, unupdated
            )  # Overflowed index must be updated
            _idxs = jnp.where(updatable[:, jnp.newaxis], _idxs, jnp.full_like(_idxs, -1))
            unique_idxs = jnp.unique(_idxs, axis=0, size=batch_len, return_index=True)[
                1
            ]  # val = (unique_len, 2), unique_idxs = (unique_len,)
            not_uniques = (
                jnp.ones((batch_len,), dtype=jnp.bool_).at[unique_idxs].set(False)
            )  # set the unique index to True

            unupdated = jnp.logical_and(updatable, not_uniques)
            unupdated = jnp.logical_or(unupdated, overflowed)
            return seeds, _idxs, unupdated

        _idxs = jnp.where(updatable[:, jnp.newaxis], index, jnp.full_like(index, -1))
        unique_idxs = jnp.unique(_idxs, axis=0, size=batch_len, return_index=True)[
            1
        ]  # val = (unique_len, 2), unique_idxs = (unique_len,)
        not_uniques = (
            jnp.ones((batch_len,), dtype=jnp.bool_).at[unique_idxs].set(False)
        )  # set the unique index to True
        unupdated = jnp.logical_and(
            updatable, not_uniques
        )  # remove the unique index from the unupdated index

        seeds, index, _ = jax.lax.while_loop(_cond, _while, (seeds, _idxs, unupdated))

        idx, table_idx = index[:, 0], index[:, 1].astype(HASH_TABLE_IDX_DTYPE)
        table.table = table.table.at[idx, table_idx].set_as_condition(updatable, inputs)
        table.table_idx = table.table_idx.at[idx].add(updatable)
        table.size += jnp.sum(updatable, dtype=SIZE_DTYPE)
        return table, idx, table_idx

    def parallel_insert(table: "HashTable", inputs: Xtructurable, filled: chex.Array):
        """
        Parallel insertion of multiple states into the hash table.

        Args:
            table: Hash table instance
            inputs: States to insert
            filled: Boolean array indicating which inputs are valid

        Returns:
            Tuple of (updated_table, updatable, unique_filled, idx, table_idx)

        Note:
            This implementation has a known issue with the search functionality
            after parallel insertion. This should be fixed in future versions.

        TODO: Fix search functionality after parallel insertion
        """

        # Get initial indices and byte representations
        initial_idx, bytes = jax.vmap(
            partial(HashTable.get_new_idx_byterized), in_axes=(None, 0, None)
        )(table, inputs, table.seed)

        batch_len = filled.shape[0]

        # Find unique states to avoid duplicates
        unique_bytes_idx = jnp.unique(bytes, axis=0, size=batch_len, return_index=True)[1]
        unique = jnp.zeros((batch_len,), dtype=jnp.bool_).at[unique_bytes_idx].set(True)
        unique_filled = jnp.logical_and(filled, unique)

        # Look up each state
        seeds, idx, table_idx, found = jax.vmap(
            partial(HashTable._lookup), in_axes=(None, 0, 0, None, None, 0)
        )(table, inputs, initial_idx, HASH_TABLE_IDX_DTYPE(0), table.seed, ~unique_filled)

        idxs = jnp.stack([idx, table_idx], axis=1, dtype=SIZE_DTYPE)
        updatable = jnp.logical_and(~found, unique_filled)

        # Perform parallel insertion
        table, idx, table_idx = HashTable._parallel_insert(
            table, inputs, seeds, idxs, updatable, batch_len
        )

        # Get final indices
        _, idx, table_idx, _ = jax.vmap(
            partial(HashTable._lookup), in_axes=(None, 0, 0, None, None, 0)
        )(table, inputs, initial_idx, HASH_TABLE_IDX_DTYPE(0), table.seed, ~filled)

        return table, updatable, unique_filled, idx, table_idx
