"""
Batched GPU Priority Queue (BGPQ) Implementation
This module provides a JAX-compatible priority queue optimized for GPU operations.
Key features:
- Fully batched operations for GPU efficiency
- Supports custom value types through dataclass
- Uses infinity padding for unused slots
- Maintains sorted order for efficient min/max operations
"""

from functools import partial

import chex
import jax
import jax.numpy as jnp

from ..core import Xtructurable, base_dataclass
from ..core.xtructure_numpy.array_ops import _where_no_broadcast
from ._constants import SIZE_DTYPE
from ._delete import _bgpq_delete_mins_jit
from ._insert import (
    _bgpq_insert_jit,
    _bgpq_make_batched_jit,
    _bgpq_make_batched_like_jit,
    _bgpq_merge_buffer_jit,
)


@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def _bgpq_build_jit(total_size, batch_size, value_class=Xtructurable, key_dtype=jnp.float16):
    total_size = total_size
    # Calculate branch size, rounding up if total_size not divisible by batch_size
    branch_size = (
        total_size // batch_size if total_size % batch_size == 0 else total_size // batch_size + 1
    )
    max_size = branch_size * batch_size
    heap_size = SIZE_DTYPE(0)
    buffer_size = SIZE_DTYPE(0)

    # Initialize storage arrays with infinity for unused slots
    key_store = jnp.full((branch_size, batch_size), jnp.inf, dtype=key_dtype)
    val_store = value_class.default((branch_size, batch_size))
    key_buffer = jnp.full((batch_size - 1,), jnp.inf, dtype=key_dtype)
    val_buffer = value_class.default((batch_size - 1,))

    return BGPQ(
        max_size=max_size,
        heap_size=heap_size,
        buffer_size=buffer_size,
        branch_size=branch_size,
        batch_size=batch_size,
        key_store=key_store,
        val_store=val_store,
        key_buffer=key_buffer,
        val_buffer=val_buffer,
    )


@base_dataclass(static_fields=("max_size", "branch_size", "batch_size"))
class BGPQ:
    """
    Batched GPU Priority Queue implementation.
    Optimized for parallel operations on GPU using JAX.

    Attributes:
        max_size: Maximum number of elements the queue can hold
        size: Current number of elements in the queue
        branch_size: Number of branches in the heap tree
        batch_size: Size of batched operations
        key_store: Array storing keys in a binary heap structure
        val_store: Array storing associated values
        key_buffer: Buffer for keys waiting to be inserted
        val_buffer: Buffer for values waiting to be inserted
    """

    max_size: int
    heap_size: int
    buffer_size: int
    branch_size: int
    batch_size: int
    key_store: chex.Array  # shape = (total_size, batch_size)
    val_store: Xtructurable  # shape = (total_size, batch_size, ...)
    key_buffer: chex.Array  # shape = (batch_size - 1,)
    val_buffer: Xtructurable  # shape = (batch_size - 1, ...)

    @staticmethod
    def build(total_size, batch_size, value_class=Xtructurable, key_dtype=jnp.float16) -> "BGPQ":
        """
        Create a new BGPQ instance with specified capacity.

        Args:
            total_size: Total number of elements the queue can store
            batch_size: Size of batched operations
            value_class: Class to use for storing values (must implement default())

        Returns:
            BGPQ: A new priority queue instance initialized with empty storage
        """
        return _bgpq_build_jit(total_size, batch_size, value_class, key_dtype)

    @property
    def size(self):
        cond = jnp.asarray(self.heap_size == 0, dtype=jnp.bool_)
        empty_branch = jnp.asarray(jnp.sum(jnp.isfinite(self.key_store[0])) + self.buffer_size)
        non_empty_branch = jnp.asarray((self.heap_size + 1) * self.batch_size + self.buffer_size)
        target_dtype = jnp.result_type(empty_branch.dtype, non_empty_branch.dtype)
        return _where_no_broadcast(
            cond,
            empty_branch.astype(target_dtype),
            non_empty_branch.astype(target_dtype),
        )

    def merge_buffer(self, blockk: chex.Array, blockv: Xtructurable):
        """
        Merge buffer contents with block contents, handling overflow conditions.

        This method is crucial for maintaining the heap property when inserting new elements.
        It handles the case where the buffer might overflow into the main storage.

        Args:
            blockk: Block keys array
            blockv: Block values
            bufferk: Buffer keys array
            bufferv: Buffer values

        Returns:
            tuple containing:
                - Updated block keys
                - Updated block values
                - Updated buffer keys
                - Updated buffer values
                - Boolean indicating if buffer overflow occurred
        """
        return _bgpq_merge_buffer_jit(self, blockk, blockv)

    @staticmethod
    def make_batched(key: chex.Array, val: Xtructurable, batch_size: int):
        """
        Convert unbatched arrays into batched format suitable for the queue.

        Args:
            key: Array of keys to batch
            val: Xtructurable of values to batch
            batch_size: Desired batch size

        Returns:
            tuple containing:
                - Batched key array
                - Batched value array
        """
        return _bgpq_make_batched_jit(key, val, batch_size)

    def make_batched_like(self, key: chex.Array, val: Xtructurable):
        """Pad `key`/`val` to this heap's `batch_size` (a `static_fields` config)."""
        return _bgpq_make_batched_like_jit(self, key, val)

    def insert(self, block_key: chex.Array, block_val: Xtructurable) -> "BGPQ":
        """
        Insert new elements into the priority queue.
        Maintains heap property through merge operations and heapification.

        Args:
            block_key: Keys to insert
            block_val: Values to insert
            added_size: Optional size of insertion (calculated if None)

        Returns:
            Updated heap instance
        """
        return _bgpq_insert_jit(self, block_key, block_val)

    def delete_mins(self):
        """
        Remove and return the minimum elements from the queue.

        Returns:
            tuple containing:
                - Updated heap instance
                - Array of minimum keys removed
                - Xtructurable of corresponding values
        """
        return _bgpq_delete_mins_jit(self)
