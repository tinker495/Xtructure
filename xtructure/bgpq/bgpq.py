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

from ..core import Xtructurable
from .merge_split import merge_arrays_parallel, merge_sort_split_idx

SORT_STABLE = True  # Use stable sorting to maintain insertion order for equal keys
SIZE_DTYPE = jnp.uint32

# TODO: Make merge_arrays_parallel for TPU.
merge_array_backend = merge_sort_split_idx if jax.default_backend() == "tpu" else merge_arrays_parallel

@jax.jit
def merge_sort_split(
    ak: chex.Array, av: Xtructurable, bk: chex.Array, bv: Xtructurable
) -> tuple[chex.Array, Xtructurable, chex.Array, Xtructurable]:
    """
    Merge and split two sorted arrays while maintaining their relative order.
    This is a key operation for maintaining heap property in batched operations.

    Args:
        ak: First array of keys
        av: First array of values
        bk: Second array of keys
        bv: Second array of values

    Returns:
        tuple containing:
            - First half of merged and sorted keys
            - First half of corresponding values
            - Second half of merged and sorted keys
            - Second half of corresponding values
    """
    n = ak.shape[-1]  # size of group
    val = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b]), av, bv)
    sorted_key, sorted_idx = merge_array_backend(ak, bk)
    sorted_val = val[sorted_idx]
    return sorted_key[:n], sorted_val[:n], sorted_key[n:], sorted_val[n:]


def sort_arrays(k: chex.Array, v: Xtructurable):
    sorted_k, sorted_idx = jax.lax.sort_key_val(k, jnp.arange(k.shape[0]), is_stable=SORT_STABLE)
    sorted_v = v[sorted_idx]
    return sorted_k, sorted_v


@jax.jit
def _next(current, target):
    """
    Calculate the next index in the heap traversal path.
    Uses leading zero count (clz) for efficient binary tree navigation.

    Args:
        current: Current index in the heap
        target: Target index to reach

    Returns:
        Next index in the path from current to target
    """
    clz_current = jax.lax.clz(current)
    clz_target = jax.lax.clz(target)
    shift_amount = clz_current - clz_target - 1
    next_index = target.astype(SIZE_DTYPE) >> shift_amount
    return next_index


@chex.dataclass
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
    @partial(jax.jit, static_argnums=(0, 1, 2, 3))
    def build(total_size, batch_size, value_class=Xtructurable, key_dtype=jnp.float16):
        """
        Create a new BGPQ instance with specified capacity.

        Args:
            total_size: Total number of elements the queue can store
            batch_size: Size of batched operations
            value_class: Class to use for storing values (must implement default())

        Returns:
            BGPQ: A new priority queue instance initialized with empty storage
        """
        total_size = total_size
        # Calculate branch size, rounding up if total_size not divisible by batch_size
        branch_size = (
            total_size // batch_size
            if total_size % batch_size == 0
            else total_size // batch_size + 1
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

    @property
    def size(self):
        return jnp.where(
            self.heap_size == 0,
            jnp.sum(jnp.isfinite(self.key_store[0])) + self.buffer_size,
            (self.heap_size + 1) * self.batch_size + self.buffer_size,
        )

    @jax.jit
    def merge_buffer(heap: "BGPQ", blockk: chex.Array, blockv: Xtructurable):
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
        n = blockk.shape[0]
        # Concatenate block and buffer
        sorted_key, sorted_idx = merge_array_backend(blockk, heap.key_buffer)
        val = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b]), blockv, heap.val_buffer)
        val = val[sorted_idx]

        # Check for active elements (non-infinity)
        filled = jnp.isfinite(sorted_key)
        n_filled = jnp.sum(filled)
        buffer_overflow = n_filled >= n

        def overflowed(key, val):
            """Handle case where buffer overflows"""
            return key[:n], val[:n], key[n:], val[n:]

        def not_overflowed(key, val):
            return key[-n:], val[-n:], key[:-n], val[:-n]

        blockk, blockv, heap.key_buffer, heap.val_buffer = jax.lax.cond(
            buffer_overflow,
            overflowed,
            not_overflowed,
            sorted_key,
            val,
        )
        heap.buffer_size = jnp.sum(jnp.isfinite(heap.key_buffer), dtype=SIZE_DTYPE)
        return heap, blockk, blockv, buffer_overflow

    @staticmethod
    @partial(jax.jit, static_argnums=(2))
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
        n = key.shape[0]
        # Pad arrays to match batch size
        key = jnp.pad(key, (0, batch_size - n), mode="constant", constant_values=jnp.inf)
        val = val.padding_as_batch((batch_size,))
        return key, val

    @staticmethod
    def _insert_heapify(heap: "BGPQ", block_key: chex.Array, block_val: Xtructurable):
        """
        Internal method to maintain heap property after insertion.
        Performs heapification by traversing up the tree and merging nodes.

        Args:
            heap: The priority queue instance
            block_key: Keys to insert
            block_val: Values to insert

        Returns:
            tuple containing:
                - Updated heap
                - Boolean indicating if insertion was successful
        """
        last_node = SIZE_DTYPE(heap.heap_size + 1)

        def _cond(var):
            """Continue while not reached last node"""
            _, _, _, n = var
            return n < last_node

        def insert_heapify(var):
            """Perform one step of heapification"""
            heap, keys, values, n = var
            head, hvalues, keys, values = merge_sort_split(
                heap.key_store[n], heap.val_store[n], keys, values
            )
            heap.key_store = heap.key_store.at[n].set(head)
            heap.val_store = heap.val_store.at[n].set(hvalues)
            return heap, keys, values, _next(n, last_node)

        heap, keys, values, _ = jax.lax.while_loop(
            _cond,
            insert_heapify,
            (
                heap,
                block_key,
                block_val,
                _next(SIZE_DTYPE(0), last_node),
            ),
        )

        def _size_not_full(heap, keys, values):
            """Insert remaining elements if heap not full"""
            heap.key_store = heap.key_store.at[last_node].set(keys)
            heap.val_store = heap.val_store.at[last_node].set(values)
            return heap

        added = last_node < heap.branch_size
        heap = jax.lax.cond(
            added, _size_not_full, lambda heap, keys, values: heap, heap, keys, values
        )
        return heap, added

    @jax.jit
    def insert(heap: "BGPQ", block_key: chex.Array, block_val: Xtructurable):
        """
        Insert new elements into the priority queue.
        Maintains heap property through merge operations and heapification.

        Args:
            heap: The priority queue instance
            block_key: Keys to insert
            block_val: Values to insert
            added_size: Optional size of insertion (calculated if None)

        Returns:
            Updated heap instance
        """
        block_key, block_val = sort_arrays(block_key, block_val)
        # Merge with root node
        root_key, root_val, block_key, block_val = merge_sort_split(
            heap.key_store[0], heap.val_store[0], block_key, block_val
        )
        heap.key_store = heap.key_store.at[0].set(root_key)
        heap.val_store = heap.val_store.at[0].set(root_val)

        # Handle buffer overflow
        heap, block_key, block_val, buffer_overflow = heap.merge_buffer(block_key, block_val)

        # Perform heapification if needed
        heap, added = jax.lax.cond(
            buffer_overflow,
            BGPQ._insert_heapify,
            lambda heap, block_key, block_val: (heap, False),
            heap,
            block_key,
            block_val,
        )
        heap.heap_size = SIZE_DTYPE(heap.heap_size + added)
        return heap

    @staticmethod
    def delete_heapify(heap: "BGPQ"):
        """
        Maintain heap property after deletion of minimum elements.

        Args:
            heap: The priority queue instance

        Returns:
            Updated heap instance
        """

        last = heap.heap_size
        heap.heap_size = SIZE_DTYPE(last - 1)

        # Move last node to root and clear last position
        last_key = heap.key_store[last]
        last_val = heap.val_store[last]

        heap.key_store = heap.key_store.at[last].set(jnp.inf)

        root_key, root_val, heap.key_buffer, heap.val_buffer = merge_sort_split(
            last_key, last_val, heap.key_buffer, heap.val_buffer
        )

        heap.key_store = heap.key_store.at[0].set(root_key)
        heap.val_store = heap.val_store.at[0].set(root_val)

        def _lr(n):
            """Get left and right child indices"""
            left_child = n * 2 + 1
            right_child = n * 2 + 2
            return left_child, right_child

        def _cond(var):
            """Continue while heap property is violated"""
            heap, c, l, r = var
            max_c = heap.key_store[c][-1]
            min_l = heap.key_store[l][0]
            min_r = heap.key_store[r][0]
            min_lr = jnp.minimum(min_l, min_r)
            return max_c > min_lr

        def _f(var):
            """Perform one step of heapification"""
            heap, current_node, left_child, right_child = var
            max_left_child = heap.key_store[left_child][-1]
            max_right_child = heap.key_store[right_child][-1]

            # Choose child with smaller key
            x, y = jax.lax.cond(
                max_left_child > max_right_child,
                lambda _: (left_child, right_child),
                lambda _: (right_child, left_child),
                None,
            )

            # Merge and swap nodes
            ky, vy, kx, vx = merge_sort_split(
                heap.key_store[left_child],
                heap.val_store[left_child],
                heap.key_store[right_child],
                heap.val_store[right_child],
            )
            kc, vc, ky, vy = merge_sort_split(
                heap.key_store[current_node], heap.val_store[current_node], ky, vy
            )
            heap.key_store = heap.key_store.at[y].set(ky)
            heap.key_store = heap.key_store.at[current_node].set(kc)
            heap.key_store = heap.key_store.at[x].set(kx)
            heap.val_store = heap.val_store.at[y].set(vy)
            heap.val_store = heap.val_store.at[current_node].set(vc)
            heap.val_store = heap.val_store.at[x].set(vx)

            nc = y
            nl, nr = _lr(y)
            return heap, nc, nl, nr

        c = SIZE_DTYPE(0)
        l, r = _lr(c)
        heap, _, _, _ = jax.lax.while_loop(_cond, _f, (heap, c, l, r))
        return heap

    @jax.jit
    def delete_mins(heap: "BGPQ"):
        """
        Remove and return the minimum elements from the queue.

        Args:
            heap: The priority queue instance

        Returns:
            tuple containing:
                - Updated heap instance
                - Array of minimum keys removed
                - Xtructurable of corresponding values
        """
        min_keys = heap.key_store[0]
        min_values = heap.val_store[0]

        def make_empty(heap: "BGPQ"):
            """Handle case where heap becomes empty"""
            root_key, root_val, heap.key_buffer, heap.val_buffer = merge_sort_split(
                jnp.full_like(heap.key_store[0], jnp.inf),
                heap.val_store[0],
                heap.key_buffer,
                heap.val_buffer,
            )
            heap.key_store = heap.key_store.at[0].set(root_key)
            heap.val_store = heap.val_store.at[0].set(root_val)
            heap.buffer_size = SIZE_DTYPE(0)
            return heap

        heap = jax.lax.cond(heap.heap_size == 0, make_empty, BGPQ.delete_heapify, heap)
        return heap, min_keys, min_values
