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
from ..core import xtructure_numpy as xnp
from ..core.xtructure_numpy.array_ops import _where_no_broadcast
from .merge_split import merge_arrays_parallel, merge_sort_split_idx

SORT_STABLE = True  # Use stable sorting to maintain insertion order for equal keys
SIZE_DTYPE = jnp.uint32

# TODO: Make merge_arrays_parallel for TPU.
merge_array_backend = (
    merge_sort_split_idx if jax.default_backend() == "tpu" else merge_arrays_parallel
)


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
    val = xnp.concatenate([av, bv], axis=0)
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
    This implementation handles the 0-indexed heap structure by temporarily
    converting to 1-based indices for the underlying bitwise logic.

    Args:
        current: Current index in the heap
        target: Target index to reach

    Returns:
        Next index in the path from current to target
    """
    current_1based = current.astype(SIZE_DTYPE) + 1
    target_1based = target.astype(SIZE_DTYPE) + 1

    clz_current = jax.lax.clz(current_1based)
    clz_target = jax.lax.clz(target_1based)
    shift_amount = clz_current - clz_target - 1

    next_index_1based = target_1based >> shift_amount
    return next_index_1based - 1


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


@jax.jit
def _bgpq_merge_buffer_jit(heap: "BGPQ", blockk: chex.Array, blockv: Xtructurable):
    n = int(heap.batch_size)
    # Concatenate block and buffer
    sorted_key, sorted_idx = merge_array_backend(blockk, heap.key_buffer)
    val = xnp.concatenate([blockv, heap.val_buffer], axis=0)
    val = val[sorted_idx]

    # Check for active elements (non-infinity)
    filled = jnp.isfinite(sorted_key)
    n_filled = jnp.sum(filled, dtype=SIZE_DTYPE)
    buffer_overflow = n_filled >= n

    def overflowed(key, val):
        """Handle case where buffer overflows"""
        return key[:n], val[:n], key[n:], val[n:]

    def not_overflowed(key, val):
        return key[-n:], val[-n:], key[:-n], val[:-n]

    blockk, blockv, key_buffer, val_buffer = jax.lax.cond(
        buffer_overflow,
        overflowed,
        not_overflowed,
        sorted_key,
        val,
    )

    buffer_size = jnp.sum(jnp.isfinite(key_buffer), dtype=SIZE_DTYPE)
    heap = heap.replace(key_buffer=key_buffer, val_buffer=val_buffer, buffer_size=buffer_size)
    return heap, blockk, blockv, buffer_overflow


@partial(jax.jit, static_argnums=(2))
def _bgpq_make_batched_jit(key: chex.Array, val: Xtructurable, batch_size: int):
    n = key.shape[0]
    # Pad arrays to match batch size
    key = jnp.pad(key, (0, batch_size - n), mode="constant", constant_values=jnp.inf)
    val = xnp.pad(val, (0, batch_size - n))
    return key, val


def _bgpq_insert_heapify_internal(heap: "BGPQ", block_key: chex.Array, block_val: Xtructurable):
    is_full = heap.heap_size >= (heap.branch_size - 1)

    def _get_target_full(h):
        # Find the leaf with the largest max key (worst leaf) to challenge
        return jnp.argmax(h.key_store[:, -1]).astype(SIZE_DTYPE)

    last_node = jax.lax.cond(is_full, _get_target_full, lambda h: SIZE_DTYPE(h.heap_size + 1), heap)

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
        # To maintain purity, we need to update the heap instance properly.
        # But here heap is being updated in a loop.
        # We can update the components.
        new_key_store = heap.key_store.at[n].set(head)
        new_val_store = heap.val_store.at[n].set(hvalues)
        heap = heap.replace(key_store=new_key_store, val_store=new_val_store)
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

    def _update_node(heap, keys, values):
        # Merge with existing content (handles both Inf for new nodes and values for eviction)
        head, hvalues, _, _ = merge_sort_split(
            heap.key_store[last_node], heap.val_store[last_node], keys, values
        )
        new_key_store = heap.key_store.at[last_node].set(head)
        new_val_store = heap.val_store.at[last_node].set(hvalues)
        return heap.replace(key_store=new_key_store, val_store=new_val_store)

    # If last_node is valid, we update it.
    # If is_full, last_node is a valid leaf.
    # If not full, last_node is the next slot.
    valid_node = last_node < heap.branch_size

    heap = jax.lax.cond(valid_node, _update_node, lambda heap, k, v: heap, heap, keys, values)

    # Only increment size if we filled a NEW node (not full and valid)
    added = valid_node & (~is_full)
    return heap, added


@jax.jit
def _bgpq_insert_jit(heap: "BGPQ", block_key: chex.Array, block_val: Xtructurable):
    block_key, block_val = sort_arrays(block_key, block_val)
    # Merge with root node
    root_key, root_val, block_key, block_val = merge_sort_split(
        heap.key_store[0], heap.val_store[0], block_key, block_val
    )
    heap = heap.replace(
        key_store=heap.key_store.at[0].set(root_key), val_store=heap.val_store.at[0].set(root_val)
    )

    # Handle buffer overflow
    heap, block_key, block_val, buffer_overflow = _bgpq_merge_buffer_jit(heap, block_key, block_val)

    # Perform heapification if needed
    heap, added = jax.lax.cond(
        buffer_overflow,
        _bgpq_insert_heapify_internal,
        lambda heap, block_key, block_val: (heap, False),
        heap,
        block_key,
        block_val,
    )
    heap = heap.replace(heap_size=SIZE_DTYPE(heap.heap_size + added))
    return heap


def _bgpq_delete_heapify_internal(heap: "BGPQ"):
    last = heap.heap_size
    heap = heap.replace(heap_size=SIZE_DTYPE(last - 1))

    # Move last node to root and clear last position
    last_key = heap.key_store[last]
    last_val = heap.val_store[last]

    root_key, root_val, key_buffer, val_buffer = merge_sort_split(
        last_key, last_val, heap.key_buffer, heap.val_buffer
    )
    heap = heap.replace(key_buffer=key_buffer, val_buffer=val_buffer)

    inf_row = jnp.full_like(last_key, jnp.inf)
    key_indices = jnp.array([last, SIZE_DTYPE(0)], dtype=jnp.int32)
    key_updates = jnp.stack((inf_row, root_key), axis=0)
    heap = heap.replace(
        key_store=heap.key_store.at[key_indices].set(key_updates),
        val_store=heap.val_store.at[0].set(root_val),
    )

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
            lambda: (left_child, right_child),
            lambda: (right_child, left_child),
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
        key_indices = jnp.stack((y, current_node, x)).astype(jnp.int32)
        key_updates = jnp.stack((ky, kc, kx), axis=0)
        new_key_store = heap.key_store.at[key_indices].set(key_updates)

        val_indices = key_indices
        val_updates = xnp.stack((vy, vc, vx), axis=0)
        new_val_store = heap.val_store.at[val_indices].set(val_updates)

        heap = heap.replace(key_store=new_key_store, val_store=new_val_store)

        nc = y
        nl, nr = _lr(y)
        return heap, nc, nl, nr

    c = SIZE_DTYPE(0)
    l, r = _lr(c)
    heap, _, _, _ = jax.lax.while_loop(_cond, _f, (heap, c, l, r))
    return heap


@jax.jit
def _bgpq_delete_mins_jit(heap: "BGPQ"):
    min_keys = heap.key_store[0]
    min_values = heap.val_store[0]

    def make_empty(heap: "BGPQ"):
        """Handle case where heap becomes empty"""
        root_key, root_val, key_buffer, val_buffer = merge_sort_split(
            jnp.full_like(heap.key_store[0], jnp.inf),
            heap.val_store[0],
            heap.key_buffer,
            heap.val_buffer,
        )
        heap = heap.replace(
            key_store=heap.key_store.at[0].set(root_key),
            val_store=heap.val_store.at[0].set(root_val),
            buffer_size=SIZE_DTYPE(0),
            key_buffer=key_buffer,
            val_buffer=val_buffer,
        )
        return heap

    heap = jax.lax.cond(heap.heap_size == 0, make_empty, _bgpq_delete_heapify_internal, heap)
    return heap, min_keys, min_values


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
