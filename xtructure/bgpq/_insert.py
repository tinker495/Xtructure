"""Insert path for BGPQ."""

from functools import partial
from typing import Any

import chex
import jax
import jax.numpy as jnp

from ..core.dtype_facts import SIZE_DTYPE
from ..core.protocol import Xtructurable
from ..core.xtructure_numpy import concatenate as xnp_concatenate
from ..core.xtructure_numpy import pad
from ..core.xtructure_numpy import where as xnp_where
from ._backend import merge_array_backend
from ._merge import merge_sort_split

SORT_STABLE = True  # Use stable sorting to maintain insertion order for equal keys.


def sort_arrays(k: chex.Array, v: Xtructurable):
    sorted_k, sorted_idx = jax.lax.sort_key_val(k, jnp.arange(k.shape[0]), is_stable=SORT_STABLE)
    sorted_v = v[sorted_idx]
    return sorted_k, sorted_v


@jax.jit
def _next(current, target):
    # 0-indexed binary heap navigation via clz on 1-based indices.
    current_1based = current.astype(SIZE_DTYPE) + 1
    target_1based = target.astype(SIZE_DTYPE) + 1

    clz_current = jax.lax.clz(current_1based)
    clz_target = jax.lax.clz(target_1based)
    shift_amount = clz_current - clz_target - 1

    next_index_1based = target_1based >> shift_amount
    return next_index_1based - 1


@jax.jit
def _bgpq_merge_buffer_jit(heap: Any, blockk: chex.Array, blockv: Xtructurable):
    n = int(heap.batch_size)
    # Concatenate block and buffer
    sorted_key, sorted_idx = merge_array_backend(blockk, heap.key_buffer)
    val = xnp_concatenate([blockv, heap.val_buffer], axis=0)
    val = val[sorted_idx]

    # Check for active elements (non-infinity)
    n_filled = jnp.sum(jnp.isfinite(sorted_key), dtype=SIZE_DTYPE)
    buffer_overflow = n_filled >= n

    # Pure window-selection on the sorted arrays (overflowed keeps the small
    # front window as the block, otherwise the block is the large tail
    # window; the buffer gets the complement). A lax.cond here costs a host
    # predicate readback (sync) per insert call; where selects the same bits.
    blockk = jnp.where(buffer_overflow, sorted_key[:n], sorted_key[-n:])
    blockv = xnp_where(buffer_overflow, val[:n], val[-n:])
    key_buffer = jnp.where(buffer_overflow, sorted_key[n:], sorted_key[:-n])
    val_buffer = xnp_where(buffer_overflow, val[n:], val[:-n])

    buffer_size = jnp.where(buffer_overflow, n_filled - n, n_filled).astype(SIZE_DTYPE)
    heap = heap.replace(key_buffer=key_buffer, val_buffer=val_buffer, buffer_size=buffer_size)
    return heap, blockk, blockv, buffer_overflow


@partial(jax.jit, static_argnums=(2))
def _bgpq_make_batched_jit(key: chex.Array, val: Xtructurable, batch_size: int):
    n = key.shape[0]
    # Pad arrays to match batch size
    key = jnp.pad(key, (0, batch_size - n), mode="constant", constant_values=jnp.inf)
    val = pad(val, (0, batch_size - n))
    return key, val


@jax.jit
def _bgpq_make_batched_like_jit(heap: Any, key: chex.Array, val: Xtructurable):
    batch_size = int(heap.batch_size)
    n = key.shape[0]
    key = jnp.pad(key, (0, batch_size - n), mode="constant", constant_values=jnp.inf)
    val = pad(val, (0, batch_size - n))
    return key, val


def _bgpq_insert_heapify_internal(heap: Any, block_key: chex.Array, block_val: Xtructurable):
    is_full = heap.heap_size >= (heap.branch_size - 1)

    # Worst-leaf argmax is a ~branch_size-element reduce; computing it always
    # and selecting with where is cheaper than a lax.cond host predicate sync.
    worst_leaf = jnp.argmax(heap.key_store[:, -1]).astype(SIZE_DTYPE)
    last_node = jnp.where(is_full, worst_leaf, (heap.heap_size + 1).astype(SIZE_DTYPE))

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

    # last_node < branch_size always holds: is_full picks an argmax leaf
    # (< branch_size) and otherwise heap_size <= branch_size - 2 makes
    # heap_size + 1 <= branch_size - 1. The old lax.cond guard was a pure
    # host-sync tax on an invariantly-true predicate.
    heap = _update_node(heap, keys, values)

    # Only increment size if we filled a NEW node (not full)
    added = ~is_full
    return heap, added


@jax.jit
def _bgpq_insert_jit(heap: Any, block_key: chex.Array, block_val: Xtructurable):
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
