"""Insert path for BGPQ."""

from functools import partial
from typing import TYPE_CHECKING

import chex
import jax
import jax.numpy as jnp

from ..core import Xtructurable
from ..core import xtructure_numpy as xnp
from ._constants import SIZE_DTYPE, merge_array_backend
from ._merge import merge_sort_split
from ._utils import _next, sort_arrays

if TYPE_CHECKING:
    from .bgpq import BGPQ


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


@jax.jit
def _bgpq_make_batched_like_jit(heap: "BGPQ", key: chex.Array, val: Xtructurable):
    batch_size = int(heap.batch_size)
    n = key.shape[0]
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
