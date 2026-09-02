"""Delete path for BGPQ."""

from typing import Any

import jax
import jax.numpy as jnp

from ..core.dtype_facts import SIZE_DTYPE
from ..core.xtructure_numpy import concatenate as xnp_concatenate
from ..core.xtructure_numpy import stack as xnp_stack
from ..core.xtructure_numpy import where as xnp_where
from ._merge import merge_sort_split


def _sift_rounds(branch_size: int) -> int:
    """Levels below the root on the deepest possible path for this heap."""
    return max(int(branch_size).bit_length() - 1, 0)


def _bgpq_delete_heapify_internal(heap: Any):
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

    def _violated(heap, c, left, r):
        """Heap property broken between node c and its smaller child?"""
        max_c = heap.key_store[c][-1]
        min_lr = jnp.minimum(heap.key_store[left][0], heap.key_store[r][0])
        return max_c > min_lr

    def _round(var):
        """One masked sift-down step.

        The loop exits as soon as the heap property holds, which is data dependent,
        so it ran as a while_loop with one host predicate readback per level. The
        static bound is the deepest path for this branch_size; once `active` drops
        the rows are written back to themselves and the node index stays put, so
        the result is bit-identical to the early-exit loop.
        """
        heap, current_node, left_child, right_child, active = var
        active = jnp.logical_and(active, _violated(heap, current_node, left_child, right_child))
        max_left_child = heap.key_store[left_child][-1]
        max_right_child = heap.key_store[right_child][-1]

        # Choose child with smaller key
        swap = max_left_child > max_right_child
        x = jnp.where(swap, left_child, right_child)
        y = jnp.where(swap, right_child, left_child)

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
        key_updates = jnp.where(
            active, jnp.stack((ky, kc, kx), axis=0), heap.key_store[key_indices]
        )
        val_updates = xnp_where(
            active, xnp_stack((vy, vc, vx), axis=0), heap.val_store[key_indices]
        )
        heap = heap.replace(
            key_store=heap.key_store.at[key_indices].set(key_updates),
            val_store=heap.val_store.at[key_indices].set(val_updates),
        )

        nc = jnp.where(active, y, current_node)
        nl, nr = _lr(nc)
        return heap, nc, nl, nr, active

    c = SIZE_DTYPE(0)
    left, right = _lr(c)
    var = (heap, c, left, right, jnp.bool_(True))
    # Static trip count -> XLA while with a known trip count (no host readback).
    var = jax.lax.fori_loop(0, _sift_rounds(heap.branch_size), lambda _, v: _round(v), var)
    heap = var[0]
    return heap


@jax.jit
def _bgpq_delete_mins_jit(heap: Any):
    min_keys = heap.key_store[0]
    min_values = heap.val_store[0]

    def make_empty(heap: Any):
        """Handle case where heap becomes empty"""

        def clear_root(heap: Any):
            return heap.replace(
                key_store=heap.key_store.at[0].set(jnp.full_like(heap.key_store[0], jnp.inf)),
                buffer_size=SIZE_DTYPE(0),
                key_buffer=jnp.full_like(heap.key_buffer, jnp.inf),
            )

        def refill_root_from_buffer(heap: Any):
            root_key = jnp.concatenate(
                [heap.key_buffer, jnp.full_like(heap.key_store[0][:1], jnp.inf)],
                axis=0,
            )
            root_val = xnp_concatenate([heap.val_buffer, heap.val_store[0][:1]], axis=0)
            return heap.replace(
                key_store=heap.key_store.at[0].set(root_key),
                val_store=heap.val_store.at[0].set(root_val),
                buffer_size=SIZE_DTYPE(0),
                key_buffer=jnp.full_like(heap.key_buffer, jnp.inf),
            )

        return jax.lax.cond(heap.buffer_size == 0, clear_root, refill_root_from_buffer, heap)

    heap = jax.lax.cond(heap.heap_size == 0, make_empty, _bgpq_delete_heapify_internal, heap)
    return heap, min_keys, min_values
