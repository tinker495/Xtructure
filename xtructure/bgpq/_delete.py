"""Delete path for BGPQ."""

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from ..core import xtructure_numpy as xnp
from ._constants import SIZE_DTYPE
from ._merge import merge_sort_split

if TYPE_CHECKING:
    from .bgpq import BGPQ


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
