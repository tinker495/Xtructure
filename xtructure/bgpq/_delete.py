"""Delete path for BGPQ."""

import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from ._constants import SIZE_DTYPE
from ._merge import merge_sort_split
from ._utils import _scatter_update_rows

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

    max_depth = int(math.ceil(math.log2(int(heap.branch_size) + 1)))
    max_updates = max_depth * 2 + 1

    def _init_buffers():
        update_idx = jnp.zeros((max_updates,), dtype=jnp.int32)
        update_keys = jnp.zeros(
            (max_updates,) + heap.key_store.shape[1:], heap.key_store.dtype
        )

        def _make_val_buf(leaf):
            return jnp.zeros((max_updates,) + leaf.shape[1:], leaf.dtype)

        update_vals = jax.tree_util.tree_map(_make_val_buf, heap.val_store)
        update_mask = jnp.zeros((max_updates,), dtype=jnp.bool_)
        return update_idx, update_keys, update_vals, update_mask

    def _write_update(
        write_pos, idx, keys, vals, update_idx, update_keys, update_vals, update_mask
    ):
        update_idx = update_idx.at[write_pos].set(idx.astype(jnp.int32))
        update_keys = update_keys.at[write_pos].set(keys)
        update_vals = jax.tree_util.tree_map(
            lambda buf, leaf: buf.at[write_pos].set(leaf), update_vals, vals
        )
        update_mask = update_mask.at[write_pos].set(True)
        return update_idx, update_keys, update_vals, update_mask, write_pos + 1

    def _cond(state):
        current_idx, current_keys, _, _, _, _, _, _ = state
        left_child, right_child = _lr(current_idx)
        max_c = current_keys[-1]
        min_l = heap.key_store[left_child][0]
        min_r = heap.key_store[right_child][0]
        min_lr = jnp.minimum(min_l, min_r)
        return max_c > min_lr

    def _body(state):
        (
            current_idx,
            current_keys,
            current_vals,
            update_idx,
            update_keys,
            update_vals,
            update_mask,
            write_pos,
        ) = state

        left_child, right_child = _lr(current_idx)
        max_left_child = heap.key_store[left_child][-1]
        max_right_child = heap.key_store[right_child][-1]

        x, y = jax.lax.cond(
            max_left_child > max_right_child,
            lambda: (left_child, right_child),
            lambda: (right_child, left_child),
        )

        ky, vy, kx, vx = merge_sort_split(
            heap.key_store[left_child],
            heap.val_store[left_child],
            heap.key_store[right_child],
            heap.val_store[right_child],
        )
        kc, vc, ky, vy = merge_sort_split(current_keys, current_vals, ky, vy)

        update_idx, update_keys, update_vals, update_mask, write_pos = _write_update(
            write_pos,
            current_idx,
            kc,
            vc,
            update_idx,
            update_keys,
            update_vals,
            update_mask,
        )
        update_idx, update_keys, update_vals, update_mask, write_pos = _write_update(
            write_pos,
            x,
            kx,
            vx,
            update_idx,
            update_keys,
            update_vals,
            update_mask,
        )

        return (
            y,
            ky,
            vy,
            update_idx,
            update_keys,
            update_vals,
            update_mask,
            write_pos,
        )

    update_idx, update_keys, update_vals, update_mask = _init_buffers()
    state = (
        SIZE_DTYPE(0),
        root_key,
        root_val,
        update_idx,
        update_keys,
        update_vals,
        update_mask,
        jnp.array(0, dtype=jnp.int32),
    )

    state = jax.lax.while_loop(_cond, _body, state)
    (
        current_idx,
        current_keys,
        current_vals,
        update_idx,
        update_keys,
        update_vals,
        update_mask,
        write_pos,
    ) = state

    update_idx, update_keys, update_vals, update_mask, _ = _write_update(
        write_pos,
        current_idx,
        current_keys,
        current_vals,
        update_idx,
        update_keys,
        update_vals,
        update_mask,
    )

    oob = jnp.asarray(heap.branch_size, dtype=update_idx.dtype)
    scatter_indices = jnp.where(update_mask, update_idx, oob)

    new_key_store = _scatter_update_rows(
        heap.key_store,
        scatter_indices,
        update_keys,
        indices_are_sorted=False,
    )
    new_val_store = jax.tree_util.tree_map(
        lambda store, updates: _scatter_update_rows(
            store,
            scatter_indices,
            updates,
            indices_are_sorted=False,
        ),
        heap.val_store,
        update_vals,
    )
    heap = heap.replace(key_store=new_key_store, val_store=new_val_store)
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

    heap = jax.lax.cond(
        heap.heap_size == 0, make_empty, _bgpq_delete_heapify_internal, heap
    )
    return heap, min_keys, min_values
