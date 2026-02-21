"""Insert path for BGPQ."""

import math
import os
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import chex
import jax
import jax.numpy as jnp

from ..core import Xtructurable
from ..core import xtructure_numpy as xnp
from ._constants import SIZE_DTYPE, merge_array_backend
from ._merge import _gather_sorted_values, merge_sort_split
from ._utils import _scatter_update_rows, _use_kv_backend, sort_arrays
from .merge_split.parallel import merge_arrays_parallel_kv

if TYPE_CHECKING:
    from .bgpq import BGPQ


@jax.jit
def _bgpq_merge_buffer_jit(heap: "BGPQ", blockk: chex.Array, blockv: Xtructurable):
    n = int(heap.batch_size)
    buffer_backend = os.environ.get("XTRUCTURE_BGPQ_MERGE_VALUE_BACKEND_BUFFER")
    if buffer_backend is None:
        buffer_backend = os.environ.get("XTRUCTURE_BGPQ_MERGE_VALUE_BACKEND", "")

    use_parallel_values = _use_kv_backend(buffer_backend, n, context="BACKEND_BUFFER")

    if use_parallel_values and jax.default_backend() == "gpu":
        sorted_key, val = merge_arrays_parallel_kv(
            blockk,
            blockv,
            heap.key_buffer,
            heap.val_buffer,
        )
    else:
        sorted_key, sorted_idx = merge_array_backend(blockk, heap.key_buffer)
        val = _gather_sorted_values(blockv, heap.val_buffer, sorted_idx)

    val = cast(Xtructurable, val)
    val_any = cast(Any, val)

    # Check for active elements (non-infinity)
    filled = jnp.isfinite(sorted_key)
    n_filled = jnp.sum(filled, dtype=SIZE_DTYPE)
    buffer_overflow = n_filled >= n

    def overflowed(key, values):
        """Handle case where buffer overflows"""
        values_any = cast(Any, values)
        return key[:n], values_any[:n], key[n:], values_any[n:]

    def not_overflowed(key, values):
        values_any = cast(Any, values)
        return key[-n:], values_any[-n:], key[:-n], values_any[:-n]

    blockk, blockv, key_buffer, val_buffer = jax.lax.cond(
        buffer_overflow,
        overflowed,
        not_overflowed,
        sorted_key,
        val_any,
    )

    buffer_size = jnp.sum(jnp.isfinite(key_buffer), dtype=SIZE_DTYPE)
    heap = heap.replace(
        key_buffer=key_buffer, val_buffer=val_buffer, buffer_size=buffer_size
    )
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


def _bgpq_insert_heapify_internal(
    heap: "BGPQ", block_key: chex.Array, block_val: Xtructurable
):
    is_full = heap.heap_size >= (heap.branch_size - 1)

    def _get_target_full(h):
        # Find the leaf with the largest max key (worst leaf) to challenge
        return jnp.argmax(h.key_store[:, -1]).astype(SIZE_DTYPE)

    last_node = jax.lax.cond(
        is_full, _get_target_full, lambda h: SIZE_DTYPE(h.heap_size + 1), heap
    )

    max_depth = int(math.ceil(math.log2(int(heap.branch_size) + 1)))

    def _path_indices(target):
        node = target.astype(SIZE_DTYPE) + SIZE_DTYPE(1)
        bit_width = jnp.iinfo(SIZE_DTYPE).bits
        depth = (bit_width - 1) - jax.lax.clz(node)
        levels = jnp.arange(max_depth, dtype=SIZE_DTYPE)
        shift = jnp.maximum(depth - levels, SIZE_DTYPE(0))
        path_1b = node >> shift
        path_0b = path_1b - SIZE_DTYPE(1)
        mask = levels <= depth
        return path_0b.astype(jnp.int32), mask

    def _update_root_only(heap, keys, values):
        head, hvalues, _, _ = merge_sort_split(
            heap.key_store[SIZE_DTYPE(0)], heap.val_store[SIZE_DTYPE(0)], keys, values
        )
        new_key_store = heap.key_store.at[SIZE_DTYPE(0)].set(head)
        new_val_store = heap.val_store.at[SIZE_DTYPE(0)].set(hvalues)
        return heap.replace(key_store=new_key_store, val_store=new_val_store)

    def _update_path(heap, keys, values):
        path_indices, mask = _path_indices(last_node)
        mask = jnp.logical_and(mask, jnp.arange(max_depth) > 0)
        safe_indices = jnp.where(mask, path_indices, jnp.zeros_like(path_indices))
        path_keys = heap.key_store[safe_indices]
        path_vals = heap.val_store[safe_indices]

        def scan_body(carry, xs):
            node_keys, node_vals, do_merge = xs

            def merge_case(_):
                head, hvalues, next_keys, next_vals = merge_sort_split(
                    node_keys, node_vals, carry[0], carry[1]
                )
                return (next_keys, next_vals), (head, hvalues)

            def skip_case(_):
                return carry, (node_keys, node_vals)

            return jax.lax.cond(do_merge, merge_case, skip_case, operand=None)

        (_, _), (head_keys, head_vals) = jax.lax.scan(
            scan_body, (keys, values), (path_keys, path_vals, mask)
        )

        scatter_mask = jnp.logical_or(mask, jnp.arange(max_depth) == 0)
        oob = jnp.asarray(heap.branch_size, dtype=path_indices.dtype)
        scatter_indices = jnp.where(scatter_mask, path_indices, oob)

        new_key_store = _scatter_update_rows(
            heap.key_store,
            scatter_indices,
            head_keys,
            indices_are_sorted=True,
        )
        new_val_store = jax.tree_util.tree_map(
            lambda store, updates: _scatter_update_rows(
                store,
                scatter_indices,
                updates,
                indices_are_sorted=True,
            ),
            heap.val_store,
            head_vals,
        )
        return heap.replace(key_store=new_key_store, val_store=new_val_store)

    def _update_if_valid(heap, keys, values):
        return jax.lax.cond(
            last_node == SIZE_DTYPE(0),
            _update_root_only,
            _update_path,
            heap,
            keys,
            values,
        )

    valid_node = last_node < heap.branch_size
    heap = jax.lax.cond(
        valid_node, _update_if_valid, lambda h, k, v: h, heap, block_key, block_val
    )

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
        key_store=heap.key_store.at[0].set(root_key),
        val_store=heap.val_store.at[0].set(root_val),
    )

    # Handle buffer overflow
    heap, block_key, block_val, buffer_overflow = _bgpq_merge_buffer_jit(
        heap, block_key, block_val
    )

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
