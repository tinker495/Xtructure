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


def _heapify_rounds(branch_size: int) -> int:
    """Interior levels on any root->leaf path: root is merged before heapify and
    the target node after it, so only depth-2 rounds can ever be active."""
    return max(int(branch_size).bit_length() - 2, 0)


def _bgpq_insert_heapify_internal(
    heap: Any, block_key: chex.Array, block_val: Xtructurable, active: chex.Array
):
    """Sink `block` down the heap when `active`; an exact no-op otherwise.

    Callers pass the buffer-overflow flag as `active` instead of guarding this
    function with lax.cond: on GPU the conditional costs a host predicate
    readback per insert, while the masked rounds below are ~16us of merge work
    each inside one command buffer.
    """
    is_full = heap.heap_size >= (heap.branch_size - 1)

    # Worst-leaf argmax is a ~branch_size-element reduce; computing it always
    # and selecting with where is cheaper than a lax.cond host predicate sync.
    worst_leaf = jnp.argmax(heap.key_store[:, -1]).astype(SIZE_DTYPE)
    last_node = jnp.where(is_full, worst_leaf, (heap.heap_size + 1).astype(SIZE_DTYPE))

    def insert_heapify(var):
        """One masked step down the root->last_node path.

        The path length is data dependent (log2 of the live branch count), but a
        data-dependent while_loop pays a host predicate readback per level (~100us
        of idle GPU for ~16us of merge work). The static bound is the deepest
        possible path for this heap's branch_size; rounds past the real path write
        every row/carry back to itself, so the result is bit-identical to the loop.
        """
        heap, keys, values, n = var
        active = jnp.logical_and(active_insert, n < last_node)
        row_key = heap.key_store[n]
        row_val = heap.val_store[n]
        head, hvalues, merged_keys, merged_values = merge_sort_split(row_key, row_val, keys, values)
        heap = heap.replace(
            key_store=heap.key_store.at[n].set(jnp.where(active, head, row_key)),
            val_store=heap.val_store.at[n].set(xnp_where(active, hvalues, row_val)),
        )
        keys = jnp.where(active, merged_keys, keys)
        values = xnp_where(active, merged_values, values)
        return heap, keys, values, jnp.where(active, _next(n, last_node), n)

    active_insert = active
    # Static trip count: lowers to an XLA while with a known trip count, which the
    # GPU runtime executes without host predicate readbacks; keeps the HLO small
    # (a Python unroll of the same rounds compiled ~45% slower at the benchmark
    # scale for no runtime gain).
    var = (heap, block_key, block_val, _next(SIZE_DTYPE(0), last_node))
    heap, keys, values, _ = jax.lax.fori_loop(
        0, _heapify_rounds(heap.branch_size), lambda _, v: insert_heapify(v), var
    )

    def _update_node(heap, keys, values):
        # Merge with existing content (handles both Inf for new nodes and values for eviction)
        row_key = heap.key_store[last_node]
        row_val = heap.val_store[last_node]
        head, hvalues, _, _ = merge_sort_split(row_key, row_val, keys, values)
        new_key_store = heap.key_store.at[last_node].set(jnp.where(active_insert, head, row_key))
        new_val_store = heap.val_store.at[last_node].set(xnp_where(active_insert, hvalues, row_val))
        return heap.replace(key_store=new_key_store, val_store=new_val_store)

    # last_node < branch_size always holds: is_full picks an argmax leaf
    # (< branch_size) and otherwise heap_size <= branch_size - 2 makes
    # heap_size + 1 <= branch_size - 1. The old lax.cond guard was a pure
    # host-sync tax on an invariantly-true predicate.
    heap = _update_node(heap, keys, values)

    # Only increment size if we filled a NEW node (not full)
    added = jnp.logical_and(active_insert, ~is_full)
    return heap, added


@jax.jit
def _bgpq_insert_sorted_jit(heap: Any, block_key: chex.Array, block_val: Xtructurable):
    # Merge with root node
    root_key, root_val, block_key, block_val = merge_sort_split(
        heap.key_store[0], heap.val_store[0], block_key, block_val
    )
    heap = heap.replace(
        key_store=heap.key_store.at[0].set(root_key), val_store=heap.val_store.at[0].set(root_val)
    )

    # Handle buffer overflow
    heap, block_key, block_val, buffer_overflow = _bgpq_merge_buffer_jit(heap, block_key, block_val)

    # Heapify only when the buffer overflowed; masked instead of lax.cond so the
    # whole insert stays one command buffer (no host predicate readback).
    heap, added = _bgpq_insert_heapify_internal(heap, block_key, block_val, buffer_overflow)
    heap = heap.replace(heap_size=SIZE_DTYPE(heap.heap_size + added.astype(SIZE_DTYPE)))
    return heap


@jax.jit
def _bgpq_insert_jit(heap: Any, block_key: chex.Array, block_val: Xtructurable):
    block_key, block_val = sort_arrays(block_key, block_val)
    return _bgpq_insert_sorted_jit(heap, block_key, block_val)


def _insert_rows_wavefront(heap: Any, block_keys: chex.Array, block_vals: Xtructurable, n_rows):
    """Sorted rows ``[0, n_rows)`` inserted with every row's sink running concurrently.

    Bit-identical to the sequential row loop as long as no row sees a full heap
    (the caller's guard): then every row targets the next new node, so targets
    are known up front. Phase A runs each row's root and buffer merges in order;
    they touch only node 0 and the buffer. Phase B starts row j's sink at depth 1
    on step j, so live rows sit at distinct depths (never the same node) and every
    node sees rows in order. A row sinks on steps [j, j + rounds], so at most
    ``rounds + 1`` rows are live at once: each step merges only that lane window,
    and both loops stop at the last inserted / overflowing row.
    """
    k = block_keys.shape[0]

    def root_and_buffer(j, carry):
        heap, keys, vals, overflow = carry
        root_key, root_val, key, val = merge_sort_split(
            heap.key_store[0], heap.val_store[0], block_keys[j], block_vals[j]
        )
        merged, key, val, buffer_overflow = _bgpq_merge_buffer_jit(heap, key, val)
        heap = heap.replace(
            key_store=heap.key_store.at[0].set(root_key),
            val_store=heap.val_store.at[0].set(root_val),
            key_buffer=merged.key_buffer,
            val_buffer=merged.val_buffer,
            buffer_size=merged.buffer_size,
        )
        overflow = overflow.at[j].set(buffer_overflow)
        return heap, keys.at[j].set(key), vals.at[j].set(val), overflow

    heap, keys, vals, active = jax.lax.fori_loop(
        0, n_rows, root_and_buffer, (heap, block_keys, block_vals, jnp.zeros((k,), jnp.bool_))
    )
    rounds = _heapify_rounds(heap.branch_size)
    width = min(k, rounds + 1)
    last = heap.heap_size + jnp.cumsum(active, dtype=SIZE_DTYPE)
    lane_where = jax.vmap(xnp_where)

    def sink(step, carry):
        heap, keys, vals, node, sinking = carry
        start = jnp.clip(step - rounds, 0, k - width)
        lane = start + jnp.arange(width, dtype=start.dtype)

        def window(x):
            return jax.lax.dynamic_slice_in_dim(x, start, width)

        def put(x, w):
            return jax.lax.dynamic_update_slice_in_dim(x, w, start, axis=0)

        w_keys, w_node, w_sinking, w_last = map(window, (keys, node, sinking, last))
        w_vals = jax.tree_util.tree_map(window, vals)
        live = jnp.logical_and(w_sinking, lane <= step)
        # Idle lanes merge an all-inf block, which takes the merge kernel's copy path.
        head_key, head_val, tail_key, tail_val = jax.vmap(merge_sort_split)(
            heap.key_store[w_node],
            heap.val_store[w_node],
            jnp.where(live[:, None], w_keys, jnp.inf),
            w_vals,
        )
        # Idle lanes write distinct out-of-range rows, which are dropped.
        rows = jnp.where(live, w_node, heap.branch_size + lane.astype(SIZE_DTYPE))

        def scatter(store, head):
            return store.at[rows].set(head, mode="drop", unique_indices=True)

        heap = heap.replace(
            key_store=scatter(heap.key_store, head_key),
            val_store=jax.tree_util.tree_map(scatter, heap.val_store, head_val),
        )
        at_last = w_node == w_last
        move = jnp.logical_and(live, ~at_last)
        return (
            heap,
            put(keys, lane_where(move, tail_key, w_keys)),
            jax.tree_util.tree_map(put, vals, lane_where(move, tail_val, w_vals)),
            put(node, jnp.where(move, _next(w_node, w_last), w_node)),
            put(sinking, jnp.logical_and(w_sinking, ~jnp.logical_and(live, at_last))),
        )

    last_active = jnp.max(jnp.where(active, jnp.arange(k, dtype=jnp.int32), -1))
    steps = jnp.where(last_active >= 0, last_active + 1 + rounds, 0)
    added = jnp.sum(active, dtype=SIZE_DTYPE)
    carry = (heap, keys, vals, _next(jnp.zeros((k,), SIZE_DTYPE), last), active)
    heap = jax.lax.fori_loop(0, steps, sink, carry)[0]
    return heap.replace(heap_size=SIZE_DTYPE(heap.heap_size + added))


@jax.jit
def _bgpq_insert_sorted_rows_jit(
    heap: Any, block_keys: chex.Array, block_vals: Xtructurable, n_rows: chex.Array
):
    n_rows = jnp.clip(n_rows, 0, block_keys.shape[0]).astype(jnp.int32)

    # Narrower rows are promoted to the store dtype; wider rows stay wide and are cast only when
    # stored. One insert per row leaves the buffer at the wider dtype, so promote it up front:
    # the wavefront and fallback loop carries then keep one dtype.
    def promote(x, other):
        return x.astype(jnp.promote_types(x.dtype, other.dtype))

    block_keys = promote(block_keys, heap.key_store)
    block_vals = jax.tree_util.tree_map(promote, block_vals, heap.val_store)
    heap = heap.replace(
        key_buffer=promote(heap.key_buffer, block_keys),
        val_buffer=jax.tree_util.tree_map(promote, heap.val_buffer, block_vals),
    )
    # Rows only see a full heap if heap_size can reach branch_size - 1 during the call;
    # those calls take the sequential row loop instead.
    fits = heap.heap_size + n_rows.astype(SIZE_DTYPE) < heap.branch_size
    # Both paths always run, the unused one with zero trips: a lax.cond here makes XLA
    # copy the whole store into and out of its branches (~1 ms per call measured).
    heap = _insert_rows_wavefront(heap, block_keys, block_vals, jnp.where(fits, n_rows, 0))
    return jax.lax.fori_loop(
        0,
        jnp.where(fits, 0, n_rows),
        lambda j, h: _bgpq_insert_sorted_jit(h, block_keys[j], block_vals[j]),
        heap,
    )


@jax.jit
def _bgpq_insert_rows_jit(
    heap: Any, block_keys: chex.Array, block_vals: Xtructurable, n_rows: chex.Array
):
    block_keys, block_vals = jax.vmap(sort_arrays)(block_keys, block_vals)
    return _bgpq_insert_sorted_rows_jit(heap, block_keys, block_vals, n_rows)
