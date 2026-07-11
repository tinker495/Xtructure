"""Byte-packing of Xtructurable batches into single row-major uint8 buffers.

Linear containers (Queue, Stack) pay one GPU kernel per pytree leaf for
every functional update; on submission-latency-bound hosts (e.g. WSL2) that
per-leaf kernel count — not kernel time — dominates op wall time. Packing a
value batch into one ``uint8[batch, row_bytes]`` buffer collapses the
update to a single dynamic slice write regardless of leaf count.

Round-tripping is bit-exact: every leaf is bitcast (never converted) to
bytes, except bool leaves which are widened to uint8 (0/1) and narrowed
back, which is also exact.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .protocol import Xtructurable


class _LeafSpec(NamedTuple):
    offset: int
    nbytes: int
    dtype: Any
    inner_shape: tuple[int, ...]


class RowSpec(NamedTuple):
    row_bytes: int
    leaves: tuple[_LeafSpec, ...]
    treedef: Any


@lru_cache(maxsize=None)
def row_spec(value_class: Any) -> RowSpec:
    """Static per-leaf byte layout for one batch row of ``value_class``."""
    template = value_class.default((1,))
    flat, treedef = jax.tree_util.tree_flatten(template)
    leaves = []
    offset = 0
    for leaf in flat:
        inner_shape = tuple(int(d) for d in leaf.shape[1:])
        count = int(np.prod(inner_shape, dtype=np.int64)) if inner_shape else 1
        nbytes = count * int(np.dtype(leaf.dtype).itemsize)
        leaves.append(_LeafSpec(offset, nbytes, leaf.dtype, inner_shape))
        offset += nbytes
    return RowSpec(row_bytes=offset, leaves=tuple(leaves), treedef=treedef)


def pack_rows(value_class: Any, items: Xtructurable) -> jnp.ndarray:
    """Pack a batched Xtructurable into ``uint8[batch, row_bytes]``."""
    flat = jax.tree_util.tree_leaves(items)
    batch = flat[0].shape[0]
    rows = []
    for leaf in flat:
        if leaf.dtype == jnp.bool_:
            leaf = leaf.astype(jnp.uint8)
        rows.append(jax.lax.bitcast_convert_type(leaf, jnp.uint8).reshape(batch, -1))
    return jnp.concatenate(rows, axis=1)


def unpack_rows(value_class: Any, packed: jnp.ndarray) -> Xtructurable:
    """Inverse of :func:`pack_rows` for a ``uint8[batch, row_bytes]`` buffer."""
    spec = row_spec(value_class)
    batch = packed.shape[0]
    out = []
    for leaf in spec.leaves:
        end = leaf.offset + leaf.nbytes
        chunk = packed[:, leaf.offset : end]
        if leaf.dtype == jnp.bool_:
            arr = chunk.reshape((batch,) + leaf.inner_shape).astype(jnp.bool_)
        else:
            itemsize = int(np.dtype(leaf.dtype).itemsize)
            arr = jax.lax.bitcast_convert_type(
                chunk.reshape(batch, -1, itemsize), leaf.dtype
            ).reshape((batch,) + leaf.inner_shape)
        out.append(arr)
    return jax.tree_util.tree_unflatten(spec.treedef, out)
