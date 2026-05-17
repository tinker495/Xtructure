"""Instance Layout-aware hash adapter for xtructure dataclasses.

Owns both ends of the hash pipeline in one Module:

* **Value Uint32 Encoding** policy — ``leaf_to_bytes`` / ``tree_to_bytes`` for
  byte streams, ``leaf_to_uint32`` / ``tree_to_uint32`` for scalar uint32 lanes,
  ``batched_leaf_to_uint32_rows`` / ``batched_tree_to_uint32_rows`` for
  ``(batch, lanes)`` rows. Padding and lane packing stay parity-locked across
  the scalar and batched surfaces via a single **DType Kind** dispatch.

* Hash reducers — ``hash_fast_uint32ed`` for flat uint32 streams and
  ``hash_fast_uint32ed_batched`` for ``(n, k) -> (n,)``.

The decorator attaches four Instance Layout-aware members onto each class:
``bytes`` / ``uint32ed`` / ``hash`` / ``hash_with_uint32ed``. ``StructuredType``
dispatch happens in the polymorphic helpers below; per-``cls`` closures are
forbidden so hash semantics stay ``cls``-agnostic.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from xtructure.core.dtype_facts import DTypeKind, dtype_kind
from xtructure.core.structuredtype import StructuredType

# ---------------------------------------------------------------------------
# Low-level lane packers
# ---------------------------------------------------------------------------


def split_uint64_to_uint32(u64: Any) -> jnp.ndarray:
    """Split a ``uint64`` array into interleaved ``uint32`` words."""
    u64 = jnp.asarray(u64, dtype=jnp.uint64)
    lo = jnp.uint32(u64 & jnp.uint64(0xFFFFFFFF))
    hi = jnp.uint32(u64 >> jnp.uint64(32))
    return jnp.stack((lo, hi), axis=-1).reshape(-1)


def pack_uint16_to_uint32(u16: Any) -> jnp.ndarray:
    """Pack ``uint16`` values into ``uint32`` lanes with zero-padding."""
    u16 = jnp.asarray(u16, dtype=jnp.uint16).reshape(-1)
    pad = (-u16.size) % 2
    if pad:
        u16 = jnp.pad(u16, (0, pad), mode="constant", constant_values=0)
    u16 = jnp.reshape(u16, (-1, 2))
    lo = u16[:, 0].astype(jnp.uint32)
    hi = u16[:, 1].astype(jnp.uint32)
    return (lo | (hi << jnp.uint32(16))).reshape(-1)


@jax.jit
def leaf_to_bytes(leaf: Any) -> jnp.ndarray:
    """Convert an array-like leaf to a flat byte array."""
    leaf = jnp.asarray(leaf)
    if dtype_kind(leaf.dtype) is DTypeKind.BOOL:
        leaf = leaf.astype(jnp.uint8)
    return jax.lax.bitcast_convert_type(leaf, jnp.uint8).reshape(-1)


@jax.jit
def tree_to_bytes(value: Any) -> jnp.ndarray:
    """Convert an entire PyTree to a flattened byte array."""
    byte_leaves = jax.tree_util.tree_map(leaf_to_bytes, value)
    flat_leaves, _ = jax.tree_util.tree_flatten(byte_leaves)
    if len(flat_leaves) == 0:
        return jnp.array([], dtype=jnp.uint8)
    return jnp.concatenate(flat_leaves)


def bytes_to_uint32(byte_array: Any) -> jnp.ndarray:
    """Convert byte values to ``uint32`` lanes with runtime-safe padding."""
    byte_array = jnp.asarray(byte_array, dtype=jnp.uint8).reshape(-1)
    bytes_len = byte_array.shape[0]
    if bytes_len == 0:
        return jnp.zeros((0,), dtype=jnp.uint32)

    pad_len = (-bytes_len) % 4
    if pad_len:
        byte_array = jnp.pad(byte_array, (0, pad_len), mode="constant", constant_values=0)

    chunks = jnp.reshape(byte_array, (-1, 4))
    return jax.lax.bitcast_convert_type(chunks, jnp.uint32).reshape(-1)


def _uint_lane_policy(dtype):
    """Return ``(staging_cast, pack_mode)`` for a primitive JAX dtype.

    ``staging_cast(value)`` casts/bitcasts the value to the canonical staging
    dtype (u8 / u16 / u32 / u64).  ``pack_mode`` is one of:
    ``"passthrough"``, ``"byte_pack"``, ``"u16_pack"``, ``"u64_split"``.

    Raises ``TypeError`` for unknown DType Kind.
    """
    kind = dtype_kind(dtype)

    if kind is DTypeKind.BOOL:
        return (lambda v: jnp.asarray(v).astype(jnp.uint8)), "byte_pack"

    if kind in (DTypeKind.UINT, DTypeKind.INT):
        bits = jnp.iinfo(dtype).bits
        if bits == 8:
            return (lambda v: jnp.asarray(v).astype(jnp.uint8)), "byte_pack"
        if bits == 16:
            return (lambda v: jnp.asarray(v).astype(jnp.uint16)), "u16_pack"
        if bits == 32:
            return (lambda v: jnp.asarray(v).astype(jnp.uint32)), "passthrough"
        if bits == 64:
            return (lambda v: jnp.asarray(v).astype(jnp.uint64)), "u64_split"

    if kind is DTypeKind.FLOAT:
        if dtype == jnp.float32:
            return (
                lambda v: jax.lax.bitcast_convert_type(jnp.asarray(v), jnp.uint32)
            ), "passthrough"
        if dtype == jnp.float64:
            return (lambda v: jax.lax.bitcast_convert_type(jnp.asarray(v), jnp.uint64)), "u64_split"
        if dtype in (jnp.float16, jnp.bfloat16):
            return (lambda v: jax.lax.bitcast_convert_type(jnp.asarray(v), jnp.uint16)), "u16_pack"

    raise TypeError(f"Unsupported DType Kind for value uint32 encoding: {dtype!r}.")


def _scalar_pack(staged: jnp.ndarray, mode: str) -> jnp.ndarray:
    """Pack a staged scalar array into a 1-D ``uint32`` array."""
    if mode == "passthrough":
        return staged.reshape(-1)
    if mode == "byte_pack":
        return bytes_to_uint32(staged)
    if mode == "u16_pack":
        return pack_uint16_to_uint32(staged)
    # u64_split
    return split_uint64_to_uint32(staged)


def _row_pack(staged: jnp.ndarray, mode: str, batch_len: int) -> jnp.ndarray:
    """Pack a staged batched array into a ``(batch_len, lanes)`` ``uint32`` array."""
    if mode == "passthrough":
        return staged.reshape(batch_len, -1)
    if mode == "byte_pack":
        return pack_uint8_rows_to_uint32(staged.reshape(batch_len, -1))
    if mode == "u16_pack":
        return pack_uint16_rows_to_uint32(staged.reshape(batch_len, -1))
    # u64_split
    return split_uint64_rows_to_uint32(staged.reshape(batch_len, -1))


def leaf_to_uint32(leaf: Any) -> jnp.ndarray:
    """Convert a single PyTree leaf to the canonical ``uint32`` representation."""
    if not hasattr(leaf, "dtype"):
        return bytes_to_uint32(leaf_to_bytes(leaf))
    cast, mode = _uint_lane_policy(leaf.dtype)
    return _scalar_pack(cast(leaf), mode)


@jax.jit
def tree_to_uint32(value: Any) -> jnp.ndarray:
    """Convert a PyTree to a flat canonical ``uint32`` array."""
    uint32_leaves = jax.tree_util.tree_map(leaf_to_uint32, value)
    flat_leaves, _ = jax.tree_util.tree_flatten(uint32_leaves)
    if len(flat_leaves) == 0:
        return jnp.zeros((0,), dtype=jnp.uint32)
    return jnp.concatenate(flat_leaves)


def pack_uint8_rows_to_uint32(rows: Any) -> jnp.ndarray:
    """Pack ``(batch, bytes)`` ``uint8`` rows into per-row ``uint32`` lanes."""
    rows = jnp.asarray(rows, dtype=jnp.uint8)
    batch_len, row_len = rows.shape
    pad_len = (-row_len) % 4
    if pad_len:
        rows = jnp.pad(rows, ((0, 0), (0, pad_len)), mode="constant", constant_values=0)
    if rows.shape[1] == 0:
        return jnp.zeros((batch_len, 0), dtype=jnp.uint32)
    chunks = rows.reshape(batch_len, -1, 4)
    return jax.lax.bitcast_convert_type(chunks, jnp.uint32).reshape(batch_len, -1)


def pack_uint16_rows_to_uint32(rows: Any) -> jnp.ndarray:
    """Pack ``(batch, lanes)`` ``uint16`` rows into per-row ``uint32`` lanes."""
    rows = jnp.asarray(rows, dtype=jnp.uint16)
    batch_len, row_len = rows.shape
    pad_len = (-row_len) % 2
    if pad_len:
        rows = jnp.pad(rows, ((0, 0), (0, pad_len)), mode="constant", constant_values=0)
    if rows.shape[1] == 0:
        return jnp.zeros((batch_len, 0), dtype=jnp.uint32)
    pairs = rows.reshape(batch_len, -1, 2)
    lo = pairs[:, :, 0].astype(jnp.uint32)
    hi = pairs[:, :, 1].astype(jnp.uint32)
    return lo | (hi << jnp.uint32(16))


def split_uint64_rows_to_uint32(rows: Any) -> jnp.ndarray:
    """Split ``(batch, lanes)`` ``uint64`` rows into interleaved ``uint32`` lanes."""
    rows = jnp.asarray(rows, dtype=jnp.uint64)
    lo = (rows & jnp.uint64(0xFFFFFFFF)).astype(jnp.uint32)
    hi = (rows >> jnp.uint64(32)).astype(jnp.uint32)
    return jnp.stack((lo, hi), axis=-1).reshape(rows.shape[0], -1)


def batched_leaf_to_uint32_rows(leaf: Any, batch_len: int) -> jnp.ndarray:
    """Convert a batched PyTree leaf into row-wise canonical ``uint32`` lanes.

    Padding is applied per row and per leaf, matching the scalar ``uint32ed``
    representation produced by :func:`tree_to_uint32`.
    """
    leaf = jnp.asarray(leaf)
    if leaf.ndim == 0 or leaf.shape[0] != batch_len:
        raise ValueError(
            "default uint32 row encoding expects every dynamic leaf to carry "
            f"leading batch dimension {batch_len}; got leaf shape {leaf.shape}."
        )
    cast, mode = _uint_lane_policy(leaf.dtype)
    return _row_pack(cast(leaf), mode, batch_len)


def batched_tree_to_uint32_rows(value: Any, batch_len: int) -> jnp.ndarray:
    """Return row-wise canonical ``uint32`` keys for a batched PyTree."""
    uint32_leaves = [
        batched_leaf_to_uint32_rows(leaf, batch_len) for leaf in jax.tree_util.tree_leaves(value)
    ]
    if not uint32_leaves:
        return jnp.zeros((batch_len, 0), dtype=jnp.uint32)
    return jnp.concatenate(uint32_leaves, axis=1)


# ---------------------------------------------------------------------------
# Hash reducers
# ---------------------------------------------------------------------------


def _avalanche32(h):
    """Avalanche mixer for 32-bit words."""
    h = jnp.uint32(h)
    h = h ^ (h >> 16)
    h = h * jnp.uint32(0x85EBCA77)
    h = h ^ (h >> 13)
    h = h * jnp.uint32(0xC2B2AE3D)
    h = h ^ (h >> 16)
    return h


@jax.jit
def hash_fast_uint32ed(uint32ed, seed=jnp.uint32(0)):
    """Vectorized hash reducer for a flat uint32 stream."""
    uint32ed = jnp.asarray(uint32ed, dtype=jnp.uint32).reshape(-1)
    seed = jnp.uint32(seed)
    if uint32ed.size == 0:
        return _avalanche32(seed ^ jnp.uint32(0x9E3779B1))

    idx = jnp.arange(uint32ed.shape[0], dtype=jnp.uint32)
    salt = idx * jnp.uint32(0x9E3779B1)
    lanes = _avalanche32(uint32ed ^ salt ^ seed)
    combined = jnp.bitwise_xor.reduce(lanes)
    combined ^= jnp.uint32(uint32ed.shape[0] << 2)
    combined ^= seed
    return _avalanche32(combined)


@jax.jit
def hash_fast_uint32ed_batched(rows, seed=jnp.uint32(0)):
    """Row-wise hash reducer for (n, k) uint32 keys -> (n,) hashes.

    Algorithmically identical to ``vmap(hash_fast_uint32ed)`` but reduces along
    axis 1 directly to avoid the per-row vmap overhead.
    """
    rows = jnp.asarray(rows, dtype=jnp.uint32)
    if rows.ndim == 1:
        rows = rows[:, None]
    seed = jnp.uint32(seed)
    n, k = rows.shape
    if k == 0:
        scalar = _avalanche32(seed ^ jnp.uint32(0x9E3779B1))
        return jnp.full((n,), scalar, dtype=jnp.uint32)

    idx = jnp.arange(k, dtype=jnp.uint32)
    salt = idx * jnp.uint32(0x9E3779B1)
    lanes = _avalanche32(rows ^ salt[None, :] ^ seed)
    combined = jnp.bitwise_xor.reduce(lanes, axis=1)
    combined ^= jnp.uint32(k << 2)
    combined ^= seed
    return _avalanche32(combined)


# ---------------------------------------------------------------------------
# Polymorphic per-instance dispatch
# ---------------------------------------------------------------------------


def _unstructured_uint32ed_error(cls_name, shape):
    return TypeError(
        f"{cls_name}.uint32ed is undefined for UNSTRUCTURED instance " f"(shape: {shape})."
    )


def _uint32ed(self):
    """Polymorphic uint32 lanes dispatched by Instance Layout.

    SINGLE  -> 1-D ``(lanes,)`` via :func:`tree_to_uint32`.
    BATCHED -> ``(batch, lanes)`` via :func:`batched_tree_to_uint32_rows`.
    UNSTRUCTURED -> ``TypeError``.
    """
    st = self.structured_type
    if st is StructuredType.SINGLE:
        return tree_to_uint32(self)
    if st is StructuredType.BATCHED:
        return batched_tree_to_uint32_rows(self, self.shape.batch[0])
    raise _unstructured_uint32ed_error(type(self).__name__, self.shape)


def _h(self, seed=0):
    """Hash polymorphic by Instance Layout.

    SINGLE  -> scalar hash.
    BATCHED -> ``(batch,)`` hash array via :func:`hash_fast_uint32ed_batched`.
    UNSTRUCTURED -> ``TypeError``.
    """
    st = self.structured_type
    if st is StructuredType.SINGLE:
        return hash_fast_uint32ed(tree_to_uint32(self), seed)
    if st is StructuredType.BATCHED:
        rows = batched_tree_to_uint32_rows(self, self.shape.batch[0])
        return hash_fast_uint32ed_batched(rows, seed)
    raise _unstructured_uint32ed_error(type(self).__name__, self.shape)


def _h_with_uint32ed(self, seed=0):
    """Hash and uint32 lanes polymorphic by Instance Layout.

    Returns ``(hash, uint32ed)`` whose shapes mirror :func:`_h` and
    :func:`_uint32ed`.
    """
    st = self.structured_type
    if st is StructuredType.SINGLE:
        u = tree_to_uint32(self)
        return hash_fast_uint32ed(u, seed), u
    if st is StructuredType.BATCHED:
        u = batched_tree_to_uint32_rows(self, self.shape.batch[0])
        return hash_fast_uint32ed_batched(u, seed), u
    raise _unstructured_uint32ed_error(type(self).__name__, self.shape)


def hash_function_decorator(cls):
    """Attach Instance Layout-aware hash adapter members to ``cls``.

    All four members are module-level callables (no per-``cls`` closures), so
    hash semantics remain ``cls``-agnostic; the only ``cls``-specific
    behaviour is the dispatch on ``self.structured_type`` (read from the
    Layout Cache).
    """
    setattr(cls, "bytes", property(tree_to_bytes))
    setattr(cls, "uint32ed", property(_uint32ed))
    setattr(cls, "hash", _h)
    setattr(cls, "hash_with_uint32ed", _h_with_uint32ed)
    return cls
