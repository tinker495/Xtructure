import jax
import jax.numpy as jnp

from xtructure.core.dtype_facts import DTypeKind, dtype_kind


def _avalanche32(h):
    """Avalanche mixer for 32-bit words."""
    h = jnp.uint32(h)
    h = h ^ (h >> 16)
    h = h * jnp.uint32(0x85EBCA77)
    h = h ^ (h >> 13)
    h = h * jnp.uint32(0xC2B2AE3D)
    h = h ^ (h >> 16)
    return h


def _split_uint64_to_uint32(u64):
    """Split uint64 array into interleaved uint32 words."""
    lo = jnp.uint32(u64 & jnp.uint64(0xFFFFFFFF))
    hi = jnp.uint32(u64 >> jnp.uint64(32))
    return jnp.stack((lo, hi), axis=-1).reshape(-1)


def _pack_uint16_to_uint32(u16):
    """Pack uint16 array into uint32 lanes with zero-padding."""
    u16 = jnp.reshape(u16, (-1,))
    pad = (-u16.size) % 2
    if pad:
        u16 = jnp.pad(u16, (0, pad), mode="constant", constant_values=0)
    u16 = jnp.reshape(u16, (-1, 2))
    lo = u16[:, 0].astype(jnp.uint32)
    hi = u16[:, 1].astype(jnp.uint32)
    return (lo | (hi << 16)).reshape(-1)


@jax.jit
def hash_fast_uint32ed(uint32ed, seed=jnp.uint32(0)):
    """Vectorized hash reducer for uint32 streams."""
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


def uint32ed_to_hash(uint32ed, seed):
    """Convert uint32 array to hash value."""
    return hash_fast_uint32ed(uint32ed, seed)


@jax.jit
def _to_bytes(x):
    """Convert an array-like leaf to a byte array."""
    x = jnp.asarray(x)
    if dtype_kind(x.dtype) is DTypeKind.BOOL:
        x = x.astype(jnp.uint8)
    return jax.lax.bitcast_convert_type(x, jnp.uint8).reshape(-1)


@jax.jit
def _byterize(x):
    """Convert an entire state tree to a flattened byte array."""
    byte_leaves = jax.tree_util.tree_map(_to_bytes, x)
    flat_leaves, _ = jax.tree_util.tree_flatten(byte_leaves)
    if len(flat_leaves) == 0:
        return jnp.array([], dtype=jnp.uint8)
    return jnp.concatenate(flat_leaves)


def _to_uint32_from_bytes(byte_array):
    """Convert byte array to uint32 array with runtime-safe padding."""
    byte_array = jnp.asarray(byte_array, dtype=jnp.uint8).reshape(-1)
    bytes_len = byte_array.shape[0]
    if bytes_len == 0:
        return jnp.zeros((0,), dtype=jnp.uint32)

    pad_len = (-bytes_len) % 4
    if pad_len:
        byte_array = jnp.pad(byte_array, (0, pad_len), mode="constant", constant_values=0)

    chunks = jnp.reshape(byte_array, (-1, 4))
    # Fast-path: bitcast a (N, 4) uint8 buffer to (N,) uint32 without per-row vmap.
    # Verified by tests: jax.lax.bitcast_convert_type keeps leading axes and repacks trailing bytes.
    return jax.lax.bitcast_convert_type(chunks, jnp.uint32).reshape(-1)


def _leaf_to_uint32(leaf):
    """Convert a single leaf to uint32 representation."""
    if not hasattr(leaf, "dtype"):
        return _to_uint32_from_bytes(_to_bytes(leaf))

    dtype = leaf.dtype
    kind = dtype_kind(dtype)
    if kind is DTypeKind.BOOL:
        return _to_uint32_from_bytes(leaf.astype(jnp.uint8))

    if kind in (DTypeKind.UINT, DTypeKind.INT):
        bits = jnp.iinfo(dtype).bits
        if bits == 8:
            return _to_uint32_from_bytes(leaf.astype(jnp.uint8))
        if bits == 16:
            return _pack_uint16_to_uint32(leaf.astype(jnp.uint16))
        if bits == 32:
            return leaf.astype(jnp.uint32).reshape(-1)
        if bits == 64:
            return _split_uint64_to_uint32(leaf.astype(jnp.uint64))

    if kind is DTypeKind.FLOAT:
        if dtype == jnp.float32:
            return jax.lax.bitcast_convert_type(leaf, jnp.uint32).reshape(-1)
        if dtype == jnp.float64:
            return _split_uint64_to_uint32(jax.lax.bitcast_convert_type(leaf, jnp.uint64))
        if dtype in (jnp.float16, jnp.bfloat16):
            return _pack_uint16_to_uint32(jax.lax.bitcast_convert_type(leaf, jnp.uint16))

    raise TypeError(f"Unsupported DType Kind for hash byte encoding: {dtype!r}.")


@jax.jit
def _to_uint32(x):
    """Convert a PyTree to a flat uint32 array."""
    uint32_leaves = jax.tree_util.tree_map(_leaf_to_uint32, x)
    flat_leaves, _ = jax.tree_util.tree_flatten(uint32_leaves)
    if len(flat_leaves) == 0:
        return jnp.zeros((0,), dtype=jnp.uint32)
    return jnp.concatenate(flat_leaves)


@jax.jit
def _h(x, seed=0):
    """Hash a state tree by first converting it to uint32 lanes."""
    return uint32ed_to_hash(_to_uint32(x), seed)


@jax.jit
def _h_with_uint32ed(x, seed=0):
    """Hash a state tree and return the uint32 representation used."""
    uint32ed = _to_uint32(x)
    return uint32ed_to_hash(uint32ed, seed), uint32ed


def hash_function_decorator(cls):
    """Attach module-level hash functions to a class."""
    setattr(cls, "bytes", property(_byterize))
    setattr(cls, "uint32ed", property(_to_uint32))
    setattr(cls, "hash", _h)
    setattr(cls, "hash_with_uint32ed", _h_with_uint32ed)
    return cls
