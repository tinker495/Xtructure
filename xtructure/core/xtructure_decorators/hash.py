import jax
import jax.numpy as jnp

from ..protocol import Xtructurable


def _avalanche32(h):
    """Avalanche mixer for 32-bit words."""
    h = jnp.uint32(h)
    h = h ^ (h >> 16)
    h = h * jnp.uint32(0x85EBCA77)
    h = h ^ (h >> 13)
    h = h * jnp.uint32(0xC2B2AE3D)
    h = h ^ (h >> 16)
    return h


def _mix_fingerprint(primary, secondary, length):
    """Mix two hash values and a length into a fingerprint."""
    # Constants from cityhash/farmhash-like mixing
    const_a = jnp.uint32(0x85EBCA6B)
    const_b = jnp.uint32(0xC2B2AE35)

    mix = jnp.asarray(primary, dtype=jnp.uint32)
    secondary = jnp.asarray(secondary, dtype=jnp.uint32)
    length = jnp.asarray(length, dtype=jnp.uint32)

    mix ^= jnp.uint32(0x9E3779B9)
    mix = jnp.uint32(mix + secondary * const_a + length * const_b)
    mix ^= mix >> 16
    mix *= jnp.uint32(0x7FEB352D)
    mix ^= mix >> 15
    return mix


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


def byterize_hash_func_builder(x: Xtructurable):
    """
    Build a hash function for the pytree.
    This function creates a JIT-compiled hash function that converts pytree leaves to
    uint32 lanes and then reduces them with a vectorized avalanche hash.

    Args:
        x: Example pytree to determine the structure
    Returns:
        JIT-compiled hash function that takes a pytree and seed
    """

    @jax.jit
    def _to_bytes(x):
        """Convert input to byte array."""
        # Check if x is a JAX boolean array and cast to uint8 if true
        if x.dtype == jnp.bool_:
            x = x.astype(jnp.uint8)
        return jax.lax.bitcast_convert_type(x, jnp.uint8).reshape(-1)

    @jax.jit
    def _byterize(x):
        """Convert entire state tree to flattened byte array."""
        x = jax.tree_util.tree_map(_to_bytes, x)
        x, _ = jax.tree_util.tree_flatten(x)
        if len(x) == 0:
            return jnp.array([], dtype=jnp.uint8)
        return jnp.concatenate(x)

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
        if dtype == jnp.bool_:
            return _to_uint32_from_bytes(leaf.astype(jnp.uint8))

        if jnp.issubdtype(dtype, jnp.integer):
            bits = jnp.iinfo(dtype).bits
            if bits == 8:
                return _to_uint32_from_bytes(leaf.astype(jnp.uint8))
            if bits == 16:
                return _pack_uint16_to_uint32(leaf.astype(jnp.uint16))
            if bits == 32:
                return leaf.astype(jnp.uint32).reshape(-1)
            if bits == 64:
                return _split_uint64_to_uint32(leaf.astype(jnp.uint64))

        if jnp.issubdtype(dtype, jnp.floating):
            if dtype == jnp.float32:
                return jax.lax.bitcast_convert_type(leaf, jnp.uint32).reshape(-1)
            if dtype == jnp.float64:
                return _split_uint64_to_uint32(jax.lax.bitcast_convert_type(leaf, jnp.uint64))
            if dtype in (jnp.float16, jnp.bfloat16):
                return _pack_uint16_to_uint32(jax.lax.bitcast_convert_type(leaf, jnp.uint16))

        return _to_uint32_from_bytes(_to_bytes(leaf))

    def _to_uint32(x):
        """Convert pytree to uint32 array."""
        uint32_leaves = jax.tree_util.tree_map(_leaf_to_uint32, x)
        flat_leaves, _ = jax.tree_util.tree_flatten(uint32_leaves)
        if len(flat_leaves) == 0:
            return jnp.zeros((0,), dtype=jnp.uint32)
        return jnp.concatenate(flat_leaves)

    def _h(x, seed=0):
        """Main hash function that converts state to uint32 lanes and hashes them."""
        return uint32ed_to_hash(_to_uint32(x), seed)

    def _h_with_uint32ed(x, seed=0):
        """
        Main hash function that converts state to uint32 lanes and hashes them.
        Returns both hash value and its uint32 representation.
        """
        uint32ed = _to_uint32(x)
        return uint32ed_to_hash(uint32ed, seed), uint32ed

    return jax.jit(_byterize), jax.jit(_to_uint32), jax.jit(_h), jax.jit(_h_with_uint32ed)


def hash_function_decorator(cls):
    """
    Decorator to add a hash function to a class.
    """
    _byterize, _to_uint32, _h, _h_with_uint32ed = byterize_hash_func_builder(cls)

    setattr(cls, "bytes", property(_byterize))
    setattr(cls, "uint32ed", property(_to_uint32))
    setattr(cls, "hash", _h)
    setattr(cls, "hash_with_uint32ed", _h_with_uint32ed)

    return cls
