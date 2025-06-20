import jax
import jax.numpy as jnp

from .. import Xtructurable


def rotl(x, n):
    """Rotate left operation for 32-bit integers."""
    return (x << n) | (x >> (32 - n))


@jax.jit
def xxhash(x, seed):
    """
    Implementation of xxHash algorithm for 32-bit integers.
    Args:
        x: Input value to hash
        seed: Seed value for hash function
    Returns:
        32-bit hash value
    """
    prime_1 = jnp.uint32(0x9E3779B1)
    prime_2 = jnp.uint32(0x85EBCA77)
    prime_3 = jnp.uint32(0xC2B2AE3D)
    prime_5 = jnp.uint32(0x165667B1)
    acc = jnp.uint32(seed) + prime_5
    for _ in range(4):
        lane = x & 255
        acc = acc + lane * prime_5
        acc = rotl(acc, 11) * prime_1
        x = x >> 8
    acc = acc ^ (acc >> 15)
    acc = acc * prime_2
    acc = acc ^ (acc >> 13)
    acc = acc * prime_3
    acc = acc ^ (acc >> 16)
    return acc


def uint32ed_to_hash(uint32ed, seed):
    """Convert uint32 array to hash value."""

    def scan_body(seed, x):
        result = xxhash(x, seed)
        return result, result

    hash_value, _ = jax.lax.scan(scan_body, seed, uint32ed)
    return hash_value


def byterize_hash_func_builder(x: Xtructurable):
    """
    Build a hash function for the pytree.
    This function creates a JIT-compiled hash function that converts pytree to bytes
    and then to uint32 arrays for hashing.

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

    default_bytes = _byterize(x.default())
    bytes_len = default_bytes.shape[0]
    # Calculate padding needed to make byte length multiple of 4
    pad_len = jnp.where(bytes_len % 4 != 0, 4 - (bytes_len % 4), 0)

    if pad_len > 0:

        def _to_uint32_from_bytes(bytes):
            """Convert padded bytes to uint32 array."""
            x_padded = jnp.pad(bytes, (pad_len, 0), mode="constant", constant_values=0)
            x_reshaped = jnp.reshape(x_padded, (-1, 4))
            return jax.vmap(lambda x: jax.lax.bitcast_convert_type(x, jnp.uint32))(
                x_reshaped
            ).reshape(-1)

    else:

        def _to_uint32_from_bytes(bytes):
            """Convert bytes directly to uint32 array."""
            x_reshaped = jnp.reshape(bytes, (-1, 4))
            return jax.vmap(lambda x: jax.lax.bitcast_convert_type(x, jnp.uint32))(
                x_reshaped
            ).reshape(-1)

    def _to_uint32(x):
        """Convert pytree to uint32 array."""
        bytes = _byterize(x)
        return _to_uint32_from_bytes(bytes)

    def _h(x, seed=0):
        """
        Main hash function that converts state to bytes and applies xxhash.
        Returns both hash value and its uint32 representation.
        """
        return uint32ed_to_hash(_to_uint32(x), seed)

    def _h_with_uint32ed(x, seed=0):
        """
        Main hash function that converts state to bytes and applies xxhash.
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
