import os
from typing import Any, cast

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


@jax.jit
def hash_fast_uint32ed_pair(uint32ed, seed=jnp.uint32(0)):
    """Vectorized hash reducer for uint32 streams (returns two 32-bit hashes).

    This is intended to support double-hashing and wide signatures without
    requiring a second full pass over the input.
    """
    uint32ed = jnp.asarray(uint32ed, dtype=jnp.uint32).reshape(-1)
    seed = jnp.uint32(seed)
    if uint32ed.size == 0:
        h0 = _avalanche32(seed ^ jnp.uint32(0x9E3779B1))
        h1 = _avalanche32(seed ^ jnp.uint32(0x85EBCA6B))
        return h0, h1

    idx = jnp.arange(uint32ed.shape[0], dtype=jnp.uint32)
    salt = idx * jnp.uint32(0x9E3779B1)
    lanes = _avalanche32(uint32ed ^ salt ^ seed)

    combined0 = jnp.bitwise_xor.reduce(lanes)
    combined1 = jnp.bitwise_xor.reduce(lanes ^ (salt * jnp.uint32(0x85EBCA6B)))

    length_mix = jnp.uint32(uint32ed.shape[0] << 2)
    combined0 = combined0 ^ length_mix ^ seed
    combined1 = combined1 ^ length_mix ^ (seed ^ jnp.uint32(0xC2B2AE35))
    return _avalanche32(combined0), _avalanche32(combined1)


def uint32ed_to_hash(uint32ed, seed):
    """Convert uint32 array to hash value."""
    return hash_fast_uint32ed(uint32ed, seed)


def uint32ed_to_hash_pair(uint32ed, seed):
    """Convert uint32 array to a pair of 32-bit hashes."""
    return hash_fast_uint32ed_pair(uint32ed, seed)


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

    Packed = getattr(x, "Packed", None)
    agg_tail_bytes = getattr(Packed, "__agg_tail_bytes__", None) if Packed is not None else None
    agg_words_len = getattr(Packed, "__agg_words_len__", None) if Packed is not None else None

    has_agg_layout = agg_tail_bytes is not None and agg_words_len is not None
    agg_mode = os.environ.get("XTRUCTURE_HASH_AGG_MODE", "raw").strip().lower()
    if agg_mode not in {"raw", "packed", "auto"}:
        raise ValueError("XTRUCTURE_HASH_AGG_MODE must be one of: raw, packed, auto")

    stream_mode = os.environ.get("XTRUCTURE_HASH_STREAM", "auto").strip().lower()
    if stream_mode not in {"off", "on", "auto"}:
        raise ValueError("XTRUCTURE_HASH_STREAM must be one of: off, on, auto")

    try:
        stream_threshold = int(os.environ.get("XTRUCTURE_HASH_STREAM_THRESHOLD_U32", "8192"))
    except ValueError as exc:
        raise ValueError("XTRUCTURE_HASH_STREAM_THRESHOLD_U32 must be an integer") from exc
    if stream_threshold < 0:
        raise ValueError("XTRUCTURE_HASH_STREAM_THRESHOLD_U32 must be >= 0")

    if not has_agg_layout:
        use_agg_packed = False
    elif agg_mode == "packed":
        use_agg_packed = True
    elif agg_mode == "raw":
        use_agg_packed = False
    else:
        # auto: prefer raw on GPU/CPU (usually faster), packed on TPU.
        use_agg_packed = jax.default_backend() == "tpu"

    tail_bytes = int(cast(int, agg_tail_bytes)) if use_agg_packed else 0
    stored_words_len = int(cast(int, agg_words_len)) if use_agg_packed else 0

    @jax.jit
    def _to_bytes_leaf(x):
        """Convert a single leaf to a byte array."""
        if x.dtype == jnp.bool_:
            x = x.astype(jnp.uint8)
        return jax.lax.bitcast_convert_type(x, jnp.uint8).reshape(-1)

    @jax.jit
    def _byterize_raw(x):
        """Convert entire state tree to flattened byte array."""
        x = jax.tree_util.tree_map(_to_bytes_leaf, x)
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
            return _to_uint32_from_bytes(_to_bytes_leaf(leaf))

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

        return _to_uint32_from_bytes(_to_bytes_leaf(leaf))

    def _to_uint32_raw(x):
        """Convert pytree to uint32 array."""
        uint32_leaves = jax.tree_util.tree_map(_leaf_to_uint32, x)
        flat_leaves, _ = jax.tree_util.tree_flatten(uint32_leaves)
        if len(flat_leaves) == 0:
            return jnp.zeros((0,), dtype=jnp.uint32)
        return jnp.concatenate(flat_leaves)

    def _hash_streaming_from_leaves(flat_leaves, seed):
        """Hash without materializing the full concatenated uint32ed buffer."""
        seed_u32 = jnp.uint32(seed)
        if len(flat_leaves) == 0:
            return _avalanche32(seed_u32 ^ jnp.uint32(0x9E3779B1))

        combined = jnp.uint32(0)
        const_salt = jnp.uint32(0x9E3779B1)

        offset = 0
        for leaf in flat_leaves:
            leaf_u32 = jnp.asarray(leaf, dtype=jnp.uint32).reshape(-1)
            seg_len = int(leaf_u32.shape[0])
            if seg_len == 0:
                continue
            idx = jnp.arange(seg_len, dtype=jnp.uint32) + jnp.uint32(offset)
            salt = idx * const_salt
            lanes = _avalanche32(leaf_u32 ^ salt ^ seed_u32)
            combined = combined ^ jnp.bitwise_xor.reduce(lanes)
            offset += seg_len

        combined = combined ^ jnp.uint32(offset << 2) ^ seed_u32
        return _avalanche32(combined)

    def _hash_pair_streaming_from_leaves(flat_leaves, seed):
        """Hash pair without materializing the full concatenated uint32ed buffer."""
        seed_u32 = jnp.uint32(seed)
        if len(flat_leaves) == 0:
            h0 = _avalanche32(seed_u32 ^ jnp.uint32(0x9E3779B1))
            h1 = _avalanche32(seed_u32 ^ jnp.uint32(0x85EBCA6B))
            return h0, h1

        combined0 = jnp.uint32(0)
        combined1 = jnp.uint32(0)
        const_salt = jnp.uint32(0x9E3779B1)
        const_b = jnp.uint32(0x85EBCA6B)
        const_c = jnp.uint32(0xC2B2AE35)

        offset = 0
        for leaf in flat_leaves:
            leaf_u32 = jnp.asarray(leaf, dtype=jnp.uint32).reshape(-1)
            seg_len = int(leaf_u32.shape[0])
            if seg_len == 0:
                continue
            idx = jnp.arange(seg_len, dtype=jnp.uint32) + jnp.uint32(offset)
            salt = idx * const_salt
            lanes = _avalanche32(leaf_u32 ^ salt ^ seed_u32)
            combined0 = combined0 ^ jnp.bitwise_xor.reduce(lanes)
            combined1 = combined1 ^ jnp.bitwise_xor.reduce(lanes ^ (salt * const_b))
            offset += seg_len

        length_mix = jnp.uint32(offset << 2)
        combined0 = combined0 ^ length_mix ^ seed_u32
        combined1 = combined1 ^ length_mix ^ (seed_u32 ^ const_c)
        return _avalanche32(combined0), _avalanche32(combined1)

    if use_agg_packed:

        def _words_all_from_packed_instance(packed):
            words = jnp.asarray(packed.words, dtype=jnp.uint32).reshape((-1,))
            if tail_bytes == 0:
                return words
            tail = jnp.asarray(packed.tail, dtype=jnp.uint8).reshape((-1,))
            last = jnp.uint32(0)
            for i in range(tail_bytes):
                last = last | (tail[i].astype(jnp.uint32) << jnp.uint32(8 * i))
            if stored_words_len:
                return jnp.concatenate([words, last[None]], axis=0)
            return last[None]

        @jax.jit
        def _byterize_fn(x):
            """Byte representation based on aggregate-packed storage."""
            packed = x.packed
            words = jnp.asarray(packed.words, dtype=jnp.uint32).reshape((-1,))
            words_bytes = jax.lax.bitcast_convert_type(words, jnp.uint8).reshape((-1,))
            if tail_bytes == 0:
                return words_bytes
            tail = jnp.asarray(packed.tail, dtype=jnp.uint8).reshape((-1,))
            if words_bytes.size == 0:
                return tail
            return jnp.concatenate([words_bytes, tail], axis=0)

        @jax.jit
        def _to_uint32_fn(x):
            """Uint32 representation based on aggregate-packed words_all."""
            return _words_all_from_packed_instance(x.packed)

    else:
        _byterize_fn = _byterize_raw
        _to_uint32_fn = cast(Any, jax.jit(_to_uint32_raw))

    def _h(x, seed=0):
        """Main hash function that converts state to uint32 lanes and hashes them."""
        if use_agg_packed or stream_mode == "off":
            return uint32ed_to_hash(_to_uint32_fn(x), seed)

        uint32_leaves = jax.tree_util.tree_map(_leaf_to_uint32, x)
        flat_leaves, _ = jax.tree_util.tree_flatten(uint32_leaves)
        total_len = 0
        for leaf in flat_leaves:
            total_len += int(jnp.asarray(leaf).shape[0])

        if stream_mode == "auto" and total_len <= stream_threshold:
            if len(flat_leaves) == 0:
                return uint32ed_to_hash(jnp.zeros((0,), dtype=jnp.uint32), seed)
            return uint32ed_to_hash(jnp.concatenate(flat_leaves), seed)

        return _hash_streaming_from_leaves(flat_leaves, seed)

    def _h_pair(x, seed=0):
        """Hash function that returns two 32-bit hashes."""
        if use_agg_packed or stream_mode == "off":
            return uint32ed_to_hash_pair(_to_uint32_fn(x), seed)

        uint32_leaves = jax.tree_util.tree_map(_leaf_to_uint32, x)
        flat_leaves, _ = jax.tree_util.tree_flatten(uint32_leaves)
        total_len = 0
        for leaf in flat_leaves:
            total_len += int(jnp.asarray(leaf).shape[0])

        if stream_mode == "auto" and total_len <= stream_threshold:
            if len(flat_leaves) == 0:
                return uint32ed_to_hash_pair(jnp.zeros((0,), dtype=jnp.uint32), seed)
            return uint32ed_to_hash_pair(jnp.concatenate(flat_leaves), seed)

        return _hash_pair_streaming_from_leaves(flat_leaves, seed)

    def _h_pair_with_uint32ed(x, seed=0):
        """Hash function that returns two 32-bit hashes and the uint32 lanes."""
        uint32ed = _to_uint32_fn(x)
        return uint32ed_to_hash_pair(uint32ed, seed), uint32ed

    def _h_with_uint32ed(x, seed=0):
        """
        Main hash function that converts state to uint32 lanes and hashes them.
        Returns both hash value and its uint32 representation.
        """
        uint32ed = _to_uint32_fn(x)
        return uint32ed_to_hash(uint32ed, seed), uint32ed

    return (
        _byterize_fn,
        _to_uint32_fn,
        jax.jit(_h),
        jax.jit(_h_with_uint32ed),
        jax.jit(_h_pair),
        jax.jit(_h_pair_with_uint32ed),
    )


def hash_function_decorator(cls):
    """
    Decorator to add a hash function to a class.
    """
    (
        _byterize,
        _to_uint32,
        _h,
        _h_with_uint32ed,
        _h_pair,
        _h_pair_with_uint32ed,
    ) = byterize_hash_func_builder(cls)

    setattr(cls, "bytes", property(_byterize))
    setattr(cls, "uint32ed", property(_to_uint32))
    setattr(cls, "hash", _h)
    setattr(cls, "hash_with_uint32ed", _h_with_uint32ed)
    setattr(cls, "hash_pair", _h_pair)
    setattr(cls, "hash_pair_with_uint32ed", _h_pair_with_uint32ed)

    return cls
