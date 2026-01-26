"""Bitpacking utilities for compact serialization.

This module packs arrays whose values use only a small number of active bits
into a compact uint8 stream, and unpacks them back to arrays.

Design notes:
- Supports 1..8 bits per value (inclusive).
- Uses JAX-compatible primitives (jnp/jax.vmap) so it can run on device.
"""

from __future__ import annotations

from functools import lru_cache

import chex
import jax
import jax.numpy as jnp
import numpy as np

from .kernels import pack_words_all_xla
from .spec import _AggLeafSpec, _build_word_contrib_tables


def packed_num_bytes(num_values: int, active_bits: int) -> int:
    """Return the number of uint8 bytes required to pack `num_values` values.

    This mirrors the packing strategy of :func:`to_uint8` (block-aligned for 3/5/6/7 bits).
    """
    if not isinstance(num_values, (int, np.integer)):
        raise TypeError(f"num_values must be an int, got {type(num_values).__name__}")
    if num_values < 0:
        raise ValueError(f"num_values must be non-negative, got {num_values}")
    if not isinstance(active_bits, int):
        raise TypeError(f"active_bits must be an int, got {type(active_bits).__name__}")
    if active_bits < 1 or active_bits > 32:
        raise ValueError(f"active_bits must be 1-32, got {active_bits}")

    if num_values == 0:
        return 0

    if active_bits == 8:
        return int(num_values)
    if active_bits == 1:
        return int((num_values + 7) // 8)
    if active_bits in (2, 4):
        values_per_byte = 8 // active_bits
        return int((num_values + values_per_byte - 1) // values_per_byte)

    L = int(np.lcm(active_bits, 8))
    num_values_per_block = L // active_bits
    num_bytes_per_block = L // 8
    num_blocks = int((num_values + num_values_per_block - 1) // num_values_per_block)
    return int(num_blocks * num_bytes_per_block)


@lru_cache(maxsize=None)
def _get_block_pack_tables(active_bits: int):
    """Cache per-block word contribution tables for bitpacking."""
    if not isinstance(active_bits, int):
        raise TypeError(f"active_bits must be int, got {type(active_bits).__name__}")
    if active_bits < 1 or active_bits > 32:
        raise ValueError(f"active_bits must be 1-32, got {active_bits}")

    L = int(np.lcm(active_bits, 8))
    values_per_block = int(L // active_bits)
    bytes_per_block = int(L // 8)
    words_all_len = int((L + 31) // 32) if L else 0

    spec = _AggLeafSpec(
        path=("leaf",),
        bits=int(active_bits),
        unpacked_shape=(values_per_block,),
        nvalues=values_per_block,
        bit_offset=0,
        bit_len=int(values_per_block * active_bits),
        unpack_dtype=jnp.uint32,
        declared_dtype=jnp.uint32,
    )
    tables = _build_word_contrib_tables([spec], words_all_len=words_all_len)
    return tables, values_per_block, bytes_per_block, words_all_len


def _mask_u32(bits: int) -> chex.Array:
    return jnp.uint32(0xFFFFFFFF) if bits == 32 else jnp.uint32((1 << bits) - 1)


def _insert_bits(
    words: jax.Array, bit_pos: jax.Array, value_u32: jax.Array, bits: int
) -> jax.Array:
    """Insert `bits` LSBs of value into words at bit_pos (little-endian bit numbering)."""
    # Optimized branch-free implementation
    bit_pos = bit_pos.astype(jnp.uint32)
    idx = jnp.right_shift(bit_pos, jnp.uint32(5)).astype(jnp.int32)
    shift = (bit_pos & jnp.uint32(31)).astype(jnp.uint32)

    # Mask value to ensure no upper bits pollute
    mask = _mask_u32(bits)
    v = (value_u32.astype(jnp.uint32) & mask).astype(jnp.uint32)

    # Calculate contributions to current and next word
    low_part = v << shift
    high_part = v >> (jnp.uint32(32) - shift)

    # Update current word
    w0 = words[idx] | low_part
    words = words.at[idx].set(w0)

    # Update next word safely (branch-free OOB handling)
    # We read w1 safely (fill 0 if OOB), OR in the high part, and write back.
    # If OOB, the write is dropped.
    w1 = jnp.take(words, idx + 1, mode="fill")
    w1 = w1 | high_part
    words = words.at[idx + 1].set(w1)

    return words


def _extract_bits(words: jax.Array, bit_pos: jax.Array, bits: int) -> jax.Array:
    """Extract `bits` value from words at bit_pos (little-endian bit numbering)."""
    # Optimized branch-free implementation
    bit_pos = bit_pos.astype(jnp.uint32)
    idx = jnp.right_shift(bit_pos, jnp.uint32(5)).astype(jnp.int32)
    shift = (bit_pos & jnp.uint32(31)).astype(jnp.uint32)
    mask = _mask_u32(bits)

    w0 = words[idx]
    w1 = jnp.take(words, idx + 1, mode="fill")

    low = w0 >> shift
    high = w1 << (jnp.uint32(32) - shift)

    return (low | high) & mask


def to_uint8(values: chex.Array, active_bits: int = 1) -> chex.Array:
    """Pack an array into a uint8 stream using `active_bits` per value.

    Args:
        values: Input array. For active_bits==1, can be bool or integer (0/!=0).
            For active_bits>1, must be integer.
        active_bits: Bits per value in [1, 8].

    Returns:
        A 1D uint8 array of packed bytes.
    """
    assert 1 <= active_bits <= 32, f"active_bits must be 1-32, got {active_bits}"

    if active_bits == 1:
        flatten_input = values.reshape((-1,))
        if flatten_input.dtype != jnp.bool_:
            flatten_input = flatten_input != 0
        return jnp.packbits(flatten_input, axis=-1, bitorder="little")

    assert jnp.issubdtype(
        values.dtype, jnp.integer
    ), f"values must be integer array for active_bits={active_bits}, got dtype={values.dtype}"

    values_flat = values.reshape((-1,))

    # Fast path for byte-aligned packing.
    if active_bits == 8:
        return values_flat.astype(jnp.uint8)

    if active_bits in (2, 4):
        values_per_byte = 8 // active_bits
        padding = (values_per_byte - (values_flat.size % values_per_byte)) % values_per_byte
        if padding:
            values_flat = jnp.concatenate(
                [values_flat, jnp.zeros((padding,), dtype=values_flat.dtype)]
            )
        grouped = values_flat.reshape((-1, values_per_byte))

        mask = jnp.uint8((1 << active_bits) - 1)
        shifts = (jnp.arange(values_per_byte, dtype=jnp.uint8) * jnp.uint8(active_bits)).astype(
            jnp.uint8
        )

        grouped_u8 = (grouped.astype(jnp.uint8) & mask).astype(jnp.uint8)
        parts = jnp.left_shift(grouped_u8, shifts[None, :])
        return jnp.bitwise_or.reduce(parts, axis=1).astype(jnp.uint8)

    # General path for any other bit-width (3..32 except 4,8). Use L = lcm(active_bits, 8) to align blocks.
    tables, num_values_per_block, num_bytes_per_block, _words_all_len = _get_block_pack_tables(
        active_bits
    )

    padding = (
        num_values_per_block - (values_flat.size % num_values_per_block)
    ) % num_values_per_block
    if padding:
        values_flat = jnp.concatenate([values_flat, jnp.zeros((padding,), dtype=values_flat.dtype)])

    grouped = values_flat.reshape((-1, num_values_per_block)).astype(jnp.uint32)
    words = pack_words_all_xla(grouped, tables)
    bytes_all = jax.lax.bitcast_convert_type(words, jnp.uint8).reshape((words.shape[0], -1))
    return bytes_all[:, :num_bytes_per_block].reshape((-1,)).astype(jnp.uint8)


def from_uint8(
    packed_bytes: chex.Array, target_shape: tuple[int, ...], active_bits: int = 1
) -> chex.Array:
    """Unpack a uint8 stream back into an array of shape `target_shape`.

    Notes:
    - For active_bits==1, returns bool.
    - For active_bits>1, returns uint8 values in [0, 2**active_bits - 1].
      Caller can cast to a desired integer dtype.
    """
    packed_bytes = jnp.asarray(packed_bytes, dtype=jnp.uint8).reshape((-1,))
    assert 1 <= active_bits <= 32, f"active_bits must be 1-32, got {active_bits}"

    num_target_elements = int(np.prod(target_shape))
    assert num_target_elements >= 0, "target_shape must have non-negative product"
    if num_target_elements == 0:
        # Preserve dtype semantics even for empty tensors.
        if active_bits == 1:
            return jnp.zeros(target_shape, dtype=jnp.bool_)
        return jnp.zeros(target_shape, dtype=jnp.uint8)

    if active_bits == 1:
        bits = jnp.unpackbits(packed_bytes, count=num_target_elements, bitorder="little")
        return bits.reshape(target_shape).astype(jnp.bool_)

    if active_bits == 8:
        return packed_bytes[:num_target_elements].reshape(target_shape)

    if active_bits in (2, 4):
        values_per_byte = 8 // active_bits
        mask = jnp.uint8((1 << active_bits) - 1)

        shifts = (jnp.arange(values_per_byte, dtype=jnp.uint8) * jnp.uint8(active_bits)).astype(
            jnp.uint8
        )
        vals = jnp.right_shift(packed_bytes[:, None], shifts[None, :]) & mask
        all_values = vals.reshape((-1,))
        return all_values[:num_target_elements].reshape(target_shape)

    tables, num_values_per_block, num_bytes_per_block, words_all_len = _get_block_pack_tables(
        active_bits
    )
    # Avoid Python int overflow and handle active_bits == 32.
    mask = jnp.uint32(0xFFFFFFFF) if active_bits == 32 else jnp.uint32((1 << active_bits) - 1)

    total_blocks = (packed_bytes.size + num_bytes_per_block - 1) // num_bytes_per_block
    total_bytes = total_blocks * num_bytes_per_block
    if total_bytes != packed_bytes.size:
        packed_bytes = jnp.pad(packed_bytes, (0, total_bytes - packed_bytes.size), mode="constant")

    grouped = packed_bytes.reshape((-1, num_bytes_per_block))
    pad = (-num_bytes_per_block) % 4
    if pad:
        grouped = jnp.pad(grouped, ((0, 0), (0, pad)), mode="constant", constant_values=0)

    words = jax.lax.bitcast_convert_type(
        grouped.reshape((-1, words_all_len, 4)), jnp.uint32
    ).reshape((-1, words_all_len))
    words_padded = jnp.concatenate(
        [words, jnp.zeros((words.shape[0], 1), dtype=jnp.uint32)], axis=1
    )

    bit_pos = jnp.arange(num_values_per_block, dtype=jnp.uint32) * jnp.uint32(active_bits)
    word_idx = jnp.right_shift(bit_pos, jnp.uint32(5)).astype(jnp.int32)
    shift = (bit_pos & jnp.uint32(31)).astype(jnp.uint32)

    w0 = jnp.take(words_padded, word_idx, axis=1)
    w1 = jnp.take(words_padded, word_idx + 1, axis=1)

    shift2d = shift[None, :]
    low = jnp.right_shift(w0, shift2d)
    high = jnp.where(shift2d == 0, jnp.uint32(0), jnp.left_shift(w1, jnp.uint32(32) - shift2d))
    vals = jnp.bitwise_and(jnp.bitwise_or(low, high), mask)

    out_dtype = jnp.uint8 if active_bits <= 8 else jnp.uint32
    all_values = vals.astype(out_dtype).reshape((-1,))
    return all_values[:num_target_elements].reshape(target_shape)
