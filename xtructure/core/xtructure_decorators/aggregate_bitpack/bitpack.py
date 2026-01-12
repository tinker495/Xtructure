"""Bitpacking utilities for compact serialization.

This module packs arrays whose values use only a small number of active bits
into a compact uint8 stream, and unpacks them back to arrays.

Design notes:
- Supports 1..8 bits per value (inclusive).
- Uses JAX-compatible primitives (jnp/jax.vmap) so it can run on device.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np


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


def _mask_u32(bits: int) -> jnp.uint32:
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

        def pack_group(group):
            out = jnp.uint8(0)
            for i in range(values_per_byte):
                out = out | (group[i].astype(jnp.uint8) << jnp.uint8(i * active_bits))
            return out

        return jax.vmap(pack_group)(grouped)

    # General path for any other bit-width (3..32 except 4,8). Use L = lcm(active_bits, 8) to align blocks.
    L = int(np.lcm(active_bits, 8))  # total bits per block
    num_values_per_block = L // active_bits
    num_bytes_per_block = L // 8

    padding = (
        num_values_per_block - (values_flat.size % num_values_per_block)
    ) % num_values_per_block
    if padding:
        values_flat = jnp.concatenate([values_flat, jnp.zeros((padding,), dtype=values_flat.dtype)])
    grouped = values_flat.reshape((-1, num_values_per_block))

    # General path for any other bit-width: pack into uint32 buffers and bitcast
    def pack_block(group):
        bits_needed = num_values_per_block * active_bits
        words_needed = (bits_needed + 31) // 32

        scratch = jnp.zeros((words_needed,), dtype=jnp.uint32)

        for i in range(num_values_per_block):
            val = group[i]
            # Must cast to uint32 for shift operations inside _insert_bits
            scratch = _insert_bits(
                scratch, jnp.uint32(i * active_bits), val.astype(jnp.uint32), active_bits
            )

        out_bytes = jax.lax.bitcast_convert_type(scratch, jnp.uint8)
        return out_bytes.reshape(-1)[:num_bytes_per_block]

    packed = jax.vmap(pack_block)(grouped)
    return packed.reshape((-1,))


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

        def unpack_byte(b):
            vals = []
            for i in range(values_per_byte):
                vals.append((b >> jnp.uint8(i * active_bits)) & mask)
            return jnp.array(vals, dtype=jnp.uint8)

        groups = jax.vmap(unpack_byte)(packed_bytes)
        all_values = groups.reshape((-1,))
        return all_values[:num_target_elements].reshape(target_shape)

    L = int(np.lcm(active_bits, 8))
    num_values_per_block = L // active_bits
    num_bytes_per_block = L // 8
    # Avoid Python int overflow and handle active_bits == 32.
    mask = jnp.uint32(0xFFFFFFFF) if active_bits == 32 else jnp.uint32((1 << active_bits) - 1)

    total_blocks = (packed_bytes.size + num_bytes_per_block - 1) // num_bytes_per_block
    total_bytes = total_blocks * num_bytes_per_block
    if total_bytes != packed_bytes.size:
        packed_bytes = jnp.pad(packed_bytes, (0, total_bytes - packed_bytes.size), mode="constant")
    grouped = packed_bytes.reshape((-1, num_bytes_per_block))

    def unpack_block(byte_group):
        # Cast bytes to uint32 words to use optimized _extract_bits
        # Pad to multiple of 4 bytes first
        pad = (-num_bytes_per_block) % 4
        if pad:
            byte_group = jnp.pad(byte_group, (0, pad), constant_values=0)

        words = jax.lax.bitcast_convert_type(byte_group.reshape(-1, 4), jnp.uint32).reshape(-1)

        # Extract values
        # Since we are inside a block, indices are small, no overflow risk
        idxs = jnp.arange(num_values_per_block, dtype=jnp.uint32)

        def extract_one(i):
            return _extract_bits(words, i * jnp.uint32(active_bits), active_bits)

        vals = jax.vmap(extract_one)(idxs)

        out_dtype = jnp.uint8 if active_bits <= 8 else jnp.uint32
        return vals.astype(out_dtype)

    blocks = jax.vmap(unpack_block)(grouped)
    all_values = blocks.reshape((-1,))
    return all_values[:num_target_elements].reshape(target_shape)
