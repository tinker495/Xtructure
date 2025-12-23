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

    if L <= 32:

        def pack_block(group):
            acc = jnp.uint32(0)
            for i in range(num_values_per_block):
                acc = acc | (group[i].astype(jnp.uint32) << jnp.uint32(i * active_bits))
            return jnp.array(
                [(acc >> jnp.uint32(8 * j)) & jnp.uint32(0xFF) for j in range(num_bytes_per_block)],
                dtype=jnp.uint8,
            )

        packed = jax.vmap(pack_block)(grouped)
        return packed.reshape((-1,))

    # Stream bytes out of a uint32 accumulator for larger blocks (e.g. bits=5,7,9,15,...).
    def pack_block(group):
        packed_bytes = jnp.zeros((num_bytes_per_block,), dtype=jnp.uint8)
        acc = jnp.uint32(0)
        bits_in_acc = 0
        byte_idx = 0
        for i in range(num_values_per_block):
            acc = acc | (group[i].astype(jnp.uint32) << jnp.uint32(bits_in_acc))
            bits_in_acc += active_bits
            while bits_in_acc >= 8:
                packed_bytes = packed_bytes.at[byte_idx].set(
                    (acc & jnp.uint32(0xFF)).astype(jnp.uint8)
                )
                acc = acc >> jnp.uint32(8)
                bits_in_acc -= 8
                byte_idx += 1
        if byte_idx < num_bytes_per_block:
            packed_bytes = packed_bytes.at[byte_idx].set((acc & jnp.uint32(0xFF)).astype(jnp.uint8))
        return packed_bytes

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

    if L <= 32:

        def unpack_block(byte_group):
            acc = jnp.uint32(0)
            for j in range(num_bytes_per_block):
                acc = acc | (byte_group[j].astype(jnp.uint32) << jnp.uint32(8 * j))
            vals = [
                (acc >> jnp.uint32(i * active_bits)) & mask for i in range(num_values_per_block)
            ]
            dtype_out = jnp.uint8 if active_bits <= 8 else jnp.uint32
            return jnp.array(vals, dtype=dtype_out)

        blocks = jax.vmap(unpack_block)(grouped)
        all_values = blocks.reshape((-1,))
        return all_values[:num_target_elements].reshape(target_shape)

    def unpack_block(byte_group):
        # For bits > 8, values won't fit in uint8. We'll emit uint32 and let caller cast if desired.
        out_dtype = jnp.uint8 if active_bits <= 8 else jnp.uint32
        vals = jnp.zeros((num_values_per_block,), dtype=out_dtype)
        acc = jnp.uint32(0)
        bits_in_acc = 0
        byte_idx = 0
        for i in range(num_values_per_block):
            while bits_in_acc < active_bits and byte_idx < num_bytes_per_block:
                acc = acc | (byte_group[byte_idx].astype(jnp.uint32) << jnp.uint32(bits_in_acc))
                bits_in_acc += 8
                byte_idx += 1
            vals = vals.at[i].set((acc & mask).astype(out_dtype))
            acc = acc >> jnp.uint32(active_bits)
            bits_in_acc -= active_bits
        return vals

    blocks = jax.vmap(unpack_block)(grouped)
    all_values = blocks.reshape((-1,))
    return all_values[:num_target_elements].reshape(target_shape)
