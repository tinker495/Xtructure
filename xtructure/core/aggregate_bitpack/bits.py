"""Bit-level insert/extract helpers for aggregate bitpack."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax


def _mask_u32(bits: int) -> jnp.uint32:
    return jnp.uint32(0xFFFFFFFF) if bits == 32 else jnp.uint32((1 << bits) - 1)


def _insert_bits(
    words: jax.Array, bit_pos: jax.Array, value_u32: jax.Array, bits: int
) -> jax.Array:
    """Insert `bits` LSBs of value into words at bit_pos (little-endian bit numbering)."""
    bit_pos = bit_pos.astype(jnp.uint32)
    idx = jnp.right_shift(bit_pos, jnp.uint32(5)).astype(jnp.int32)
    shift = (bit_pos & jnp.uint32(31)).astype(jnp.uint32)
    bits_u32 = jnp.uint32(bits)
    mask = _mask_u32(bits)
    v = (value_u32.astype(jnp.uint32) & mask).astype(jnp.uint32)

    def _fits(_):
        w = lax.dynamic_index_in_dim(words, idx, axis=0, keepdims=False)
        w2 = w | (v << shift)
        return words.at[idx].set(w2)

    def _spans(_):
        low_bits = jnp.uint32(32) - shift  # in [1,31]
        low_mask = (
            jnp.uint32(0xFFFFFFFF)
            if bits == 32
            else (jnp.left_shift(jnp.uint32(1), low_bits) - jnp.uint32(1))
        )
        low_part = v & low_mask
        high_part = v >> low_bits
        w0 = lax.dynamic_index_in_dim(words, idx, axis=0, keepdims=False) | (low_part << shift)
        w1 = lax.dynamic_index_in_dim(words, idx + 1, axis=0, keepdims=False) | high_part
        out = words.at[idx].set(w0)
        out = out.at[idx + 1].set(w1)
        return out

    return lax.cond(shift + bits_u32 <= jnp.uint32(32), _fits, _spans, operand=None)


def _extract_bits(words: jax.Array, bit_pos: jax.Array, bits: int) -> jax.Array:
    """Extract `bits` value from words at bit_pos (little-endian bit numbering)."""
    bit_pos = bit_pos.astype(jnp.uint32)
    idx = jnp.right_shift(bit_pos, jnp.uint32(5)).astype(jnp.int32)
    shift = (bit_pos & jnp.uint32(31)).astype(jnp.uint32)
    bits_u32 = jnp.uint32(bits)
    mask = _mask_u32(bits)

    def _fits(_):
        w = lax.dynamic_index_in_dim(words, idx, axis=0, keepdims=False)
        return (w >> shift) & mask

    def _spans(_):
        w0 = lax.dynamic_index_in_dim(words, idx, axis=0, keepdims=False)
        w1 = lax.dynamic_index_in_dim(words, idx + 1, axis=0, keepdims=False)
        low = w0 >> shift
        high = w1 << (jnp.uint32(32) - shift)
        return (low | high) & mask

    return lax.cond(shift + bits_u32 <= jnp.uint32(32), _fits, _spans, operand=None)
