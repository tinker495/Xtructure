"""Dependency-light Bitpack Layout policy helpers.

This internal file keeps Type Layout's Packed Field Layout construction
acyclic. FieldDescriptor must not import it or treat it as Schema-local policy;
callers consume the interpreted facts through Type Layout / Bitpack Layout.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np


def default_unpack_dtype(bits: int) -> Any:
    if bits == 1:
        return jnp.bool_
    if bits <= 8:
        return jnp.uint8
    return jnp.uint32


def packed_num_bytes(num_values: int, active_bits: int) -> int:
    """Return the number of uint8 bytes required to pack ``num_values`` values."""
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

    block_bits = int(np.lcm(active_bits, 8))
    values_per_block = block_bits // active_bits
    bytes_per_block = block_bits // 8
    blocks = int((num_values + values_per_block - 1) // values_per_block)
    return int(blocks * bytes_per_block)
