"""Aggregate bitpack layout and specification helpers."""

from __future__ import annotations

import dataclasses
from typing import Any, Type

import jax.numpy as jnp
import numpy as np

from xtructure.core.field_descriptors import get_field_descriptors
from xtructure.core.type_utils import is_xtructure_dataclass_type


@dataclasses.dataclass(frozen=True)
class _AggLeafSpec:
    path: tuple[str, ...]
    bits: int
    unpacked_shape: tuple[int, ...]
    # Number of values in this field (product of unpacked_shape)
    nvalues: int
    # Bit offset into the concatenated bitstream
    bit_offset: int
    # Total bits for this field
    bit_len: int
    # Default unpack dtype (uint8 for <=8, uint32 for >8, bool for 1)
    unpack_dtype: Any
    # Declared dtype on the original dataclass field (for reconstruction)
    declared_dtype: Any


def _default_unpack_dtype(bits: int) -> Any:
    if bits == 1:
        return jnp.bool_
    if bits <= 8:
        return jnp.uint8
    return jnp.uint32


def _build_agg_spec(root_cls: Type[Any]) -> tuple[list[_AggLeafSpec], int]:
    """Build a flat list of leaf specs (including nested leaves) in deterministic order."""
    specs: list[_AggLeafSpec] = []

    def _walk(cls: Type[Any], prefix: tuple[str, ...], bit_offset: int) -> int:
        descriptors = get_field_descriptors(cls)

        for field in dataclasses.fields(cls):
            name = field.name
            descriptor = descriptors.get(name)
            if descriptor is None:
                continue

            if is_xtructure_dataclass_type(descriptor.dtype):
                # Only support scalar nested fields for now.
                if tuple(descriptor.intrinsic_shape) not in ((),):
                    raise NotImplementedError(
                        f"aggregate_bitpack currently supports only scalar nested fields. "
                        f"Got intrinsic_shape={descriptor.intrinsic_shape} on {cls.__name__}.{name}."
                    )
                nested_cls = descriptor.dtype
                bit_offset = _walk(nested_cls, prefix + (name,), bit_offset)
                continue

            if descriptor.bits is None:
                raise ValueError(
                    f"aggregate_bitpack requires FieldDescriptor.bits on every primitive leaf. "
                    f"Missing on {cls.__name__}.{name}."
                )

            bits = int(descriptor.bits)
            unpacked_shape = tuple(descriptor.intrinsic_shape)
            nvalues = (
                int(np.prod(np.array(unpacked_shape, dtype=np.int64))) if unpacked_shape else 1
            )
            bit_len = int(nvalues * bits)

            specs.append(
                _AggLeafSpec(
                    path=prefix + (name,),
                    bits=bits,
                    unpacked_shape=unpacked_shape,
                    nvalues=nvalues,
                    bit_offset=bit_offset,
                    bit_len=bit_len,
                    unpack_dtype=_default_unpack_dtype(bits),
                    declared_dtype=descriptor.dtype,
                )
            )
            bit_offset += bit_len

        return bit_offset

    total_bits = int(_walk(root_cls, (), 0))
    return specs, total_bits


def _ceil_div(a: int, b: int) -> int:
    return int((a + b - 1) // b)


def _compute_word_tail_layout(total_bits: int) -> tuple[int, int, int]:
    """Return (words_all_len, stored_words_len, tail_bytes).

    We always pack into `words_all_len = ceil(total_bits/32)` words internally.
    Then we store either:
    - all words (tail_bytes=0), or
    - all but the last word + a 1..2 byte tail (tail_bytes in {1,2})

    Heuristic: if the remainder would require 3 bytes, store the extra word instead.
    """
    if total_bits < 0:
        raise ValueError("total_bits must be non-negative")
    words_all_len = _ceil_div(total_bits, 32) if total_bits else 0
    rem_bits = total_bits % 32
    rem_bytes = _ceil_div(rem_bits, 8) if rem_bits else 0
    if rem_bytes in (1, 2):
        tail_bytes = rem_bytes
        stored_words_len = max(words_all_len - 1, 0)
    else:
        tail_bytes = 0
        stored_words_len = words_all_len
    return words_all_len, stored_words_len, tail_bytes
