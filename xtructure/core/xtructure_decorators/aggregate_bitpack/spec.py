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


@dataclasses.dataclass(frozen=True)
class _AggWordContribTables:
    """Precomputed per-word contribution tables for aggregate packing.

    The aggregate bitstream is logically a concatenation of fixed-width values.
    Packing can be expressed as computing each output word independently by
    assembling bit-ranges from one or more values.

    Arrays are plain NumPy and intended to be embedded as JAX constants when
    used inside JIT/pallas kernels.
    """

    total_values: int
    max_contrib: int
    value_idx: np.ndarray
    value_bit_start: np.ndarray
    word_bit_start: np.ndarray
    num_bits: np.ndarray
    mask_u32: np.ndarray


def _default_unpack_dtype(bits: int) -> Any:
    if bits == 1:
        return jnp.bool_
    if bits <= 8:
        return jnp.uint8
    return jnp.uint32


def _mask_u32_np(bits: int) -> np.uint32:
    if bits <= 0:
        return np.uint32(0)
    if bits >= 32:
        return np.uint32(0xFFFFFFFF)
    return np.uint32((1 << bits) - 1)


def _build_word_contrib_tables(
    specs: list[_AggLeafSpec], *, words_all_len: int
) -> _AggWordContribTables:
    """Build per-word contribution tables for packing.

    Returns tables describing how to assemble each output word from the input
    value stream.
    """
    if words_all_len < 0:
        raise ValueError("words_all_len must be non-negative")

    # Each word gets a list of (global_value_idx, value_bit_start, word_bit_start, num_bits)
    contribs: list[list[tuple[int, int, int, int]]] = [
        [] for _ in range(int(words_all_len))
    ]

    value_offset = 0
    for s in specs:
        bits = int(s.bits)
        if bits <= 0 or bits > 32:
            raise ValueError(f"bits must be 1..32, got {bits}")

        for i in range(int(s.nvalues)):
            start = int(s.bit_offset) + int(i) * bits
            end = start + bits  # exclusive

            if bits == 0:
                continue
            if words_all_len == 0:
                raise ValueError("non-empty spec requires words_all_len > 0")

            w0 = start // 32
            w1 = (end - 1) // 32
            for w in range(w0, w1 + 1):
                word_start = 32 * w
                o_start = max(start, word_start)
                o_end = min(end, word_start + 32)
                nb = o_end - o_start
                if nb <= 0:
                    continue

                vb = o_start - start
                wb = o_start - word_start
                contribs[w].append((value_offset + i, vb, wb, nb))

        value_offset += int(s.nvalues)

    total_values = int(value_offset)
    max_contrib = max((len(c) for c in contribs), default=0)

    value_idx = np.full((words_all_len, max_contrib), -1, dtype=np.int32)
    value_bit_start = np.zeros((words_all_len, max_contrib), dtype=np.uint8)
    word_bit_start = np.zeros((words_all_len, max_contrib), dtype=np.uint8)
    num_bits = np.zeros((words_all_len, max_contrib), dtype=np.uint8)
    mask_u32 = np.zeros((words_all_len, max_contrib), dtype=np.uint32)

    for w, entries in enumerate(contribs):
        for t, (vi, vb, wb, nb) in enumerate(entries):
            value_idx[w, t] = np.int32(vi)
            value_bit_start[w, t] = np.uint8(vb)
            word_bit_start[w, t] = np.uint8(wb)
            num_bits[w, t] = np.uint8(nb)
            mask_u32[w, t] = _mask_u32_np(nb)

    return _AggWordContribTables(
        total_values=total_values,
        max_contrib=max_contrib,
        value_idx=value_idx,
        value_bit_start=value_bit_start,
        word_bit_start=word_bit_start,
        num_bits=num_bits,
        mask_u32=mask_u32,
    )


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
                int(np.prod(np.array(unpacked_shape, dtype=np.int64)))
                if unpacked_shape
                else 1
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
