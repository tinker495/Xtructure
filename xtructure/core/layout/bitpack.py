"""Bitpack layout helpers."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import jax.numpy as jnp
import numpy as np

from .types import (
    AggregateBitpackLayout,
    AggregateBitpackReason,
    AggregateLeafLayout,
    AggregateViewFieldLayout,
    FieldLayout,
)


def default_unpack_dtype(bits: int) -> Any:
    if bits == 1:
        return jnp.bool_
    if bits <= 8:
        return jnp.uint8
    return jnp.uint32


def ceil_div(a: int, b: int) -> int:
    return int((a + b - 1) // b)


def compute_word_tail_layout(total_bits: int) -> tuple[int, int, int]:
    """Return (words_all_len, stored_words_len, tail_bytes)."""
    if total_bits < 0:
        raise ValueError("total_bits must be non-negative")
    words_all_len = ceil_div(total_bits, 32) if total_bits else 0
    rem_bits = total_bits % 32
    rem_bytes = ceil_div(rem_bits, 8) if rem_bits else 0
    if rem_bytes in (1, 2):
        tail_bytes = rem_bytes
        stored_words_len = max(words_all_len - 1, 0)
    else:
        tail_bytes = 0
        stored_words_len = words_all_len
    return words_all_len, stored_words_len, tail_bytes


def build_aggregate_bitpack_layout(
    root_cls: type, fields: tuple[FieldLayout, ...]
) -> AggregateBitpackLayout:
    """Compute current aggregate bitpack eligibility without changing constraints."""
    leaves: list[AggregateLeafLayout] = []
    view_fields_by_owner: dict[type, list[AggregateViewFieldLayout]] = {}
    bit_offset = 0

    def fail(reason: str, reason_kind: AggregateBitpackReason) -> AggregateBitpackLayout:
        return AggregateBitpackLayout(
            eligible=False,
            reason=reason,
            reason_kind=reason_kind,
            leaves=(),
            total_bits=0,
            words_all_len=0,
            stored_words_len=0,
            tail_bytes=0,
            view_fields_by_owner=MappingProxyType({}),
        )

    def walk(
        owner_type: type,
        cls_fields: tuple[FieldLayout, ...],
        prefix: tuple[str, ...] = (),
    ) -> tuple[str, AggregateBitpackReason] | None:
        nonlocal bit_offset
        owner_view_fields = view_fields_by_owner.setdefault(owner_type, [])
        for field in cls_fields:
            path = prefix + (field.name,)
            if field.is_nested:
                if field.intrinsic_shape != ():
                    return (
                        "aggregate_bitpack currently supports only scalar nested fields. "
                        f"Got intrinsic_shape={field.intrinsic_shape} on {'.'.join(path)}.",
                        AggregateBitpackReason.SCALAR_NESTED,
                    )
                owner_view_fields.append(
                    AggregateViewFieldLayout(
                        owner_type=owner_type,
                        name=field.name,
                        path=path,
                        is_nested=True,
                        nested_type=field.nested_type,
                        unpack_dtype=None,
                        unpacked_shape=(),
                    )
                )
                from .type_layout import get_type_layout

                nested_layout = get_type_layout(field.nested_type)  # type: ignore[arg-type]
                nested_reason = walk(nested_layout.cls, nested_layout.fields, path)
                if nested_reason is not None:
                    return nested_reason
                continue

            if field.bits is None:
                return (
                    "aggregate_bitpack requires FieldDescriptor.bits on every primitive leaf. "
                    f"Missing on {'.'.join(path)}.",
                    AggregateBitpackReason.MISSING_BITS,
                )

            bits = int(field.bits)
            unpacked_shape = tuple(field.intrinsic_shape)
            nvalues = (
                int(np.prod(np.array(unpacked_shape, dtype=np.int64))) if unpacked_shape else 1
            )
            bit_len = int(nvalues * bits)
            unpack_dtype = default_unpack_dtype(bits)
            leaves.append(
                AggregateLeafLayout(
                    path=path,
                    bits=bits,
                    unpacked_shape=unpacked_shape,
                    nvalues=nvalues,
                    bit_offset=bit_offset,
                    bit_len=bit_len,
                    unpack_dtype=unpack_dtype,
                    declared_dtype=field.dtype,
                )
            )
            owner_view_fields.append(
                AggregateViewFieldLayout(
                    owner_type=owner_type,
                    name=field.name,
                    path=path,
                    is_nested=False,
                    nested_type=None,
                    unpack_dtype=unpack_dtype,
                    unpacked_shape=unpacked_shape,
                )
            )
            bit_offset += bit_len
        return None

    walk_result = walk(root_cls, fields)
    if walk_result is not None:
        return fail(*walk_result)
    if not leaves:
        return fail(
            "aggregate_bitpack requires at least one primitive leaf.",
            AggregateBitpackReason.NO_LEAVES,
        )

    total_bits = int(bit_offset)
    words_all_len, stored_words_len, tail_bytes = compute_word_tail_layout(total_bits)
    return AggregateBitpackLayout(
        eligible=True,
        reason=None,
        reason_kind=None,
        leaves=tuple(leaves),
        total_bits=total_bits,
        words_all_len=words_all_len,
        stored_words_len=stored_words_len,
        tail_bytes=tail_bytes,
        view_fields_by_owner=MappingProxyType(
            {owner: tuple(owner_fields) for owner, owner_fields in view_fields_by_owner.items()}
        ),
    )
