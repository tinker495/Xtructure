"""Direct fact tests for the **Internal Layout Adapter Interface**.

CONTEXT.md authorizes first-party tests to import the Internal Layout
Adapter Interface (``layout.traversal``, ``layout.bitpack``, and the
fact dataclasses on ``layout.types``). This file uses that affordance to
make the previously implicit facts of ``AggregateLeafLayout``,
``AggregateBitpackLayout`` storage shape, ``AggregateViewFieldLayout``,
and the ``default_unpack_dtype`` policy directly observable. A regression
in any of these shows up as a named fact assertion failure rather than a
roundtrip mismatch in IO or aggregate bitpack.

Coverage map (``CONTEXT.md`` deepening completion checklist):

- G1 — ``AggregateLeafLayout`` per-leaf facts (bit_offset, bit_len, nvalues,
  unpacked_shape, unpack_dtype, declared_dtype).
- G2 — ``AggregateBitpackLayout`` storage facts (words_all_len,
  stored_words_len, tail_bytes) sourced from ``compute_word_tail_layout``.
- G3 — ``AggregateViewFieldLayout`` per-field facts (is_nested, nested_type,
  unpack_dtype, unpacked_shape) for both flat and scalar-nested cases.
- G4 — ``default_unpack_dtype`` policy across the bit-width edge points
  (``CONTEXT.md`` Stage 3 plan: 1-bit, 2-8-bit, >8-bit).

Note: this file deliberately does NOT use ``from __future__ import annotations``.
The ``@xtructure_dataclass`` decorator reads class annotations at decoration
time to extract ``FieldDescriptor`` metadata; PEP 563 stringified annotations
break that flow.
"""

import jax.numpy as jnp
import pytest

from xtructure import FieldDescriptor, xtructure_dataclass
from xtructure.core.layout import get_type_layout
from xtructure.core.layout.bitpack import compute_word_tail_layout, default_unpack_dtype
from xtructure.core.layout.types import AggregateLeafLayout, AggregateViewFieldLayout

# ----- G4: default_unpack_dtype policy ------------------------------------


@pytest.mark.parametrize(
    "bits,expected_dtype",
    [
        (1, jnp.bool_),
        (2, jnp.uint8),
        (7, jnp.uint8),
        (8, jnp.uint8),
        (9, jnp.uint32),
        (12, jnp.uint32),
        (16, jnp.uint32),
        (24, jnp.uint32),
        (32, jnp.uint32),
    ],
)
def test_default_unpack_dtype_policy(bits, expected_dtype):
    """1-bit -> bool, 2-8 -> uint8, >8 -> uint32."""
    assert default_unpack_dtype(bits) == expected_dtype


# ----- G2: compute_word_tail_layout formula -------------------------------


@pytest.mark.parametrize(
    "total_bits,expected",
    [
        # (total_bits, (words_all_len, stored_words_len, tail_bytes))
        # tail kicks in only when remaining bytes are 1 or 2.
        (0, (0, 0, 0)),
        (1, (1, 0, 1)),
        (8, (1, 0, 1)),
        (9, (1, 0, 2)),
        (16, (1, 0, 2)),
        (17, (1, 1, 0)),  # rem_bytes=3 -> falls back to whole-word storage
        (24, (1, 1, 0)),
        (32, (1, 1, 0)),  # word boundary
        (40, (2, 1, 1)),
        (48, (2, 1, 2)),
        (49, (2, 2, 0)),
        (64, (2, 2, 0)),
        (156, (5, 5, 0)),  # BitWidthSweep total
    ],
)
def test_compute_word_tail_layout_formula(total_bits, expected):
    assert compute_word_tail_layout(total_bits) == expected


def test_compute_word_tail_layout_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        compute_word_tail_layout(-1)


# ----- Bit-width sweep fixture for G1 / G3 --------------------------------


@xtructure_dataclass(aggregate_bitpack=True)
class BitWidthSweep:
    bits1: FieldDescriptor.tensor(dtype=jnp.bool_, shape=(2,), bits=1, fill_value=False)
    bits8: FieldDescriptor.tensor(dtype=jnp.uint8, shape=(2,), bits=8, fill_value=0)
    bits9: FieldDescriptor.tensor(dtype=jnp.uint16, shape=(2,), bits=9, fill_value=0)
    bits12: FieldDescriptor.tensor(dtype=jnp.uint16, shape=(2,), bits=12, fill_value=0)
    bits16: FieldDescriptor.tensor(dtype=jnp.uint16, shape=(2,), bits=16, fill_value=0)
    bits32: FieldDescriptor.tensor(dtype=jnp.uint32, shape=(2,), bits=32, fill_value=0)


_SWEEP_ORDER = ("bits1", "bits8", "bits9", "bits12", "bits16", "bits32")
_SWEEP_BITS = {"bits1": 1, "bits8": 8, "bits9": 9, "bits12": 12, "bits16": 16, "bits32": 32}
_SWEEP_DECLARED = {
    "bits1": jnp.bool_,
    "bits8": jnp.uint8,
    "bits9": jnp.uint16,
    "bits12": jnp.uint16,
    "bits16": jnp.uint16,
    "bits32": jnp.uint32,
}
_SWEEP_UNPACK = {
    "bits1": jnp.bool_,
    "bits8": jnp.uint8,
    "bits9": jnp.uint32,
    "bits12": jnp.uint32,
    "bits16": jnp.uint32,
    "bits32": jnp.uint32,
}
_SWEEP_TOTAL_BITS = sum(_SWEEP_BITS.values()) * 2  # shape=(2,) per field => 156


# ----- G1: AggregateLeafLayout per-leaf facts -----------------------------


def test_aggregate_leaf_layout_order_matches_declared_field_order():
    layout = get_type_layout(BitWidthSweep)
    assert tuple(leaf.path[0] for leaf in layout.aggregate_bitpack.leaves) == _SWEEP_ORDER


def test_aggregate_leaf_layout_facts_per_bit_width():
    layout = get_type_layout(BitWidthSweep)
    leaves = layout.aggregate_bitpack.leaves
    assert len(leaves) == len(_SWEEP_ORDER)

    bit_offset = 0
    for leaf, name in zip(leaves, _SWEEP_ORDER):
        bits = _SWEEP_BITS[name]
        assert isinstance(leaf, AggregateLeafLayout)
        assert leaf.path == (name,)
        assert leaf.bits == bits
        assert leaf.unpacked_shape == (2,)
        assert leaf.nvalues == 2
        assert leaf.bit_offset == bit_offset
        assert leaf.bit_len == bits * 2
        assert leaf.unpack_dtype == _SWEEP_UNPACK[name]
        assert jnp.dtype(leaf.declared_dtype) == jnp.dtype(_SWEEP_DECLARED[name])
        bit_offset += bits * 2

    assert bit_offset == _SWEEP_TOTAL_BITS


# ----- G2: aggregate storage facts at the layout level --------------------


def test_aggregate_storage_facts_match_compute_word_tail_layout():
    layout = get_type_layout(BitWidthSweep)
    aggregate = layout.aggregate_bitpack

    expected = compute_word_tail_layout(_SWEEP_TOTAL_BITS)
    assert aggregate.total_bits == _SWEEP_TOTAL_BITS
    assert (
        aggregate.words_all_len,
        aggregate.stored_words_len,
        aggregate.tail_bytes,
    ) == expected


# ----- G3: AggregateViewFieldLayout per-field facts (flat) ----------------


def test_aggregate_view_field_layout_facts_flat():
    layout = get_type_layout(BitWidthSweep)
    view_fields = layout.aggregate_bitpack.view_fields_by_owner[BitWidthSweep]
    by_name = {field.name: field for field in view_fields}

    assert tuple(field.name for field in view_fields) == _SWEEP_ORDER

    for name in _SWEEP_ORDER:
        field = by_name[name]
        assert isinstance(field, AggregateViewFieldLayout)
        assert field.owner_type is BitWidthSweep
        assert field.name == name
        assert field.path == (name,)
        assert field.is_nested is False
        assert field.nested_type is None
        assert field.unpack_dtype == _SWEEP_UNPACK[name]
        assert field.unpacked_shape == (2,)


# ----- G3: AggregateViewFieldLayout for scalar-nested aggregate -----------


@xtructure_dataclass(aggregate_bitpack=True)
class _NestedInner:
    flag: FieldDescriptor.tensor(dtype=jnp.bool_, shape=(3,), bits=1, fill_value=False)
    code: FieldDescriptor.tensor(dtype=jnp.uint16, shape=(2,), bits=12, fill_value=0)


@xtructure_dataclass(aggregate_bitpack=True)
class _NestedOuter:
    inner: FieldDescriptor.scalar(dtype=_NestedInner)
    tag: FieldDescriptor.tensor(dtype=jnp.uint8, shape=(1,), bits=8, fill_value=0)


def test_aggregate_view_field_layout_scalar_nested_marks_nested_field():
    layout = get_type_layout(_NestedOuter)
    outer_view_fields = layout.aggregate_bitpack.view_fields_by_owner[_NestedOuter]
    by_name = {field.name: field for field in outer_view_fields}

    nested = by_name["inner"]
    assert nested.is_nested is True
    assert nested.nested_type is _NestedInner
    assert nested.unpack_dtype is None
    assert nested.unpacked_shape == ()

    leaf = by_name["tag"]
    assert leaf.is_nested is False
    assert leaf.nested_type is None
    assert leaf.unpack_dtype == jnp.uint8
    assert leaf.unpacked_shape == (1,)


def test_aggregate_view_field_layout_scalar_nested_inner_owner_present():
    """Nested owner has its own view-field tuple keyed by the inner type."""
    layout = get_type_layout(_NestedOuter)
    inner_view_fields = layout.aggregate_bitpack.view_fields_by_owner[_NestedInner]
    by_name = {field.name: field for field in inner_view_fields}

    flag = by_name["flag"]
    assert flag.owner_type is _NestedInner
    assert flag.is_nested is False
    assert flag.unpack_dtype == jnp.bool_
    assert flag.unpacked_shape == (3,)

    code = by_name["code"]
    assert code.owner_type is _NestedInner
    assert code.is_nested is False
    assert code.unpack_dtype == jnp.uint32  # 12 bits -> uint32 per default policy
    assert code.unpacked_shape == (2,)
