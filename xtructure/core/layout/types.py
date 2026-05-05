"""Layout fact types for xtructure dataclasses.

The layout package is the single source of truth for interpreting
FieldDescriptor metadata into type-level and instance-level facts.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, Literal, Mapping

from xtructure.core.structuredtype import StructuredType

AdapterFieldKind = Literal["primitive", "nested"]
AdapterRandomKind = Literal[
    "bool",
    "bits_int",
    "float",
    "nested",
    "other",
]


class AggregateBitpackReason(enum.Enum):
    """Why an aggregate bitpack layout is ineligible."""

    SCALAR_NESTED = "scalar_nested"
    MISSING_BITS = "missing_bits"
    NO_LEAVES = "no_leaves"


@dataclasses.dataclass(frozen=True)
class FieldLayout:
    """Top-level field facts for a dataclass type."""

    name: str
    path: tuple[str, ...]
    dtype: Any
    intrinsic_shape: tuple[int, ...]
    is_nested: bool
    nested_type: type | None
    bits: int | None
    packed_bits: int | None
    unpacked_dtype: Any | None
    unpacked_intrinsic_shape: tuple[int, ...] | None
    fill_value: Any
    fill_value_factory: Any
    validator: Any

    @property
    def is_packed(self) -> bool:
        return (
            self.packed_bits is not None
            and self.unpacked_intrinsic_shape is not None
            and self.unpacked_dtype is not None
        )


@dataclasses.dataclass(frozen=True)
class AdapterFieldPlan:
    """Shared per-field facts consumed by first-party Layout Adapters."""

    name: str
    path: tuple[str, ...]
    dotted_path: str
    field_kind: AdapterFieldKind
    declared_dtype: Any | None
    nested_type: type | None
    intrinsic_shape: tuple[int, ...]
    fill_value: Any
    fill_value_factory: Any
    validator: Any
    random_kind: AdapterRandomKind
    random_bits_dtype: Any | None = None
    random_view_as_signed: bool = False
    random_gen_dtype: Any | None = None
    is_primitive_jax_dtype: bool = False


@dataclasses.dataclass(frozen=True)
class LeafLayout:
    """Primitive leaf facts, flattened through nested xtructure fields."""

    path: tuple[str, ...]
    declared_dtype: Any
    intrinsic_shape: tuple[int, ...]
    local_intrinsic_shape: tuple[int, ...]
    parent_intrinsic_shape: tuple[int, ...]
    bits: int | None
    packed_bits: int | None
    unpacked_dtype: Any | None
    unpacked_intrinsic_shape: tuple[int, ...] | None
    io_pack_bits: int | None

    @property
    def name(self) -> str:
        return self.path[-1]

    @property
    def dotted_path(self) -> str:
        return ".".join(self.path)


@dataclasses.dataclass(frozen=True)
class AggregateViewFieldLayout:
    """Logical unpacked-view field facts for aggregate bitpack."""

    owner_type: type
    name: str
    path: tuple[str, ...]
    is_nested: bool
    nested_type: type | None
    unpack_dtype: Any | None
    unpacked_shape: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class AggregateLeafLayout:
    """Aggregate bitpack leaf facts for eligible primitive leaves."""

    path: tuple[str, ...]
    bits: int
    unpacked_shape: tuple[int, ...]
    nvalues: int
    bit_offset: int
    bit_len: int
    unpack_dtype: Any
    declared_dtype: Any


@dataclasses.dataclass(frozen=True)
class AggregateBitpackLayout:
    """Aggregate bitpack eligibility and storage layout facts."""

    eligible: bool
    reason: str | None
    reason_kind: AggregateBitpackReason | None
    leaves: tuple[AggregateLeafLayout, ...]
    total_bits: int
    words_all_len: int
    stored_words_len: int
    tail_bytes: int
    view_fields_by_owner: Mapping[type, tuple[AggregateViewFieldLayout, ...]]


@dataclasses.dataclass(frozen=True)
class PackedFieldLayout:
    """Field-level in-memory packed storage facts."""

    name: str
    path: tuple[str, ...]
    storage_dtype: Any
    storage_intrinsic_shape: tuple[int, ...]
    packed_bits: int
    unpacked_dtype: Any
    unpacked_intrinsic_shape: tuple[int, ...]
    value_count: int
    packed_byte_count: int
    io_pack_bits: int | None


@dataclasses.dataclass(frozen=True)
class TypeLayout:
    """Cached type-level layout facts for an xtructure dataclass type."""

    cls: type
    fields: tuple[FieldLayout, ...]
    field_by_name: Mapping[str, FieldLayout]
    adapter_field_plans: tuple[AdapterFieldPlan, ...]
    adapter_field_plan_by_name: Mapping[str, AdapterFieldPlan]
    field_names: tuple[str, ...]
    intrinsic_shapes: tuple[tuple[int, ...], ...]
    default_shape: Any
    default_dtype: Any
    dtype_tuple_cls: type
    shape_tuple_cls: type
    leaves: tuple[LeafLayout, ...]
    leaf_by_path: Mapping[tuple[str, ...], LeafLayout]
    packed_fields: tuple[FieldLayout, ...]
    packed_field_layouts: tuple[PackedFieldLayout, ...]
    packed_field_layout_by_name: Mapping[str, PackedFieldLayout]
    aggregate_bitpack: AggregateBitpackLayout


@dataclasses.dataclass(frozen=True)
class InstanceFieldLayout:
    """Concrete field shape facts for an instance."""

    name: str
    path: tuple[str, ...]
    shape: Any
    batch_shape: tuple[int, ...] | int
    mismatch_reason: str | None = None


@dataclasses.dataclass(frozen=True)
class InstanceLayout:
    """Instance-level layout facts interpreted against a Type Layout."""

    type_layout: TypeLayout
    shape_tuple: Any
    dtype_tuple: Any
    batch_shape: tuple[int, ...] | int
    structured_type: StructuredType
    field_shapes: dict[str, Any]
    fields: tuple[InstanceFieldLayout, ...]
    mismatch_reason: str | None
