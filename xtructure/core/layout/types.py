"""Layout fact types for xtructure dataclasses.

The layout package is the single source of truth for interpreting
FieldDescriptor metadata into type-level and instance-level facts.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, Literal

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
        return self.packed_bits is not None and self.unpacked_intrinsic_shape is not None


@dataclasses.dataclass(frozen=True)
class AdapterFieldPlan:
    """Shared per-field execution facts consumed by first-party Layout Adapters.

    ``logical_intrinsic_shape`` is the Xtructure Schema shape. For packed fields,
    ``storage_intrinsic_shape`` is the byte-stream shape interpreted by Packed
    Field Layout. ``intrinsic_shape`` is kept as the adapter execution/storage
    shape for compatibility with existing Layout Adapters.
    """

    name: str
    path: tuple[str, ...]
    dotted_path: str
    field_kind: AdapterFieldKind
    declared_dtype: Any | None
    nested_type: type | None
    intrinsic_shape: tuple[int, ...]
    logical_intrinsic_shape: tuple[int, ...]
    storage_intrinsic_shape: tuple[int, ...]
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
    view_fields_by_owner: tuple[tuple[type, tuple[AggregateViewFieldLayout, ...]], ...]

    def view_fields_for(self, owner_type: type) -> tuple[AggregateViewFieldLayout, ...]:
        return _lookup_or_default(self.view_fields_by_owner, owner_type, ())


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
    field_by_name: tuple[tuple[str, FieldLayout], ...]
    adapter_field_plans: tuple[AdapterFieldPlan, ...]
    adapter_field_plan_by_name: tuple[tuple[str, AdapterFieldPlan], ...]
    field_names: tuple[str, ...]
    intrinsic_shapes: tuple[tuple[int, ...], ...]
    default_shape: Any
    default_dtype: Any
    dtype_tuple_cls: type
    shape_tuple_cls: type
    leaves: tuple[LeafLayout, ...]
    leaf_by_path: tuple[tuple[tuple[str, ...], LeafLayout], ...]
    packed_fields: tuple[FieldLayout, ...]
    packed_field_layouts: tuple[PackedFieldLayout, ...]
    packed_field_layout_by_name: tuple[tuple[str, PackedFieldLayout], ...]
    aggregate_bitpack: AggregateBitpackLayout

    def field_for(self, name: str) -> FieldLayout:
        return _lookup_required(self.field_by_name, name)

    def has_field(self, name: str) -> bool:
        return _contains_key(self.field_by_name, name)

    def adapter_field_plan_for(self, name: str) -> AdapterFieldPlan:
        return _lookup_required(self.adapter_field_plan_by_name, name)

    def leaf_for(self, path: tuple[str, ...]) -> LeafLayout:
        return _lookup_required(self.leaf_by_path, path)

    def packed_field_layout_for(self, name: str) -> PackedFieldLayout:
        return _lookup_required(self.packed_field_layout_by_name, name)

    def maybe_packed_field_layout_for(self, name: str) -> PackedFieldLayout | None:
        return _lookup_or_default(self.packed_field_layout_by_name, name, None)

    def storage_intrinsic_shape_for(self, field_or_name: FieldLayout | str) -> tuple[int, ...]:
        """Return the stored field shape interpreted by Type Layout.

        For normal fields this is the declared Intrinsic Shape. For packed
        fields it is the Packed Field Layout byte-stream Intrinsic Shape.
        """

        if isinstance(field_or_name, FieldLayout):
            field = field_or_name
            name = field.name
        else:
            name = field_or_name
            field = self.field_for(name)
        packed_layout = self.maybe_packed_field_layout_for(name)
        if packed_layout is not None:
            return packed_layout.storage_intrinsic_shape
        return field.intrinsic_shape


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

    cls: type
    shape_tuple: Any
    dtype_tuple: Any
    batch_shape: tuple[int, ...] | int
    structured_type: StructuredType
    field_shapes: tuple[tuple[str, Any], ...]
    fields: tuple[InstanceFieldLayout, ...]
    mismatch_reason: str | None

    def field_shape_for(self, name: str) -> Any:
        return _lookup_required(self.field_shapes, name)


def _lookup_required(pairs: tuple[tuple[Any, Any], ...], key: Any) -> Any:
    for pair_key, value in pairs:
        if pair_key == key:
            return value
    raise KeyError(key)


def _lookup_or_default(pairs: tuple[tuple[Any, Any], ...], key: Any, default: Any) -> Any:
    for pair_key, value in pairs:
        if pair_key == key:
            return value
    return default


def _contains_key(pairs: tuple[tuple[Any, Any], ...], key: Any) -> bool:
    return any(pair_key == key for pair_key, _ in pairs)
