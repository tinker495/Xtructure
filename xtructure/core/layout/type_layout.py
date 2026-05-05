"""Type-level xtructure layout interpretation."""

from __future__ import annotations

import dataclasses
import functools
from collections import namedtuple
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np

from xtructure.core.bitpack_math import packed_num_bytes
from xtructure.core.field_descriptors import FieldDescriptor, get_field_descriptors
from xtructure.core.shape_utils import normalize_shape
from xtructure.core.type_utils import is_xtructure_dataclass_type

from .bitpack import build_aggregate_bitpack_layout, default_unpack_dtype
from .types import (
    AdapterFieldPlan,
    FieldLayout,
    LeafLayout,
    PackedFieldLayout,
    TypeLayout,
)


def _field_layout(name: str, descriptor: FieldDescriptor, path: tuple[str, ...]) -> FieldLayout:
    is_nested = is_xtructure_dataclass_type(descriptor.dtype)
    return FieldLayout(
        name=name,
        path=path,
        dtype=descriptor.dtype,
        intrinsic_shape=normalize_shape(descriptor.intrinsic_shape),
        is_nested=is_nested,
        nested_type=descriptor.dtype if is_nested else None,
        bits=descriptor.bits,
        packed_bits=descriptor.packed_bits,
        unpacked_dtype=descriptor.unpacked_dtype,
        unpacked_intrinsic_shape=(
            normalize_shape(descriptor.unpacked_intrinsic_shape)
            if descriptor.unpacked_intrinsic_shape is not None
            else None
        ),
        fill_value=descriptor.fill_value,
        fill_value_factory=descriptor.fill_value_factory,
        validator=descriptor.validator,
    )


def _build_fields(cls: type) -> tuple[FieldLayout, ...]:
    descriptors = get_field_descriptors(cls)
    fields: list[FieldLayout] = []
    for dc_field in dataclasses.fields(cls):
        descriptor = descriptors.get(dc_field.name)
        if descriptor is None:
            continue
        fields.append(_field_layout(dc_field.name, descriptor, (dc_field.name,)))
    return tuple(fields)


def _build_leaves(
    fields: tuple[FieldLayout, ...],
    *,
    prefix: tuple[str, ...] = (),
    parent_intrinsic_shape: tuple[int, ...] = (),
) -> tuple[LeafLayout, ...]:
    leaves: list[LeafLayout] = []
    for field in fields:
        path = prefix + (field.name,)
        field_parent_shape = parent_intrinsic_shape + field.intrinsic_shape
        if field.is_nested:
            nested_layout = get_type_layout(field.nested_type)  # type: ignore[arg-type]
            leaves.extend(
                _build_leaves(
                    nested_layout.fields,
                    prefix=path,
                    parent_intrinsic_shape=field_parent_shape,
                )
            )
            continue
        leaves.append(
            LeafLayout(
                path=path,
                declared_dtype=field.dtype,
                intrinsic_shape=field_parent_shape,
                local_intrinsic_shape=field.intrinsic_shape,
                parent_intrinsic_shape=parent_intrinsic_shape,
                bits=field.bits,
                packed_bits=field.packed_bits,
                unpacked_dtype=field.unpacked_dtype,
                unpacked_intrinsic_shape=field.unpacked_intrinsic_shape,
                io_pack_bits=None if field.packed_bits is not None else field.bits,
            )
        )
    return tuple(leaves)


def _is_primitive_jax_dtype(dtype) -> bool:
    try:
        return bool(jnp.issubdtype(dtype, jnp.number) or jnp.issubdtype(dtype, jnp.bool_))
    except TypeError:
        return False


def _random_facts(dtype) -> tuple[str, object | None, bool, object | None]:
    try:
        dtype_obj = jnp.dtype(dtype)
    except TypeError:
        return "other", None, False, dtype

    if jnp.issubdtype(dtype_obj, jnp.bool_):
        return "bool", None, False, dtype_obj
    if jnp.issubdtype(dtype_obj, jnp.unsignedinteger):
        return "bits_int", dtype_obj, False, dtype_obj
    if jnp.issubdtype(dtype_obj, jnp.integer):
        unsigned_equivalent = jnp.dtype(f"uint{np.dtype(dtype_obj).itemsize * 8}")
        return "bits_int", unsigned_equivalent, True, dtype_obj
    if jnp.issubdtype(dtype_obj, jnp.floating):
        return "float", None, False, dtype_obj
    return "other", None, False, dtype_obj


def _build_adapter_field_plan(field: FieldLayout) -> AdapterFieldPlan:
    if field.is_nested:
        return AdapterFieldPlan(
            name=field.name,
            path=field.path,
            dotted_path=".".join(field.path),
            field_kind="nested",
            declared_dtype=None,
            nested_type=field.nested_type,
            intrinsic_shape=field.intrinsic_shape,
            fill_value=None,
            fill_value_factory=None,
            validator=field.validator,
            random_kind="nested",
            is_primitive_jax_dtype=False,
        )

    random_kind, random_bits_dtype, random_view_as_signed, random_gen_dtype = _random_facts(
        field.dtype
    )
    return AdapterFieldPlan(
        name=field.name,
        path=field.path,
        dotted_path=".".join(field.path),
        field_kind="primitive",
        declared_dtype=field.dtype,
        nested_type=None,
        intrinsic_shape=field.intrinsic_shape,
        fill_value=field.fill_value,
        fill_value_factory=field.fill_value_factory,
        validator=field.validator,
        random_kind=random_kind,  # type: ignore[arg-type]
        random_bits_dtype=random_bits_dtype,
        random_view_as_signed=random_view_as_signed,
        random_gen_dtype=random_gen_dtype,
        is_primitive_jax_dtype=_is_primitive_jax_dtype(field.dtype),
    )


def _num_values(shape: tuple[int, ...]) -> int:
    return int(np.prod(np.array(shape, dtype=np.int64))) if shape else 1


def _build_packed_field_layout(field: FieldLayout) -> PackedFieldLayout | None:
    if not field.is_packed:
        return None
    packed_bits = int(field.packed_bits)  # type: ignore[arg-type]
    unpacked_shape = tuple(field.unpacked_intrinsic_shape)  # type: ignore[arg-type]
    unpack_dtype = field.unpacked_dtype or default_unpack_dtype(packed_bits)
    value_count = _num_values(unpacked_shape)
    return PackedFieldLayout(
        name=field.name,
        path=field.path,
        storage_dtype=field.dtype,
        storage_intrinsic_shape=field.intrinsic_shape,
        packed_bits=packed_bits,
        unpacked_dtype=unpack_dtype,
        unpacked_intrinsic_shape=unpacked_shape,
        value_count=value_count,
        packed_byte_count=packed_num_bytes(value_count, packed_bits),
        # In-memory packed fields are already byte streams and should not be IO-packed again.
        io_pack_bits=None,
    )


@functools.cache
def get_type_layout(cls: type) -> TypeLayout:
    """Return cached Type Layout facts for an xtructure dataclass type."""
    fields = _build_fields(cls)
    field_names = tuple(field.name for field in fields)
    intrinsic_shapes = tuple(field.intrinsic_shape for field in fields)
    shape_tuple_cls = namedtuple("shape", ["batch"] + list(field_names))
    dtype_tuple_cls = namedtuple("dtype", field_names)
    default_shape = namedtuple("default_shape", field_names)(*intrinsic_shapes)
    default_dtype = namedtuple("default_dtype", field_names)(*[field.dtype for field in fields])
    leaves = _build_leaves(fields)
    adapter_field_plans = tuple(_build_adapter_field_plan(field) for field in fields)
    adapter_field_plan_by_name = MappingProxyType({plan.name: plan for plan in adapter_field_plans})
    field_by_name = MappingProxyType({field.name: field for field in fields})
    leaf_by_path = MappingProxyType({leaf.path: leaf for leaf in leaves})
    packed_fields = tuple(field for field in fields if field.is_packed)
    packed_field_layouts = tuple(
        packed_layout
        for field in fields
        for packed_layout in (_build_packed_field_layout(field),)
        if packed_layout is not None
    )
    packed_field_layout_by_name = MappingProxyType(
        {packed_layout.name: packed_layout for packed_layout in packed_field_layouts}
    )
    aggregate_bitpack = build_aggregate_bitpack_layout(cls, fields)
    return TypeLayout(
        cls=cls,
        fields=fields,
        field_by_name=field_by_name,
        adapter_field_plans=adapter_field_plans,
        adapter_field_plan_by_name=adapter_field_plan_by_name,
        field_names=field_names,
        intrinsic_shapes=intrinsic_shapes,
        default_shape=default_shape,
        default_dtype=default_dtype,
        dtype_tuple_cls=dtype_tuple_cls,
        shape_tuple_cls=shape_tuple_cls,
        leaves=leaves,
        leaf_by_path=leaf_by_path,
        packed_fields=packed_fields,
        packed_field_layouts=packed_field_layouts,
        packed_field_layout_by_name=packed_field_layout_by_name,
        aggregate_bitpack=aggregate_bitpack,
    )
