"""Instance-level xtructure layout interpretation."""

from __future__ import annotations

import functools
from operator import attrgetter
from typing import Any

import jax.numpy as jnp

from xtructure.core.structuredtype import StructuredType

from .type_layout import get_type_layout
from .types import InstanceFieldLayout, InstanceLayout


def primitive_value_shape(value: Any) -> tuple[int, ...]:
    """Raw shape of a primitive (non-nested) xtructure field value.

    Internal Layout Adapter Interface helper: shared between Instance Layout
    construction and the Layout Cache replace fast path.
    """
    shape = getattr(value, "shape", None)
    if shape is None:
        shape = jnp.asarray(value).shape
    return tuple(shape)


def primitive_value_dtype(value: Any):
    """Raw dtype of a primitive (non-nested) xtructure field value.

    Internal Layout Adapter Interface helper: shared between Instance Layout
    construction and the Layout Cache replace fast path.
    """
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        dtype = jnp.asarray(value).dtype
    return dtype


def _value_shape_for_field(field_is_nested: bool, value: Any) -> Any:
    if field_is_nested:
        return value.shape
    return primitive_value_shape(value)


def _interpret_field_shape(
    field_name: str,
    value_shape: Any,
    intrinsic_shape: tuple[int, ...],
    nested_shape_cls: type | None,
):
    """Interpret a value's shape against the field's intrinsic shape.

    `nested_shape_cls` is the layout-canonical shape namedtuple for the field's
    nested xtructure type (or None for primitive fields). Compatible shape
    namedtuples are accepted so equivalent xtructure class objects created by
    reload/redefinition still infer the same layout.
    """
    if nested_shape_cls is not None and _is_compatible_nested_shape(value_shape, nested_shape_cls):
        if value_shape.batch == -1:
            return (
                value_shape,
                -1,
                f"{field_name} nested value is UNSTRUCTURED.",
            )
        if intrinsic_shape == ():
            return value_shape.__class__((), *value_shape[1:]), value_shape.batch, None
        if value_shape.batch[-len(intrinsic_shape) :] == intrinsic_shape:
            batch_shape = value_shape.batch[: -len(intrinsic_shape)]
            nested_batch = value_shape.batch[-len(intrinsic_shape) :]
            return value_shape.__class__(nested_batch, *value_shape[1:]), batch_shape, None
        return (
            value_shape,
            -1,
            f"{field_name} nested batch {value_shape.batch} does not end with intrinsic shape {intrinsic_shape}.",
        )

    shape = tuple(value_shape)
    if intrinsic_shape == ():
        return (), shape, None
    if shape[-len(intrinsic_shape) :] == intrinsic_shape:
        return shape[-len(intrinsic_shape) :], shape[: -len(intrinsic_shape)], None
    return (
        shape,
        -1,
        f"{field_name} shape {shape} does not end with intrinsic shape {intrinsic_shape}.",
    )


def _is_compatible_nested_shape(value_shape: Any, nested_shape_cls: type) -> bool:
    """Return true for canonical or structurally equivalent nested shape tuples."""
    if type(value_shape) is nested_shape_cls:
        return True

    expected_fields = getattr(nested_shape_cls, "_fields", None)
    value_fields = getattr(value_shape, "_fields", None)
    if expected_fields is None or value_fields is None:
        return False

    return (
        nested_shape_cls.__name__ == value_shape.__class__.__name__ == "shape"
        and tuple(value_fields) == tuple(expected_fields)
        and hasattr(value_shape, "batch")
        and len(value_shape) == len(expected_fields)
    )


def get_instance_layout(instance: Any) -> InstanceLayout:
    """Return Instance Layout facts for a concrete xtructure instance."""
    type_layout, getter = _type_layout_and_getter(instance.__class__)
    if type_layout.field_names:
        values = getter(instance)
        if len(type_layout.field_names) == 1:
            values = (values,)
    else:
        values = ()

    value_shapes = tuple(
        _value_shape_for_field(field.is_nested, value)
        for field, value in zip(type_layout.fields, values)
    )
    value_dtypes = tuple(
        value.dtype if field.is_nested else primitive_value_dtype(value)
        for field, value in zip(type_layout.fields, values)
    )
    return _build_instance_layout_from_signatures(instance.__class__, value_shapes, value_dtypes)


@functools.cache
def _type_layout_and_getter(cls: type):
    type_layout = get_type_layout(cls)
    getter = attrgetter(*type_layout.field_names) if type_layout.field_names else None
    return type_layout, getter


@functools.cache
def _build_instance_layout_from_signatures(
    cls: type,
    value_shapes: tuple[Any, ...],
    value_dtypes: tuple[Any, ...],
) -> InstanceLayout:
    """Build Instance Layout from hashable shape/dtype signatures."""
    type_layout = get_type_layout(cls)
    field_shapes: list[Any] = []
    batch_shapes: list[tuple[int, ...] | int] = []
    instance_fields: list[InstanceFieldLayout] = []
    mismatch_reasons: list[str] = []

    for field, raw_shape in zip(type_layout.fields, value_shapes):
        nested_shape_cls = (
            get_type_layout(field.nested_type).shape_tuple_cls if field.is_nested else None
        )
        interpreted_shape, batch_shape, reason = _interpret_field_shape(
            field.name, raw_shape, field.intrinsic_shape, nested_shape_cls
        )
        field_shapes.append(interpreted_shape)
        batch_shapes.append(batch_shape)
        if reason is not None:
            mismatch_reasons.append(reason)
        instance_fields.append(
            InstanceFieldLayout(
                name=field.name,
                path=field.path,
                shape=interpreted_shape,
                batch_shape=batch_shape,
                mismatch_reason=reason,
            )
        )

    final_batch_shape: tuple[int, ...] | int = batch_shapes[0] if batch_shapes else ()
    for batch_shape in batch_shapes[1:]:
        if batch_shape == -1 or final_batch_shape != batch_shape:
            final_batch_shape = -1
            mismatch_reasons.append(f"field batch shapes disagree: {tuple(batch_shapes)}.")
            break

    shape_tuple = type_layout.shape_tuple_cls(final_batch_shape, *field_shapes)
    dtype_tuple = type_layout.dtype_tuple_cls(*value_dtypes)

    if final_batch_shape == ():
        structured_type = StructuredType.SINGLE
    elif final_batch_shape == -1:
        structured_type = StructuredType.UNSTRUCTURED
    else:
        structured_type = StructuredType.BATCHED

    return InstanceLayout(
        cls=cls,
        shape_tuple=shape_tuple,
        dtype_tuple=dtype_tuple,
        batch_shape=final_batch_shape,
        structured_type=structured_type,
        field_shapes=tuple(
            (field.name, shape) for field, shape in zip(type_layout.fields, field_shapes)
        ),
        fields=tuple(instance_fields),
        mismatch_reason="; ".join(mismatch_reasons) if mismatch_reasons else None,
    )
