"""Instance-level xtructure layout interpretation."""

from __future__ import annotations

from operator import attrgetter
from typing import Any

import jax.numpy as jnp

from xtructure.core.structuredtype import StructuredType

from .type_layout import get_type_layout
from .types import InstanceFieldLayout, InstanceLayout


def _value_shape(value: Any) -> Any:
    if hasattr(value, "is_xtructed"):
        return value.shape
    shape = getattr(value, "shape", None)
    if shape is None:
        shape = jnp.asarray(value).shape
    return shape


def _interpret_field_shape(
    field_name: str,
    value_shape: Any,
    intrinsic_shape: tuple[int, ...],
    nested_shape_cls: type | None,
):
    """Interpret a value's shape against the field's intrinsic shape.

    `nested_shape_cls` is the layout-canonical shape namedtuple for the field's
    nested xtructure type (or None for primitive fields). We dispatch on the
    exact class identity to avoid matching unrelated user-defined namedtuples.
    """
    if nested_shape_cls is not None and type(value_shape) is nested_shape_cls:
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


def get_instance_layout(instance: Any) -> InstanceLayout:
    """Return Instance Layout facts for a concrete xtructure instance."""
    type_layout = get_type_layout(instance.__class__)
    if type_layout.field_names:
        getter = attrgetter(*type_layout.field_names)
        values = getter(instance)
        if len(type_layout.field_names) == 1:
            values = (values,)
    else:
        values = ()

    field_shapes: list[Any] = []
    batch_shapes: list[tuple[int, ...] | int] = []
    instance_fields: list[InstanceFieldLayout] = []
    mismatch_reasons: list[str] = []

    for field, value in zip(type_layout.fields, values):
        raw_shape = _value_shape(value)
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
    dtype_tuple = type_layout.dtype_tuple_cls(*(value.dtype for value in values))

    if final_batch_shape == ():
        structured_type = StructuredType.SINGLE
    elif final_batch_shape == -1:
        structured_type = StructuredType.UNSTRUCTURED
    else:
        structured_type = StructuredType.BATCHED

    return InstanceLayout(
        type_layout=type_layout,
        shape_tuple=shape_tuple,
        dtype_tuple=dtype_tuple,
        batch_shape=final_batch_shape,
        structured_type=structured_type,
        field_shapes={field.name: shape for field, shape in zip(type_layout.fields, field_shapes)},
        fields=tuple(instance_fields),
        mismatch_reason="; ".join(mismatch_reasons) if mismatch_reasons else None,
    )
