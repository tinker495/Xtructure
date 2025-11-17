from __future__ import annotations

from typing import Any, Iterable, Tuple

from .field_descriptors import FieldDescriptor

_UNSET = object()


def _normalize_shape(shape: Iterable[int] | Tuple[int, ...]) -> Tuple[int, ...]:
    if isinstance(shape, tuple):
        return shape
    return tuple(shape)


def clone_field_descriptor(
    descriptor: FieldDescriptor,
    *,
    dtype: Any = _UNSET,
    intrinsic_shape: Iterable[int] | Tuple[int, ...] | None = _UNSET,
    fill_value: Any = _UNSET,
    fill_value_factory: Any = _UNSET,
) -> FieldDescriptor:
    """
    Create a new FieldDescriptor derived from ``descriptor`` while overriding
    selected attributes.
    """

    if fill_value is not _UNSET and fill_value_factory is not _UNSET:
        raise ValueError("Provide only one of fill_value or fill_value_factory.")

    next_dtype = descriptor.dtype if dtype is _UNSET else dtype

    if intrinsic_shape is _UNSET:
        next_intrinsic_shape = descriptor.intrinsic_shape
    else:
        next_intrinsic_shape = _normalize_shape(intrinsic_shape)

    if fill_value is _UNSET and fill_value_factory is _UNSET:
        next_fill_value = descriptor.fill_value
        next_fill_value_factory = descriptor.fill_value_factory
    elif fill_value_factory is not _UNSET:
        next_fill_value = None
        next_fill_value_factory = fill_value_factory
    else:
        next_fill_value = fill_value
        next_fill_value_factory = None

    return FieldDescriptor(
        dtype=next_dtype,
        intrinsic_shape=next_intrinsic_shape,
        fill_value=next_fill_value,
        fill_value_factory=next_fill_value_factory,
    )


def with_intrinsic_shape(
    descriptor: FieldDescriptor, intrinsic_shape: Iterable[int] | Tuple[int, ...]
) -> FieldDescriptor:
    """Return a copy of ``descriptor`` with a new intrinsic shape."""
    return clone_field_descriptor(descriptor, intrinsic_shape=intrinsic_shape)


def broadcast_intrinsic_shape(
    descriptor: FieldDescriptor, batch_shape: Iterable[int] | Tuple[int, ...]
) -> FieldDescriptor:
    """
    Prepend ``batch_shape`` to the intrinsic shape, useful when scripting batched
    variants of an existing descriptor.
    """
    batch = _normalize_shape(batch_shape)
    new_shape = batch + descriptor.intrinsic_shape
    return clone_field_descriptor(descriptor, intrinsic_shape=new_shape)


def descriptor_metadata(descriptor: FieldDescriptor) -> dict[str, Any]:
    """Expose a descriptor's core metadata as a plain dict for tooling."""
    return {
        "dtype": descriptor.dtype,
        "intrinsic_shape": descriptor.intrinsic_shape,
        "fill_value": descriptor.fill_value,
        "fill_value_factory": descriptor.fill_value_factory,
    }

