from collections import namedtuple
from operator import attrgetter
from typing import Type, TypeVar

import jax.numpy as jnp

from xtructure.core.field_descriptors import FieldDescriptor, get_field_descriptors
from xtructure.core.protocol import StructuredType

T = TypeVar("T")


def add_shape_dtype_len(cls: Type[T]) -> Type[T]:
    """
    Augments the class with `shape` and `dtype` properties to inspect its
    fields, and a `__len__` method.

    The `shape` and `dtype` properties return namedtuples reflecting the
    structure of the dataclass fields.
    The `__len__` method conventionally returns the size of the first
    dimension of the first field of the instance, which is often useful
    for determining batch sizes.
    """
    field_descriptors: dict[str, FieldDescriptor] = get_field_descriptors(cls)
    field_names = list(field_descriptors.keys())
    intrinsic_shapes = [fd.intrinsic_shape for fd in field_descriptors.values()]
    shape_tuple = namedtuple("shape", ["batch"] + field_names)
    default_shape = namedtuple("default_shape", field_names)(*intrinsic_shapes)
    default_dtype = namedtuple("default_dtype", field_names)(
        *[fd.dtype for fd in field_descriptors.values()]
    )

    cls.default_shape = default_shape
    cls.default_dtype = default_dtype
    _field_getter = attrgetter(*field_names) if field_names else None

    def get_shape(self) -> shape_tuple:
        """
        Returns a namedtuple containing the batch shape (if present) and the shapes of all fields.
        If a field is itself a xtructure_dataclass, its shape is included as a nested namedtuple.
        """
        # Return cached shape if available
        # Use simple attribute access with default.
        # 'shape' property itself prevents direct access to _shape_cache if valid?
        # No, _shape_cache is a separate attribute.
        try:
            return self._shape_cache
        except AttributeError:
            pass

        # Determine batch: if all fields have a leading batch dimension of the same size, use it.
        # Otherwise, batch is ().
        if not field_names:
            result = shape_tuple((), *[])
            object.__setattr__(self, "_shape_cache", result)
            return result

        values = _field_getter(self)
        if len(field_names) == 1:
            values = (values,)

        field_shapes = []
        batch_shapes = []
        for field_name, field_value, default_shape_field in zip(
            field_names, values, intrinsic_shapes
        ):
            # Check if the field is a nested xtructure instance before attempting to convert to array
            if hasattr(field_value, "is_xtructed"):
                shape = field_value.shape
            else:
                # Prefer direct `.shape` access (fast). Fall back to `jnp.asarray` only
                # for scalars / Python objects without `.shape`.
                shape = getattr(field_value, "shape", None)
                if shape is None:
                    shape = jnp.asarray(field_value).shape
            if (
                isinstance(shape, tuple)
                and hasattr(shape, "_fields")
                and shape.__class__.__name__ == "shape"
            ):
                # If the field is itself a xtructure_dataclass (nested shape_tuple)
                if default_shape_field == ():
                    batch_shapes.append(shape.batch)
                    shape = shape.__class__((), *shape[1:])
                elif shape.batch[-len(default_shape_field) :] == default_shape_field:
                    batch_shapes.append(shape.batch[: -len(default_shape_field)])
                    cuted_batch_shape = shape.batch[-len(default_shape_field) :]
                    shape = shape.__class__(cuted_batch_shape, *shape[1:])
                else:
                    batch_shapes.append(-1)
            else:
                if default_shape_field == ():
                    batch_shapes.append(shape)
                    shape = ()
                elif shape[-len(default_shape_field) :] == default_shape_field:
                    batch_shapes.append(shape[: -len(default_shape_field)])
                    shape = shape[-len(default_shape_field) :]
                else:
                    batch_shapes.append(-1)
            field_shapes.append(shape)

        final_batch_shape = batch_shapes[0]
        for batch_shape in batch_shapes[1:]:
            if batch_shape == -1 or final_batch_shape != batch_shape:
                final_batch_shape = -1
                break
        result = shape_tuple(final_batch_shape, *field_shapes)
        # Cache the result using object.__setattr__ to avoid potential frozen issues or overhead
        object.__setattr__(self, "_shape_cache", result)
        return result

    setattr(cls, "shape", property(get_shape))

    type_tuple = namedtuple("dtype", field_names)

    def get_type(self) -> type_tuple:
        """Get dtypes of all fields in the dataclass"""
        try:
            return self._dtype_cache
        except AttributeError:
            pass

        result = type_tuple(*[getattr(self, field_name).dtype for field_name in field_names])
        object.__setattr__(self, "_dtype_cache", result)
        return result

    setattr(cls, "dtype", property(get_type))

    def get_len(self):
        """Return batch size for BATCHED instances.

        Semantics:
        - SINGLE: returns 1
        - BATCHED: returns the first batch dimension (shape.batch[0])
        - UNSTRUCTURED: raises TypeError (batch size is ill-defined)
        """
        shape = self.shape
        batch = shape.batch
        if batch == ():
            return 1
        if batch == -1:
            raise TypeError(
                f"len() is not defined for UNSTRUCTURED {cls.__name__} instances. "
                f"shape={shape}, default_shape={getattr(self, 'default_shape', None)}"
            )
        return int(batch[0])

    setattr(cls, "__len__", get_len)

    def get_structured_type(self) -> StructuredType:
        try:
            return self._structured_type_cache
        except AttributeError:
            pass

        shape = self.shape
        if shape.batch == ():
            result = StructuredType.SINGLE
        elif shape.batch == -1:
            result = StructuredType.UNSTRUCTURED
        else:
            result = StructuredType.BATCHED

        object.__setattr__(self, "_structured_type_cache", result)
        return result

    setattr(cls, "structured_type", property(get_structured_type))

    def get_batch_shape(self):
        return self.shape.batch

    setattr(cls, "batch_shape", property(get_batch_shape))

    def get_ndim(self) -> int:
        """Return number of batch dimensions for structured instances."""
        shape = self.shape
        batch = shape.batch
        if batch == ():
            return 0
        if batch == -1:
            raise TypeError(
                f"ndim is not defined for UNSTRUCTURED {cls.__name__} instances. "
                f"shape={shape}, default_shape={getattr(self, 'default_shape', None)}"
            )
        return len(batch)

    setattr(cls, "ndim", property(get_ndim))

    return cls
