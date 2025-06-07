from collections import namedtuple
from typing import Type, TypeVar

from xtructure.core.field_descriptors import FieldDescriptor
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
    shape_tuple = namedtuple("shape", ["batch"] + list(cls.__annotations__.keys()))
    field_descriptors: dict[str, FieldDescriptor] = cls.__annotations__
    default_shape = namedtuple("default_shape", cls.__annotations__.keys())(
        *[fd.intrinsic_shape for fd in field_descriptors.values()]
    )
    default_dtype = namedtuple("default_dtype", cls.__annotations__.keys())(
        *[fd.dtype for fd in field_descriptors.values()]
    )

    cls.default_shape = default_shape
    cls.default_dtype = default_dtype

    def get_shape(self) -> shape_tuple:
        """
        Returns a namedtuple containing the batch shape (if present) and the shapes of all fields.
        If a field is itself a xtructure_dataclass, its shape is included as a nested namedtuple.
        """
        # Determine batch: if all fields have a leading batch dimension of the same size, use it.
        # Otherwise, batch is ().
        field_shapes = []
        batch_shapes = []
        for field_name in cls.__annotations__.keys():
            shape = getattr(self, field_name).shape
            default_shape_field = getattr(default_shape, field_name)
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
            if batch_shape == -1:
                final_batch_shape = -1
                break
            if final_batch_shape != batch_shape:
                final_batch_shape = -1
                break
        return shape_tuple(final_batch_shape, *field_shapes)

    setattr(cls, "shape", property(get_shape))

    type_tuple = namedtuple("dtype", cls.__annotations__.keys())

    def get_type(self) -> type_tuple:
        """Get dtypes of all fields in the dataclass"""
        return type_tuple(
            *[getattr(self, field_name).dtype for field_name in cls.__annotations__.keys()]
        )

    setattr(cls, "dtype", property(get_type))

    def get_len(self):
        """Get length of the first field's first dimension"""
        return self.shape[0][0]

    setattr(cls, "__len__", get_len)

    def get_structured_type(self) -> StructuredType:
        shape = self.shape
        if shape.batch == ():
            return StructuredType.SINGLE
        elif shape.batch == -1:
            return StructuredType.UNSTRUCTURED
        else:
            return StructuredType.BATCHED

    setattr(cls, "structured_type", property(get_structured_type))

    return cls
