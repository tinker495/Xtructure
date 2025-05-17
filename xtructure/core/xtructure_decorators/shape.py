from collections import namedtuple
from typing import Type, TypeVar

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
    shape_tuple = namedtuple("shape", cls.__annotations__.keys())

    def get_shape(self) -> shape_tuple:
        """Get shapes of all fields in the dataclass"""
        return shape_tuple(
            *[getattr(self, field_name).shape for field_name in cls.__annotations__.keys()]
        )

    setattr(cls, "shape", property(get_shape))

    type_tuple = namedtuple("dtype", cls.__annotations__.keys())

    def get_type(self) -> type_tuple:
        """Get dtypes of all fields in the dataclass"""
        return type_tuple(
            *[getattr(self, field_name).dtype for field_name in cls.__annotations__.keys()]
        )

    setattr(cls, "dtype", property(get_type))

    def len(self):
        """Get length of the first field's first dimension"""
        return self.shape[0][0]

    setattr(cls, "__len__", len)
    return cls
