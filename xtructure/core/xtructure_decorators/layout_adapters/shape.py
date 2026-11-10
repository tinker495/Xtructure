from typing import Type, TypeVar

from xtructure.core.layout import get_type_layout
from xtructure.core.protocol import StructuredType

T = TypeVar("T")


def add_shape_dtype_len(cls: Type[T]) -> Type[T]:
    """Attach shape/dtype/structured-type properties using Xtructure Layout facts."""
    type_layout = get_type_layout(cls)

    cls.default_shape = type_layout.default_shape
    cls.default_dtype = type_layout.default_dtype

    def get_shape(self):
        return self._layout_cache.shape_tuple

    setattr(cls, "shape", property(get_shape))

    def get_type(self):
        return self._layout_cache.dtype_tuple

    setattr(cls, "dtype", property(get_type))

    def get_len(self):
        """Return batch size for BATCHED instances.

        Semantics:
        - SINGLE: returns 1
        - BATCHED: returns the first batch dimension (shape.batch[0])
        - UNSTRUCTURED: raises TypeError (batch size is ill-defined)
        """
        layout = self._layout_cache
        batch = layout.batch_shape
        if batch == ():
            return 1
        if batch == -1:
            raise TypeError(
                f"len() is not defined for UNSTRUCTURED {cls.__name__} instances. "
                f"shape={layout.shape_tuple}, default_shape={getattr(self, 'default_shape', None)}"
            )
        return int(batch[0])

    setattr(cls, "__len__", get_len)

    def iterate(self):
        """Yield the items along the first batch dimension, like iterating an array.

        Defined explicitly because indexing past the end does not raise under JAX
        (indices are clamped), so the legacy ``__getitem__`` protocol never stops.
        """
        batch = self._layout_cache.batch_shape
        if batch == () or batch == -1:
            raise TypeError(f"iteration over an unbatched {cls.__name__} instance")
        for index in range(batch[0]):
            yield self[index]

    setattr(cls, "__iter__", iterate)

    def get_structured_type(self) -> StructuredType:
        return self._layout_cache.structured_type

    setattr(cls, "structured_type", property(get_structured_type))

    def get_batch_shape(self):
        return self._layout_cache.batch_shape

    setattr(cls, "batch_shape", property(get_batch_shape))

    def get_ndim(self) -> int:
        """Return number of batch dimensions for structured instances."""
        layout = self._layout_cache
        batch = layout.batch_shape
        if batch == ():
            return 0
        if batch == -1:
            raise TypeError(
                f"ndim is not defined for UNSTRUCTURED {cls.__name__} instances. "
                f"shape={layout.shape_tuple}, default_shape={getattr(self, 'default_shape', None)}"
            )
        return len(batch)

    setattr(cls, "ndim", property(get_ndim))

    return cls
