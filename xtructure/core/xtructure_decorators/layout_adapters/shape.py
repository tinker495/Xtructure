from typing import Type, TypeVar

from xtructure.core.layout import get_type_layout
from xtructure.core.structuredtype import StructuredType

T = TypeVar("T")


def add_shape_dtype_len(cls: Type[T]) -> Type[T]:
    """Attach shape/dtype/structured-type properties using Xtructure Layout facts."""
    type_layout = get_type_layout(cls)

    cls.default_shape = type_layout.default_shape
    cls.default_dtype = type_layout.default_dtype

    def get_shape(self):
        """Shared leading axes: () for SINGLE, (N, ...) for BATCHED.

        UNSTRUCTURED raises TypeError, like ndim and len(); `layout` still
        reports the mismatch.
        """
        layout = self._layout_cache
        if layout.batch_shape == -1:
            raise TypeError(
                f"shape is not defined for UNSTRUCTURED {cls.__name__} instances. "
                f"shape={layout.shape_tuple}, default_shape={self.default_shape}"
            )
        return layout.batch_shape

    setattr(cls, "shape", property(get_shape))

    def get_layout(self):
        return self._layout_cache

    setattr(cls, "layout", property(get_layout))

    def get_type(self):
        return self._layout_cache.dtype_tuple

    setattr(cls, "dtype", property(get_type))

    def get_len(self):
        """Return batch size for BATCHED instances.

        Semantics:
        - SINGLE: returns 1
        - BATCHED: returns the first batch dimension (shape[0])
        - UNSTRUCTURED: raises TypeError (batch size is ill-defined)
        """
        batch = self.shape
        return int(batch[0]) if batch else 1

    setattr(cls, "__len__", get_len)

    def get_structured_type(self) -> StructuredType:
        return self._layout_cache.structured_type

    setattr(cls, "structured_type", property(get_structured_type))

    def get_batch_shape(self):
        return self._layout_cache.batch_shape

    setattr(cls, "batch_shape", property(get_batch_shape))

    def get_ndim(self) -> int:
        """Return number of batch dimensions for structured instances."""
        return len(self.shape)

    setattr(cls, "ndim", property(get_ndim))

    return cls
