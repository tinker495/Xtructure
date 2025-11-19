from typing import Any, Callable, Dict, Tuple, Type

import jax.numpy as jnp

# Represents a JAX dtype, can be a specific type like jnp.int32 or a more generic jnp.dtype
DType = Any
_FIELD_DESCRIPTOR_ATTR = "__xtructure_field_descriptors__"


class FieldDescriptor:
    """
    A descriptor for fields in an xtructure_dataclass.

    This class is used to define the properties of fields in a dataclass decorated with
    @xtructure_dataclass. It specifies the JAX dtype, shape, and default fill value
    for each field.

    Example usage:
        ```python
        @xtructure_dataclass
        class MyData:
            # A scalar uint8 field
            a: FieldDescriptor.scalar(dtype=jnp.uint8)

            # A field with shape (1, 2) of uint32 values
            b: FieldDescriptor.tensor(dtype=jnp.uint32, shape=(1, 2))

            # A float field with custom fill value
            c: FieldDescriptor.scalar(dtype=jnp.float32, default=0.0)

            # A nested xtructure_dataclass field
            d: FieldDescriptor.scalar(dtype=AnotherDataClass)
        ```

    The FieldDescriptor can be used with type annotation syntax using square brackets
    or instantiated directly with the constructor for more explicit parameter naming.
    Describes a field in an xtructure_data class, specifying its JAX dtype,
    a default fill value, and its intrinsic (non-batched) shape.
    This allows for auto-generation of the .default() classmethod.
    """

    def __init__(
        self,
        dtype: DType,
        intrinsic_shape: Tuple[int, ...] = (),
        fill_value: Any = None,
        *,
        fill_value_factory: Callable[[Tuple[int, ...], DType], Any] | None = None,
        validator: Callable[[Any], None] | None = None,
    ):
        """
        Initializes a FieldDescriptor.

        Args:
            dtype: The JAX dtype of the field (e.g., jnp.int32, jnp.float32).
            fill_value: The default value to fill the field's array with
                        (e.g., -1, 0.0). Mutually exclusive with ``fill_value_factory``.
            fill_value_factory: Callable that receives ``(field_shape, dtype)`` and returns
                        a value (or array) used to initialize the field. Useful when the
                        sentinel depends on the requested batch shape. Mutually exclusive
                        with ``fill_value``.
            intrinsic_shape: The shape of the field itself, before any batching.
                             Defaults to () for a scalar field.
            validator: Optional callable that raises an exception if the value is invalid.
        """
        if fill_value is not None and fill_value_factory is not None:
            raise ValueError("Provide only one of fill_value or fill_value_factory.")

        self.dtype: DType = dtype
        self.fill_value_factory = fill_value_factory
        self.validator = validator
        # Set default fill values based on dtype
        if fill_value is not None:
            self.fill_value = fill_value
        elif fill_value_factory is not None:
            self.fill_value = None
        else:
            self.fill_value = _default_fill_value_for_dtype(dtype)
        self.intrinsic_shape: Tuple[int, ...] = intrinsic_shape

    def __repr__(self) -> str:
        return (
            f"FieldDescriptor(dtype={self.dtype}, "
            f"fill_value={self.fill_value}, "
            f"intrinsic_shape={self.intrinsic_shape}, "
            f"fill_value_factory={self.fill_value_factory}, "
            f"validator={self.validator})"
        )

    @classmethod
    def tensor(
        cls,
        dtype: DType,
        shape: Tuple[int, ...],
        *,
        fill_value: Any = None,
        fill_value_factory: Callable[[Tuple[int, ...], DType], Any] | None = None,
        validator: Callable[[Any], None] | None = None,
    ) -> "FieldDescriptor":
        """
        Explicit factory method for creating a tensor field descriptor.
        """
        return cls(
            dtype=dtype,
            intrinsic_shape=shape,
            fill_value=fill_value,
            fill_value_factory=fill_value_factory,
            validator=validator,
        )

    @classmethod
    def scalar(
        cls,
        dtype: DType,
        *,
        default: Any = None,
        fill_value_factory: Callable[[Tuple[int, ...], DType], Any] | None = None,
        validator: Callable[[Any], None] | None = None,
    ) -> "FieldDescriptor":
        """
        Explicit factory method for creating a scalar field descriptor.
        """
        return cls(
            dtype=dtype,
            intrinsic_shape=(),
            fill_value=default,
            fill_value_factory=fill_value_factory,
            validator=validator,
        )

    @classmethod
    def __class_getitem__(cls, item: Any) -> "FieldDescriptor":
        """
        Allows for syntax like FieldDescriptor[dtype, intrinsic_shape, fill_value].
        """
        if isinstance(item, tuple):
            if len(item) == 1:
                return cls(item[0])
            elif len(item) == 2:
                # Assuming item[1] is intrinsic_shape or fill_value.
                # Heuristic: if it's a tuple, it's intrinsic_shape. Otherwise, it could be fill_value.
                # This could be ambiguous. For clarity, users might prefer named args with __init__
                # or a more structured approach if this becomes complex.
                if isinstance(item[1], tuple):
                    return cls(item[0], intrinsic_shape=item[1])
                else:  # Assuming it's a fill_value, and intrinsic_shape is default
                    return cls(item[0], fill_value=item[1])
            elif len(item) == 3:
                return cls(item[0], intrinsic_shape=item[1], fill_value=item[2])
            else:
                raise ValueError(
                    "FieldDescriptor[...] expects 1 to 3 arguments: "
                    "dtype, optional intrinsic_shape, optional fill_value"
                )
        else:
            # Single item is treated as dtype
            return cls(item)


def _default_fill_value_for_dtype(dtype: DType) -> Any:
    """Return a dtype-aware sentinel that plays nicely with jnp.full."""
    if isinstance(dtype, type) and hasattr(dtype, "is_xtructed"):
        return None

    try:
        dtype_obj = jnp.dtype(dtype)
    except TypeError:
        return None

    if jnp.issubdtype(dtype_obj, jnp.bool_):
        return False
    if jnp.issubdtype(dtype_obj, jnp.unsignedinteger):
        return jnp.iinfo(dtype_obj).max
    if jnp.issubdtype(dtype_obj, jnp.integer):
        return 0
    if jnp.issubdtype(dtype_obj, jnp.floating):
        return jnp.inf
    return None


# Example usage (to be placed in your class definitions later):
#
# from xtructure.field_descriptors import FieldDescriptor
#
# @xtructure_data
# class MyData:
#     my_scalar_int: FieldDescriptor.scalar(dtype=jnp.int32, default=-1)
#     my_vector_float: FieldDescriptor.tensor(dtype=jnp.float32, shape=(10,), fill_value=0.0)
#     my_default_shape_int: FieldDescriptor.scalar(dtype=jnp.uint8)
#     # ... other fields
#
#     # The .default() method would be auto-generated by @xtructure_data
#     # using these descriptors.
#


def _descriptor_from_annotation(annotation: Any) -> FieldDescriptor | None:
    if isinstance(annotation, FieldDescriptor):
        return annotation
    metadata = getattr(annotation, "__metadata__", ())
    for meta in metadata:
        if isinstance(meta, FieldDescriptor):
            return meta
    return None


def extract_field_descriptors_from_annotations(
    annotations: Dict[str, Any]
) -> Dict[str, FieldDescriptor]:
    field_descriptors: Dict[str, FieldDescriptor] = {}
    invalid_annotations = []

    for field_name, annotation in annotations.items():
        descriptor = _descriptor_from_annotation(annotation)
        if descriptor is None:
            invalid_annotations.append((field_name, type(annotation).__name__))
        else:
            field_descriptors[field_name] = descriptor

    if invalid_annotations:
        raise ValueError(
            "xtructure_dataclass fields must declare metadata via FieldDescriptor. "
            "Use either the legacy style `field: FieldDescriptor(...)` or "
            "`field: typing.Annotated[ActualType, FieldDescriptor(...)]`. "
            f"Invalid annotations: {invalid_annotations}"
        )

    return field_descriptors


def cache_field_descriptors(cls: Type[Any]) -> Dict[str, FieldDescriptor]:
    descriptors = extract_field_descriptors_from_annotations(getattr(cls, "__annotations__", {}))
    setattr(cls, _FIELD_DESCRIPTOR_ATTR, descriptors)
    return descriptors


def get_field_descriptors(cls: Type[Any]) -> Dict[str, FieldDescriptor]:
    descriptors = getattr(cls, _FIELD_DESCRIPTOR_ATTR, None)
    if descriptors is None:
        descriptors = cache_field_descriptors(cls)
    return descriptors

