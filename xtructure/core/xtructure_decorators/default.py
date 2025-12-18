from typing import Any, Callable, Dict, List, NamedTuple
from typing import Tuple as TypingTuple
from typing import Type, TypeVar, Union

import jax.numpy as jnp

from xtructure.core.field_descriptors import FieldDescriptor, get_field_descriptors
from xtructure.core.type_utils import is_xtructure_dataclass_type

T = TypeVar("T")


class FieldInfo(NamedTuple):
    """Pre-computed field information for efficient default value generation."""

    name: str
    field_type: str
    # 'jax_primitive_descriptor', 'jax_dtype_descriptor', 'nested_class_descriptor', 'nested_class_direct'
    descriptor: Union[FieldDescriptor, None]
    dtype: Any
    fill_value: Any
    fill_value_factory: Union[Callable[[TypingTuple[int, ...], Any], Any], None]
    intrinsic_shape: TypingTuple[int, ...]
    nested_class_type: Union[Type, None]


def add_default_method(cls: Type[T]) -> Type[T]:
    field_descriptors = get_field_descriptors(cls)

    # add default method to class
    setattr(cls, "default", _create_default_method(cls, field_descriptors))
    return cls


def _create_default_method(
    cls_to_modify: Type[T], field_descriptors: Dict[str, FieldDescriptor]
) -> Callable[..., T]:

    # Pre-compute field information during method creation
    field_infos: List[FieldInfo] = []

    for field_name, descriptor in field_descriptors.items():
        dtype_of_field_descriptor = descriptor.dtype

        if is_xtructure_dataclass_type(dtype_of_field_descriptor):
            # It's a user-defined xtructure class. Use its .default() method.
            nested_class_type = dtype_of_field_descriptor
            if not hasattr(nested_class_type, "default"):
                raise TypeError(
                    f"Error during method creation for '{cls_to_modify.__name__}': "
                    f"Nested field '{field_name}' (type '{nested_class_type.__name__}' "
                    f"via FieldDescriptor.dtype) does not have a .default() method. "
                    f"Ensure it's an @xtructure_data class."
                )
            intrinsic_shape = (
                descriptor.intrinsic_shape
                if isinstance(descriptor.intrinsic_shape, tuple)
                else (descriptor.intrinsic_shape,)
            )
            field_infos.append(
                FieldInfo(
                    name=field_name,
                    field_type="nested_class_descriptor",
                    descriptor=descriptor,
                    dtype=None,
                    fill_value=None,
                    fill_value_factory=None,
                    intrinsic_shape=intrinsic_shape,
                    nested_class_type=nested_class_type,
                )
            )
        elif isinstance(dtype_of_field_descriptor, type):
            # Check if it's a JAX primitive type class
            is_jax_primitive_type_class = False
            try:
                if jnp.issubdtype(dtype_of_field_descriptor, jnp.number) or jnp.issubdtype(
                    dtype_of_field_descriptor, jnp.bool_
                ):
                    is_jax_primitive_type_class = True
            except TypeError:  # Not a type that jnp.issubdtype recognizes as a primitive base
                is_jax_primitive_type_class = False

            if is_jax_primitive_type_class:
                # It's like jnp.int32, jnp.float32. Use jnp.full.
                intrinsic_shape = (
                    descriptor.intrinsic_shape
                    if isinstance(descriptor.intrinsic_shape, tuple)
                    else (descriptor.intrinsic_shape,)
                )
                field_infos.append(
                    FieldInfo(
                        name=field_name,
                        field_type="jax_primitive_descriptor",
                        descriptor=descriptor,
                        dtype=dtype_of_field_descriptor,
                        fill_value=descriptor.fill_value,
                        fill_value_factory=descriptor.fill_value_factory,
                        intrinsic_shape=intrinsic_shape,
                        nested_class_type=None,
                    )
                )
            else:
                # It's some other type class that we don't support
                raise TypeError(
                    f"Error during method creation for '{cls_to_modify.__name__}': "
                    f"Field '{field_name}' uses FieldDescriptor with an unsupported "
                    f"type class: '{dtype_of_field_descriptor}' "
                    f"(type: {type(dtype_of_field_descriptor).__name__}). "
                    f"Expected a JAX primitive type/class (like jnp.int32) or an @xtructure_data class type."
                )
        elif isinstance(dtype_of_field_descriptor, jnp.dtype):
            # dtype_of_field_descriptor is a JAX dtype INSTANCE (e.g., jnp.dtype('int32')). Use jnp.full.
            intrinsic_shape = (
                descriptor.intrinsic_shape
                if isinstance(descriptor.intrinsic_shape, tuple)
                else (descriptor.intrinsic_shape,)
            )
            field_infos.append(
                FieldInfo(
                    name=field_name,
                    field_type="jax_dtype_descriptor",
                    descriptor=descriptor,
                    dtype=dtype_of_field_descriptor,
                    fill_value=descriptor.fill_value,
                    fill_value_factory=descriptor.fill_value_factory,
                    intrinsic_shape=intrinsic_shape,
                    nested_class_type=None,
                )
            )
        else:
            # FieldDescriptor.dtype is neither a recognized class nor a jnp.dtype instance.
            raise TypeError(
                f"Error during method creation for '{cls_to_modify.__name__}': "
                f"Field '{field_name}' uses FieldDescriptor with an unsupported "
                f".dtype attribute: '{dtype_of_field_descriptor}' "
                f"(type: {type(dtype_of_field_descriptor).__name__}). "
                f"Expected a JAX primitive type/class (like jnp.int32 or "
                f"jnp.dtype('int32')), or an @xtructure_data class type (like Parent)."
            )

    @classmethod
    def default(cls: Type[T], shape: TypingTuple[int, ...] = ()) -> T:
        default_values: Dict[str, Any] = {}

        def resolve_fill_value(field_info: FieldInfo, field_shape: TypingTuple[int, ...]) -> Any:
            if field_info.fill_value_factory is not None:
                return field_info.fill_value_factory(field_shape, field_info.dtype)
            return field_info.fill_value

        # Use pre-computed field information for efficient value generation
        for field_info in field_infos:
            if field_info.field_type == "jax_primitive_descriptor":
                field_shape = shape + field_info.intrinsic_shape
                default_values[field_info.name] = jnp.full(
                    field_shape,
                    resolve_fill_value(field_info, field_shape),
                    dtype=field_info.dtype,
                )
            elif field_info.field_type == "jax_dtype_descriptor":
                field_shape = shape + field_info.intrinsic_shape
                default_values[field_info.name] = jnp.full(
                    field_shape,
                    resolve_fill_value(field_info, field_shape),
                    dtype=field_info.dtype,
                )
            elif field_info.field_type == "nested_class_descriptor":
                field_shape = shape + field_info.intrinsic_shape
                default_values[field_info.name] = field_info.nested_class_type.default(
                    shape=field_shape
                )
            elif field_info.field_type == "nested_class_direct":
                default_values[field_info.name] = field_info.nested_class_type.default(shape=shape)
        return cls(**default_values)

    return default
