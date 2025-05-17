import inspect
from typing import Any, Callable, Dict
from typing import Tuple as TypingTuple
from typing import Type, TypeVar

import jax.numpy as jnp

from xtructure.core.field_descriptors import FieldDescriptor

T = TypeVar("T")


def is_potentially_xtructure_class(annotation_obj: Any) -> bool:
    return inspect.isclass(annotation_obj)


def add_auto_default_method_if_needed(cls: Type[T]) -> Type[T]:
    if hasattr(cls, "default"):
        # User has provided a custom default method, so we don't overwrite it.
        return cls

    annotations = getattr(cls, "__annotations__", {})

    # Case 1: No annotations and no actual fields (e.g. `class A: pass`)
    # __dataclass_fields__ is populated by chex.dataclass() which is called *before* this function.
    dataclass_fields = getattr(cls, "__dataclass_fields__", {})
    if not annotations and not dataclass_fields:
        # Safe to generate a simple default() that calls cls()
        setattr(cls, "default", _create_default_method(cls))
        return cls

    # Case 2: Has annotations. Check if all are suitable.
    if annotations:  # Only proceed if there are annotations to check
        for field_name, ann_obj in annotations.items():
            is_fd = isinstance(ann_obj, FieldDescriptor)
            is_potential_nested = is_potentially_xtructure_class(ann_obj)

            if not is_fd and not is_potential_nested:
                # Found an annotation that is not a FieldDescriptor and not a class.
                # Auto-generation cannot handle this. Do not attach auto_default.
                # The assertion in xtructure_data will provide a detailed error.
                return cls

        # All annotations are suitable (FieldDescriptor or a class type).
        setattr(cls, "default", _create_default_method(cls))
        return cls

    # Case 3: No annotations, but has dataclass fields
    # (e.g. `my_field = 1` or `my_field: int` if not in __annotations__ somehow)
    # In this scenario, we can't auto-generate based on FieldDescriptors.
    # The class must provide its own default method.
    # So, we don't attach anything, and the main assertion in xtructure_data will trigger.
    return cls


def _create_default_method(cls_to_modify: Type[T]) -> Callable[..., T]:
    @classmethod
    def auto_default(cls: Type[T], shape: TypingTuple[int, ...] = ()) -> T:
        default_values: Dict[str, Any] = {}
        annotations = getattr(cls, "__annotations__", {})

        for field_name, annotation_obj in annotations.items():
            if isinstance(annotation_obj, FieldDescriptor):
                descriptor = annotation_obj
                dtype_of_field_descriptor = descriptor.dtype

                if is_potentially_xtructure_class(dtype_of_field_descriptor):
                    # dtype_of_field_descriptor is a CLASS (e.g. <class 'numpy.int32'> or <class '__main__.Parent'>)
                    # Differentiate: is it a JAX primitive *class* or a user-defined xtructure *class*?
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
                        field_shape = shape + intrinsic_shape
                        default_values[field_name] = jnp.full(
                            field_shape,
                            descriptor.fill_value,
                            dtype=dtype_of_field_descriptor,  # Use the primitive class directly
                        )
                    else:
                        # It's a user-defined class like Parent. Use its .default() method.
                        nested_class_type = dtype_of_field_descriptor
                        if not hasattr(nested_class_type, "default"):
                            raise TypeError(
                                f"Runtime error in auto-generated .default() for '{cls.__name__}': "
                                f"Nested field '{field_name}' (type '{nested_class_type.__name__}' "
                                f"via FieldDescriptor.dtype) does not have a .default() method. "
                                f"Ensure it's an @xtructure_data class."
                            )
                        default_values[field_name] = nested_class_type.default(shape=shape)
                elif isinstance(dtype_of_field_descriptor, jnp.dtype):
                    # dtype_of_field_descriptor is a JAX dtype INSTANCE (e.g., jnp.dtype('int32')). Use jnp.full.
                    intrinsic_shape = (
                        descriptor.intrinsic_shape
                        if isinstance(descriptor.intrinsic_shape, tuple)
                        else (descriptor.intrinsic_shape,)
                    )
                    field_shape = shape + intrinsic_shape
                    default_values[field_name] = jnp.full(
                        field_shape, descriptor.fill_value, dtype=dtype_of_field_descriptor
                    )
                else:
                    # FieldDescriptor.dtype is neither a recognized class nor a jnp.dtype instance.
                    raise TypeError(
                        f"Runtime error in auto-generated .default() for '{cls.__name__}': "
                        f"Field '{field_name}' uses FieldDescriptor with an unsupported "
                        f".dtype attribute: '{dtype_of_field_descriptor}' "
                        f"(type: {type(dtype_of_field_descriptor).__name__}). "
                        f"Expected a JAX primitive type/class (like jnp.int32 or "
                        f"jnp.dtype('int32')), or an @xtructure_data class type (like Parent)."
                    )
            elif is_potentially_xtructure_class(annotation_obj):
                # annotation_obj is the type of the nested class directly (e.g., parent: Parent)
                nested_class_type = annotation_obj
                if not hasattr(nested_class_type, "default"):
                    raise TypeError(
                        f"Runtime error in auto-generated .default() for '{cls.__name__}': "
                        f"Nested field '{field_name}' of type '{nested_class_type.__name__}' "
                        f"does not have a .default() method. Ensure it's an @xtructure_data class."
                    )
                default_values[field_name] = nested_class_type.default(shape=shape)
            else:
                # This case should ideally be caught by the pre-flight check in _add_auto_default_method_if_needed
                # However, this runtime check is a safeguard.
                raise TypeError(
                    f"Runtime error in auto-generated .default() for '{cls.__name__}': "
                    f"Field '{field_name}' with annotation '{annotation_obj}' "
                    f"(type: {type(annotation_obj).__name__}) is not a FieldDescriptor "
                    f"or a compatible nested xtructure_data class."
                )

        # Handle fields that are part of the dataclass but not in annotations (e.g. field_name: type = default_value)
        # These fields should pick up their class-defined defaults automatically when cls(**default_values) is called,
        # as long as they are not required in the __init__ generated by dataclass.
        # If they are required (no default value provided in class def), and not in annotations,
        # cls(**default_values) will fail, which is correct behavior.
        return cls(**default_values)

    return auto_default
