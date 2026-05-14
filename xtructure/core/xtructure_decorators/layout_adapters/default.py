from typing import Any, Callable
from typing import Tuple as TypingTuple
from typing import Type, TypeVar

import jax.numpy as jnp

from xtructure.core.layout import get_type_layout
from xtructure.core.layout.types import AdapterFieldPlan

T = TypeVar("T")


def add_default_method(cls: Type[T]) -> Type[T]:
    setattr(cls, "default", _create_default_method(cls))
    return cls


def _create_default_method(cls_to_modify: Type[T]) -> Callable[..., T]:
    type_layout = get_type_layout(cls_to_modify)
    field_plans = type_layout.adapter_field_plans

    for field_plan in field_plans:
        if field_plan.field_kind == "nested":
            nested_class_type = field_plan.nested_type
            if not hasattr(nested_class_type, "default"):
                raise TypeError(
                    f"Error during method creation for '{cls_to_modify.__name__}': "
                    f"Nested field '{field_plan.name}' (type '{nested_class_type.__name__}' "
                    f"via FieldDescriptor.dtype) does not have a .default() method. "
                    f"Ensure it's an @xtructure_data class."
                )
        elif not (
            isinstance(field_plan.declared_dtype, jnp.dtype) or field_plan.is_primitive_jax_dtype
        ):
            dtype = field_plan.declared_dtype
            raise TypeError(
                f"Error during method creation for '{cls_to_modify.__name__}': "
                f"Field '{field_plan.name}' uses FieldDescriptor with an unsupported "
                f".dtype attribute: '{dtype}' "
                f"(type: {type(dtype).__name__}). "
                f"Expected a JAX primitive type/class (like jnp.int32 or "
                f"jnp.dtype('int32')), or an @xtructure_data class type (like Parent)."
            )

    @classmethod
    def default(cls: Type[T], shape: TypingTuple[int, ...] = ()) -> T:
        default_values: dict[str, Any] = {}

        def resolve_fill_value(
            field_plan: AdapterFieldPlan, field_shape: TypingTuple[int, ...]
        ) -> Any:
            if field_plan.fill_value_factory is not None:
                return field_plan.fill_value_factory(field_shape, field_plan.declared_dtype)
            return field_plan.fill_value

        for field_plan in field_plans:
            field_shape = shape + field_plan.intrinsic_shape
            if field_plan.field_kind == "primitive":
                default_values[field_plan.name] = jnp.full(
                    field_shape,
                    resolve_fill_value(field_plan, field_shape),
                    dtype=field_plan.declared_dtype,
                )
            elif field_plan.field_kind == "nested":
                default_values[field_plan.name] = field_plan.nested_type.default(shape=field_shape)
        return cls(**default_values)

    return default
