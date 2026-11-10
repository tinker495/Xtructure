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
            field_shape = shape + field_plan.storage_intrinsic_shape
            if field_plan.nested_type is not None:
                default_values[field_plan.name] = field_plan.nested_type.default(shape=field_shape)
            else:
                default_values[field_plan.name] = jnp.full(
                    field_shape,
                    resolve_fill_value(field_plan, field_shape),
                    dtype=field_plan.declared_dtype,
                )
        return cls(**default_values)

    return default
