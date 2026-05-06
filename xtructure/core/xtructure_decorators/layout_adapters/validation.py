import functools
from typing import Any, Type, TypeVar

import jax.numpy as jnp

from xtructure.core.layout import get_type_layout
from xtructure.core.layout.types import AdapterFieldPlan

T = TypeVar("T")


def add_runtime_validation(cls: Type[T], *, enabled: bool) -> Type[T]:
    """Inject a post-init validator that checks dtype and trailing shape."""
    type_layout = get_type_layout(cls)

    def _validate_array_field(field_plan: AdapterFieldPlan, value: Any):
        try:
            array = jnp.asarray(value)
        except TypeError as err:
            raise TypeError(
                f"{cls.__name__}.{field_plan.name} could not be coerced to a JAX array for validation."
            ) from err

        try:
            expected_dtype = jnp.dtype(field_plan.declared_dtype)
        except TypeError as err:
            raise TypeError(
                f"{cls.__name__}.{field_plan.name} has unsupported dtype "
                f"{field_plan.declared_dtype}."
            ) from err

        if array.dtype != expected_dtype:
            raise TypeError(
                f"{cls.__name__}.{field_plan.name} expected dtype {expected_dtype}, "
                f"got {array.dtype}."
            )

        intrinsic_shape = tuple(field_plan.intrinsic_shape)
        if intrinsic_shape:
            intrinsic_len = len(intrinsic_shape)
            if array.ndim < intrinsic_len:
                raise ValueError(
                    f"{cls.__name__}.{field_plan.name} expected trailing shape {intrinsic_shape}, "
                    f"but array has shape {array.shape}."
                )
            trailing_shape = array.shape[-intrinsic_len:]
            if tuple(trailing_shape) != intrinsic_shape:
                raise ValueError(
                    f"{cls.__name__}.{field_plan.name} expected trailing shape {intrinsic_shape}, "
                    f"but got {trailing_shape}."
                )

    def check_invariants(self):
        for field_plan in type_layout.adapter_field_plans:
            value = getattr(self, field_plan.name)

            if field_plan.validator is not None:
                field_plan.validator(value)

            if field_plan.field_kind == "nested":
                nested_type = field_plan.nested_type
                if not isinstance(value, nested_type):
                    raise TypeError(
                        f"{cls.__name__}.{field_plan.name} expected instance of "
                        f"{nested_type.__name__}, got {type(value).__name__}."
                    )
                continue
            _validate_array_field(field_plan, value)

    setattr(cls, "check_invariants", check_invariants)

    if not enabled:
        return cls

    original_init = cls.__init__

    @functools.wraps(original_init)
    def _validated_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.check_invariants()

    setattr(cls, "__init__", _validated_init)
    return cls
