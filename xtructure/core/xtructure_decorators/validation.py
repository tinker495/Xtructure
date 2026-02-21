import functools
from typing import Any, Type, TypeVar

import jax.numpy as jnp

from xtructure.core.field_descriptors import FieldDescriptor, get_field_descriptors
from xtructure.core.type_utils import is_xtructure_dataclass_type

T = TypeVar("T")


def add_runtime_validation(cls: Type[T], *, enabled: bool) -> Type[T]:
    """Inject a post-init validator that checks dtype and trailing shape."""
    field_descriptors = get_field_descriptors(cls)

    def _validate_array_field(field_name: str, descriptor: FieldDescriptor, value: Any):
        try:
            array = jnp.asarray(value)
        except TypeError as err:
            raise TypeError(
                f"{cls.__name__}.{field_name} could not be coerced to a JAX array for validation."
            ) from err

        try:
            expected_dtype = jnp.dtype(descriptor.dtype)
        except TypeError as err:
            raise TypeError(
                f"{cls.__name__}.{field_name} has unsupported dtype {descriptor.dtype}."
            ) from err

        if array.dtype != expected_dtype:
            raise TypeError(
                f"{cls.__name__}.{field_name} expected dtype {expected_dtype}, got {array.dtype}."
            )

        intrinsic_shape = tuple(descriptor.intrinsic_shape)
        if intrinsic_shape:
            intrinsic_len = len(intrinsic_shape)
            if array.ndim < intrinsic_len:
                raise ValueError(
                    f"{cls.__name__}.{field_name} expected trailing shape {intrinsic_shape}, "
                    f"but array has shape {array.shape}."
                )
            trailing_shape = array.shape[-intrinsic_len:]
            if tuple(trailing_shape) != intrinsic_shape:
                raise ValueError(
                    f"{cls.__name__}.{field_name} expected trailing shape {intrinsic_shape}, "
                    f"but got {trailing_shape}."
                )

    def check_invariants(self):
        for field_name, descriptor in field_descriptors.items():
            value = getattr(self, field_name)

            if descriptor.validator is not None:
                descriptor.validator(value)

            if is_xtructure_dataclass_type(descriptor.dtype):
                if not isinstance(value, descriptor.dtype):
                    raise TypeError(
                        f"{cls.__name__}.{field_name} expected instance of "
                        f"{descriptor.dtype.__name__}, got {type(value).__name__}."
                    )
                continue
            _validate_array_field(field_name, descriptor, value)

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
