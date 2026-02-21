"""Instance method factory for xtructure dataclasses.

This module provides utilities to automatically generate instance methods
from xnp (xtructure_numpy) functions. When a new xnp function is added,
simply add it to the appropriate list below to make it available as an
instance method on all xtructure dataclasses.
"""

from functools import wraps
from typing import Callable, Type, TypeVar

T = TypeVar("T")


def _make_method(func: Callable) -> Callable:
    """Create an instance method from an xnp function.

    The xnp function should accept the dataclass instance as its first argument.
    """

    @wraps(func)
    def method(self, *args, **kwargs):
        return func(self, *args, **kwargs)

    return method


def add_xnp_instance_methods(cls: Type[T]) -> Type[T]:
    """Add instance methods that delegate to xnp functions.

    This decorator adds methods like .roll(), .flip(), .astype() etc.
    that call the corresponding xnp.roll(), xnp.flip(), xnp.astype() functions.

    To add a new method when a new xnp function is created:
    1. Add the function name and import to _XNP_METHOD_REGISTRY below
    2. The method will automatically be available on all xtructure dataclasses
    """
    # Lazy import to avoid circular dependencies
    from xtructure.core.xtructure_numpy.dataclass_ops import (
        batch_ops,
        comparison_ops,
        shape_ops,
        spatial_ops,
        type_ops,
    )

    # Registry of methods to add: (method_name, function_reference)
    # Only methods where the first argument is the dataclass instance
    _XNP_METHOD_REGISTRY = {
        # Core shape operations
        "reshape": shape_ops.reshape,
        "flatten": shape_ops.flatten,
        "transpose": shape_ops.transpose,
        "swapaxes": shape_ops.swapaxes,
        "moveaxis": shape_ops.moveaxis,
        "squeeze": shape_ops.squeeze,
        "expand_dims": shape_ops.expand_dims,
        "broadcast_to": shape_ops.broadcast_to,
        # Spatial operations
        "roll": spatial_ops.roll,
        "flip": spatial_ops.flip,
        "rot90": spatial_ops.rot90,
        # Type operations
        "astype": type_ops.astype,
        # Stacking operations
        "vstack": batch_ops.vstack,
        "hstack": batch_ops.hstack,
        "dstack": batch_ops.dstack,
        "column_stack": batch_ops.column_stack,
        "block": batch_ops.block,
        # Pad operation
        "pad": batch_ops.pad,
        # Comparison operations (return structure of bools)
        "equal": comparison_ops.equal,
        "not_equal": comparison_ops.not_equal,
        "isclose": comparison_ops.isclose,
        # Comparison operations (return scalar bool)
        "allclose": comparison_ops.allclose,
    }

    for method_name, func in _XNP_METHOD_REGISTRY.items():
        # Don't override existing methods (e.g., custom implementations)
        if not hasattr(cls, method_name):
            setattr(cls, method_name, _make_method(func))

    return cls
