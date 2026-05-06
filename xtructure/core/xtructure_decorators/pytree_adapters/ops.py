from typing import Any, Type, TypeVar

import jax
import jax.numpy as jnp

T = TypeVar("T")


def add_comparison_operators(cls: Type[T]) -> Type[T]:
    """
    Adds custom __eq__ and __ne__ methods to the class.

    Semantics:
    - For two instances of the same xtructure dataclass type, comparisons are performed
      field-wise and then **reduced** to a single boolean:
        - `__eq__`: True iff *all* fields are equal (via `jnp.all(x == y)` per field).
        - `__ne__`: True iff *any* field is different (via `jnp.any(x != y)` per field).
    - If `other` is not the same type, returns `NotImplemented`.

    Note: these operators return a scalar boolean (JAX bool array / Python bool),
    not a dataclass of booleans.
    """

    def _xtructure_eq(self, other: Any) -> T:
        if not isinstance(other, self.__class__):
            # If comparing with a different type, one might return False
            # or NotImplemented. For element-wise comparison resulting in a
            # structure, raising an error or returning a structure of False
            # might be alternatives. JAX's __eq__ on arrays would raise
            # an error or broadcast if shapes are incompatible.
            # Here, we'll opt for a structure of False values if types don't match
            # or if users expect a single boolean, this override might be surprising.
            # A more robust approach for general pytrees might involve checking
            # tree structure compatibility.
            # For now, returning NotImplemented is safest if 'other' isn't the same type.
            return NotImplemented

        # Element-wise comparison for each field
        tree_equal = jax.tree_util.tree_map(lambda x, y: jnp.all(x == y), self, other)
        return jax.tree_util.tree_reduce(jnp.logical_and, tree_equal)

    def _xtructure_ne(self, other: Any) -> T:
        if not isinstance(other, self.__class__):
            return NotImplemented

        # Element-wise comparison for each field
        tree_equal = jax.tree_util.tree_map(lambda x, y: jnp.any(x != y), self, other)
        return jax.tree_util.tree_reduce(jnp.logical_or, tree_equal)

    setattr(cls, "__eq__", _xtructure_eq)
    setattr(cls, "__ne__", _xtructure_ne)

    return cls
