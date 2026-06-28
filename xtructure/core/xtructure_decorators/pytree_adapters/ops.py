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
            # Different type -> NotImplemented
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
