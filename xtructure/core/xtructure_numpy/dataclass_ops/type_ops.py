"""Type system helpers for xtructure dataclasses."""

from __future__ import annotations

from typing import Any, TypeVar

import jax
import jax.numpy as jnp

T = TypeVar("T")


def astype(
    x: T,
    dtype: Any,
    copy: bool = False,
    device: Any = None,
) -> T:
    """Copy of the array, cast to a specified type."""
    # We map jnp.astype over the tree.
    # Note: jnp.astype behavior on copy/device might be backend specific,
    # but we pass it through.
    return jax.tree_util.tree_map(
        lambda leaf: jnp.astype(leaf, dtype, copy=copy, device=device), x
    )


def result_type(*args: Any) -> Any:
    """
    Returns the type that results from applying the NumPy type promotion rules
    to the arguments.

    If the arguments are structures, returns a structure of dtypes.
    """
    # result_type logic:
    # We want to map over all args simultaneously.
    # If args are [structA, structB], we map result_type(fieldA, fieldB).
    return jax.tree_util.tree_map(jnp.result_type, *args)


def can_cast(from_: Any, to: Any, casting: str = "safe") -> bool:
    """
    Returns True if cast between data types can occur according to the casting rule.

    If inputs are structures, returns True only if ALL fields can be cast.
    """
    # If 'to' is a single dtype, we broadcast it to the structure of 'from_'?
    # Or strict mapping?

    # helper:
    def _leaf_can_cast(f, t):
        # Get dtype from array if it's an array, otherwise use the value directly
        f_dtype = f.dtype if hasattr(f, "dtype") else f
        t_dtype = t.dtype if hasattr(t, "dtype") else t
        return jnp.can_cast(f_dtype, t_dtype, casting=casting)

    # We need to handle cases where 'to' is not a structure (scalar dtype)
    # vs 'to' is a matching structure.

    # jax.tree_util.tree_map supports broadcasting of the shorter tree prefix??
    # No, usually exact match or one is a leaf.

    # If 'to' is a dtype object (not a structure instance), tree_map might treat it as a leaf
    # if it's not a registered pytree. numpy dtypes are not pytrees.

    try:
        tree_result = jax.tree_util.tree_map(_leaf_can_cast, from_, to)
    except (ValueError, TypeError):
        # Mismatch structures. Assume 'to' is a scalar dtype to broadcast.
        to_broadcast = jax.tree_util.tree_map(lambda _: to, from_)
        tree_result = jax.tree_util.tree_map(_leaf_can_cast, from_, to_broadcast)

    return jax.tree_util.tree_reduce(lambda x, y: x and y, tree_result, True)
