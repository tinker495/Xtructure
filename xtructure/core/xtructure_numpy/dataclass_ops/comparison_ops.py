"""Comparison helpers for xtructure dataclasses."""

from __future__ import annotations

from typing import Any, TypeVar

import jax
import jax.numpy as jnp

T = TypeVar("T")


def equal(x: T, y: Any) -> T:
    """Return (x == y) element-wise."""
    return jax.tree_util.tree_map(lambda a, b: jnp.equal(a, b), x, y)


def not_equal(x: T, y: Any) -> T:
    """Return (x != y) element-wise."""
    return jax.tree_util.tree_map(lambda a, b: jnp.not_equal(a, b), x, y)


def isclose(
    a: T,
    b: Any,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> T:
    """Returns a boolean array where two arrays are element-wise equal within a tolerance."""
    return jax.tree_util.tree_map(
        lambda x, y: jnp.isclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan),
        a,
        b,
    )


def allclose(
    a: T,
    b: Any,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool | jnp.ndarray:
    """Returns True if two arrays are element-wise equal within a tolerance."""
    # First apply isclose element-wise
    # Then reduce using logical_and across the entire tree
    # tree_reduce applies function to leaves two at a time.
    # We first reduce each leaf to a single boolean (since isclose returns an array structure)
    # Actually, jnp.allclose returns a single scalar boolean for arrays.
    # So we should map jnp.allclose over leaves?
    # No, jnp.allclose(arr1, arr2) returns True/False.
    # If we map it, we get a structure of True/False scalars.
    # Then we reduce that structure with logical_and.

    # Wait, strict alignment means strictly following jnp signature.
    # jnp.allclose returns a scalar boolean (or boolean array scalar).

    # Let's map jnp.allclose per leaf?
    # But structural allclose implies ALL fields are close.

    def _leaf_allclose(x, y):
        # We must allow for potential broadcasting inside jax if shapes match
        return jnp.allclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)

    tree_all_close = jax.tree_util.tree_map(_leaf_allclose, a, b)
    return jax.tree_util.tree_reduce(jnp.logical_and, tree_all_close, True)
