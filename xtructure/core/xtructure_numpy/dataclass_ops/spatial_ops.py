"""Spatial transformation helpers for xtructure dataclasses."""

from __future__ import annotations

from typing import Sequence, TypeVar, Union

import jax
import jax.numpy as jnp

T = TypeVar("T")


def roll(
    a: T,
    shift: Union[int, Sequence[int]],
    axis: Union[int, Sequence[int], None] = None,
) -> T:
    """Roll array elements along a given axis."""
    return jax.tree_util.tree_map(lambda x: jnp.roll(x, shift, axis=axis), a)


def flip(
    m: T,
    axis: Union[int, Sequence[int], None] = None,
) -> T:
    """Reverse the order of elements in an array along the given axis."""
    return jax.tree_util.tree_map(lambda x: jnp.flip(x, axis=axis), m)


def rot90(
    m: T,
    k: int = 1,
    axes: tuple[int, int] = (0, 1),
) -> T:
    """Rotate an array by 90 degrees in the plane specified by axes."""
    return jax.tree_util.tree_map(lambda x: jnp.rot90(x, k=k, axes=axes), m)
