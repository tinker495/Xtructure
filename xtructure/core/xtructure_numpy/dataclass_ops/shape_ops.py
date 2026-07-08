"""Shape manipulation helpers for xtructure dataclasses."""

from __future__ import annotations

from typing import TypeVar, Union

import jax
import jax.numpy as jnp

T = TypeVar("T")


def expand_dims(dataclass_instance: T, axis: int) -> T:
    """Insert a new axis into every field."""
    return jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=axis), dataclass_instance)


def squeeze(dataclass_instance: T, axis: Union[int, tuple[int, ...], None] = None) -> T:
    """Remove axes of length one from every field."""
    return jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=axis), dataclass_instance)


def repeat(dataclass_instance: T, repeats: Union[int, jnp.ndarray], axis: int | None = None) -> T:
    """Repeat elements along the given axis."""
    return jax.tree_util.tree_map(lambda x: jnp.repeat(x, repeats, axis=axis), dataclass_instance)
