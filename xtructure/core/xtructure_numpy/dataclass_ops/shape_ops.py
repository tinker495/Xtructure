"""Shape manipulation helpers for xtructure dataclasses."""

from __future__ import annotations

from typing import TypeVar, Union

import jax.numpy as jnp

from ...protocol import Xtructurable

T = TypeVar("T", bound=Xtructurable)


def expand_dims(dataclass_instance: T, axis: int) -> T:
    """Insert a new axis into every field."""
    return dataclass_instance._map_data(lambda x: jnp.expand_dims(x, axis=axis))


def squeeze(dataclass_instance: T, axis: Union[int, tuple[int, ...], None] = None) -> T:
    """Remove axes of length one from every field."""
    return dataclass_instance._map_data(lambda x: jnp.squeeze(x, axis=axis))


def repeat(dataclass_instance: T, repeats: Union[int, jnp.ndarray], axis: int | None = None) -> T:
    """Repeat elements along the given axis."""
    return dataclass_instance._map_data(lambda x: jnp.repeat(x, repeats, axis=axis))
