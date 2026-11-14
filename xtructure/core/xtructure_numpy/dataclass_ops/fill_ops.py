"""Fill-based helpers for xtructure dataclasses."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ...protocol import Xtructurable


def full_like(dataclass_instance: Xtructurable, fill_value: Any) -> Xtructurable:
    """Return a dataclass filled with `fill_value`."""
    return dataclass_instance._map_data(lambda x: jnp.full_like(x, fill_value))


def zeros_like(dataclass_instance: Xtructurable) -> Xtructurable:
    """Return a dataclass filled with zeros."""
    return dataclass_instance._map_data(jnp.zeros_like)


def ones_like(dataclass_instance: Xtructurable) -> Xtructurable:
    """Return a dataclass filled with ones."""
    return dataclass_instance._map_data(jnp.ones_like)
