"""Fill-based helpers for xtructure dataclasses."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from ...xtructure_decorators import Xtructurable


def full_like(dataclass_instance: Xtructurable, fill_value: Any) -> Xtructurable:
    """Return a dataclass filled with `fill_value`."""
    return jax.tree_util.tree_map(lambda x: jnp.full_like(x, fill_value), dataclass_instance)


def zeros_like(dataclass_instance: Xtructurable) -> Xtructurable:
    """Return a dataclass filled with zeros."""
    return jax.tree_util.tree_map(jnp.zeros_like, dataclass_instance)


def ones_like(dataclass_instance: Xtructurable) -> Xtructurable:
    """Return a dataclass filled with ones."""
    return jax.tree_util.tree_map(jnp.ones_like, dataclass_instance)
