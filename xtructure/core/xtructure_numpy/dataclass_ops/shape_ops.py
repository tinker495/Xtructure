"""Shape manipulation helpers for xtructure dataclasses."""

from __future__ import annotations

from typing import TypeVar, Union

import jax
import jax.numpy as jnp

T = TypeVar("T")


def reshape(dataclass_instance: T, new_shape: tuple[int, ...]) -> T:
    """Delegate reshape to the dataclass instance."""
    return dataclass_instance.reshape(new_shape)


def flatten(dataclass_instance: T) -> T:
    """Delegate flatten to the dataclass instance."""
    return dataclass_instance.flatten()


def transpose(dataclass_instance: T, axes: Union[tuple[int, ...], None] = None) -> T:
    """Transpose batch dimensions of every field."""
    batch_shape = dataclass_instance.shape.batch
    if isinstance(batch_shape, int):
        batch_ndim = 1
    else:
        batch_ndim = len(batch_shape)

    if axes is None:
        axes = tuple(range(batch_ndim - 1, -1, -1))

    def transpose_batch_only(field: jnp.ndarray) -> jnp.ndarray:
        field_ndim = field.ndim
        if field_ndim <= batch_ndim:
            return jnp.transpose(field, axes=axes)
        full_axes = list(axes) + list(range(batch_ndim, field_ndim))
        return jnp.transpose(field, axes=full_axes)

    return jax.tree_util.tree_map(transpose_batch_only, dataclass_instance)


def swapaxes(dataclass_instance: T, axis1: int, axis2: int) -> T:
    """Swap two batch axes."""
    batch_shape = dataclass_instance.shape.batch
    if isinstance(batch_shape, int):
        batch_ndim = 1
    else:
        batch_ndim = len(batch_shape)

    def normalize_axis(axis: int) -> int:
        return axis if axis >= 0 else batch_ndim + axis

    axis1_norm = normalize_axis(axis1)
    axis2_norm = normalize_axis(axis2)

    if axis1_norm < 0 or axis1_norm >= batch_ndim:
        raise ValueError(f"Axis {axis1} is out of bounds for batch dimensions {batch_shape}")
    if axis2_norm < 0 or axis2_norm >= batch_ndim:
        raise ValueError(f"Axis {axis2} is out of bounds for batch dimensions {batch_shape}")

    def swap_batch_axes_only(field: jnp.ndarray) -> jnp.ndarray:
        field_ndim = field.ndim
        if field_ndim <= batch_ndim:
            return jnp.swapaxes(field, axis1_norm, axis2_norm)
        return jnp.swapaxes(field, axis1_norm, axis2_norm)

    return jax.tree_util.tree_map(swap_batch_axes_only, dataclass_instance)


def expand_dims(dataclass_instance: T, axis: int) -> T:
    """Insert a new axis into every field."""
    return jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=axis), dataclass_instance)


def squeeze(dataclass_instance: T, axis: Union[int, tuple[int, ...], None] = None) -> T:
    """Remove axes of length one from every field."""
    return jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=axis), dataclass_instance)


def repeat(dataclass_instance: T, repeats: Union[int, jnp.ndarray], axis: int | None = None) -> T:
    """Repeat elements along the given axis."""
    return jax.tree_util.tree_map(lambda x: jnp.repeat(x, repeats, axis=axis), dataclass_instance)
