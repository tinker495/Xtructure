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


def moveaxis(
    a: T,
    source: Union[int, Sequence[int]],
    destination: Union[int, Sequence[int]],
) -> T:
    """Move axes of an array to new positions."""
    return jax.tree_util.tree_map(
        lambda x: jnp.moveaxis(x, source=source, destination=destination), a
    )


def broadcast_to(array: T, shape: Sequence[int]) -> T:
    """Broadcast an array to a new shape."""
    return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, shape=shape), array)


def broadcast_arrays(*args: Any) -> list[Any]:
    """
    Broadcasts any number of arrays against each other.
    Returns a list of broadcasted arrays (structures).
    """
    if not args:
        return []

    # Use tree_map to broadcast leaves against each other.
    # jnp.broadcast_arrays(*leaves) returns a list of broadcasted leaves.
    # Because tree_map expects a single return value per leaf call to maintain structure,
    # and jnp.broadcast_arrays returns a list/tuple, we get a Structure of Lists.
    
    # We assume all args have the same structure (enforced by tree_map generally).
    broadcasted_leaves_struct = jax.tree_util.tree_map(
        lambda *xs: jnp.broadcast_arrays(*xs), *args
    )
    
    # Check strictness: tree_map might be lenient. broadcast_arrays implies checked structures.
    # If structures mismatch, tree_map raises error (usually).
    
    # Now valid: broadcasted_leaves_struct is a Pytree where leaves are Lists of Arrays.
    # We want a List of Pytrees (structures).
    
    # We know the outer structure (treedef of args[0]).
    outer_treedef = jax.tree_util.tree_structure(args[0])
    
    # We want to pull the List (length = len(args)) out.
    # We can use jax.tree_util.tree_transpose.
    # The 'inner' structure is the List structure.
    inner_treedef = jax.tree_util.tree_structure([0] * len(args))
    
    return jax.tree_util.tree_transpose(outer_treedef, inner_treedef, broadcasted_leaves_struct)


def atleast_1d(*arys: Any) -> Any:
    """Convert inputs to arrays with at least one dimension."""
    results = [jax.tree_util.tree_map(jnp.atleast_1d, arr) for arr in arys]
    if len(arys) == 1:
        return results[0]
    return results


def atleast_2d(*arys: Any) -> Any:
    """Convert inputs to arrays with at least two dimensions."""
    results = [jax.tree_util.tree_map(jnp.atleast_2d, arr) for arr in arys]
    if len(arys) == 1:
        return results[0]
    return results


def atleast_3d(*arys: Any) -> Any:
    """Convert inputs to arrays with at least three dimensions."""
    results = [jax.tree_util.tree_map(jnp.atleast_3d, arr) for arr in arys]
    if len(arys) == 1:
        return results[0]
    return results
