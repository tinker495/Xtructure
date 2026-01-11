"""Batch-oriented utilities for dataclass array operations."""

from __future__ import annotations

from typing import List, TypeVar, Union

import jax
import jax.numpy as jnp

from xtructure.core.structuredtype import StructuredType

T = TypeVar("T")


def _normalize_pad_width(
    pad_width: Union[int, tuple[int, ...], tuple[tuple[int, int], ...]], ndim: int
):
    """Normalize pad_width to (before, after) tuples for every batch axis."""
    if isinstance(pad_width, int):
        return [(pad_width, pad_width)] * ndim
    if isinstance(pad_width, (list, tuple)):
        if len(pad_width) == 0:
            raise ValueError("pad_width cannot be empty")

        if len(pad_width) == 2 and all(isinstance(x, (int, float)) for x in pad_width):
            result = [(int(pad_width[0]), int(pad_width[1]))]
            result.extend([(0, 0)] * (ndim - 1))
            return result

        if all(isinstance(x, (list, tuple)) and len(x) == 2 for x in pad_width):
            if len(pad_width) != ndim:
                raise ValueError(
                    f"pad_width length {len(pad_width)} must match number of batch dimensions {ndim}"
                )
            return [(int(before), int(after)) for before, after in pad_width]

        if len(pad_width) != ndim:
            raise ValueError(
                f"pad_width length {len(pad_width)} must match number of batch dimensions {ndim}"
            )
        return [(int(x), int(x)) for x in pad_width]
    raise ValueError("pad_width must be int, sequence of int, or sequence of pairs")


def concat(dataclasses: List[T], axis: int = 0) -> T:
    """Concatenate matching dataclasses along the provided axis."""
    if not dataclasses:
        raise ValueError("Cannot concatenate empty list of dataclasses")

    if len(dataclasses) == 1:
        return dataclasses[0]

    first_type = type(dataclasses[0])
    if not all(isinstance(dc, first_type) for dc in dataclasses):
        raise ValueError("All dataclasses must be of the same type")

    first_structured_type = dataclasses[0].structured_type
    if not all(dc.structured_type == first_structured_type for dc in dataclasses):
        raise ValueError("All dataclasses must have the same structured type")

    if first_structured_type == StructuredType.SINGLE:
        return stack(dataclasses, axis=axis)

    if first_structured_type == StructuredType.BATCHED:
        first_batch_shape = dataclasses[0].shape.batch
        concat_axis_adjusted = axis if axis >= 0 else len(first_batch_shape) + axis

        if concat_axis_adjusted >= len(first_batch_shape):
            raise ValueError(
                f"Concatenation axis {axis} is out of bounds for batch shape {first_batch_shape}"
            )

        for dc in dataclasses[1:]:
            batch_shape = dc.shape.batch
            if len(batch_shape) != len(first_batch_shape):
                raise ValueError(
                    f"Incompatible batch dimensions: {first_batch_shape} vs {batch_shape}"
                )

            for i, (dim1, dim2) in enumerate(zip(first_batch_shape, batch_shape)):
                if i != concat_axis_adjusted and dim1 != dim2:
                    raise ValueError(f"Incompatible batch dimensions at axis {i}: {dim1} vs {dim2}")

        return jax.tree_util.tree_map(
            lambda *arrays: jnp.concatenate(arrays, axis=axis), *dataclasses
        )

    raise ValueError(f"Concatenation not supported for structured type: {first_structured_type}")


def pad(
    dataclass_instance: T,
    pad_width: Union[int, tuple[int, ...], tuple[tuple[int, int], ...]],
    mode: str = "constant",
    **kwargs,
) -> T:
    """Pad xtructure dataclasses using a jnp.pad compatible interface."""
    structured_type = dataclass_instance.structured_type

    if structured_type == StructuredType.SINGLE:
        normalized_pad_width = _normalize_pad_width(pad_width, 1)
        if any(before < 0 or after < 0 for before, after in normalized_pad_width):
            raise ValueError("pad_width entries must be non-negative")

        if all(before == 0 and after == 0 for before, after in normalized_pad_width):
            return dataclass_instance

        pad_before, pad_after = normalized_pad_width[0]
        target_size = 1 + pad_before + pad_after

        if mode == "constant" and "constant_values" not in kwargs:
            result = type(dataclass_instance).default((target_size,))
            return result.at[pad_before].set(dataclass_instance)

        expanded = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), dataclass_instance)
        batch_ndim = 1
        pad_width_spec = normalized_pad_width

        return jax.tree_util.tree_map(
            lambda x: jnp.pad(
                x, pad_width_spec + [(0, 0)] * (x.ndim - batch_ndim), mode=mode, **kwargs
            ),
            expanded,
        )

    if structured_type == StructuredType.BATCHED:
        batch_shape = dataclass_instance.shape.batch
        batch_ndim = len(batch_shape)
        normalized_pad_width = _normalize_pad_width(pad_width, batch_ndim)
        if any(before < 0 or after < 0 for before, after in normalized_pad_width):
            raise ValueError("pad_width entries must be non-negative")

        if all(before == 0 and after == 0 for before, after in normalized_pad_width):
            return dataclass_instance

        if mode == "constant" and "constant_values" not in kwargs:
            target_shape = tuple(
                dim + before + after
                for dim, (before, after) in zip(batch_shape, normalized_pad_width)
            )
            insert_slices = tuple(
                slice(before, before + dim)
                for dim, (before, after) in zip(batch_shape, normalized_pad_width)
            )
            result = type(dataclass_instance).default(target_shape)
            return result.at[insert_slices].set(dataclass_instance)

        pad_width_spec = normalized_pad_width
        return jax.tree_util.tree_map(
            lambda x: jnp.pad(
                x, pad_width_spec + [(0, 0)] * (x.ndim - batch_ndim), mode=mode, **kwargs
            ),
            dataclass_instance,
        )

    raise ValueError(f"Padding not supported for structured type: {structured_type}")


def stack(dataclasses: List[T], axis: int = 0) -> T:
    """Stack dataclasses along a new axis."""
    if not dataclasses:
        raise ValueError("Cannot stack empty list of dataclasses")

    if len(dataclasses) == 1:
        return jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=axis), dataclasses[0])

    first_type = type(dataclasses[0])
    if not all(isinstance(dc, first_type) for dc in dataclasses):
        raise ValueError("All dataclasses must be of the same type")

    first_structured_type = dataclasses[0].structured_type
    if not all(dc.structured_type == first_structured_type for dc in dataclasses):
        raise ValueError("All dataclasses must have the same structured type")

    if first_structured_type == StructuredType.BATCHED:
        first_batch_shape = dataclasses[0].shape.batch
        for dc in dataclasses[1:]:
            if dc.shape.batch != first_batch_shape:
                raise ValueError(
                    f"All dataclasses must have the same batch shape: {first_batch_shape} vs {dc.shape.batch}"
                )

    return jax.tree_util.tree_map(lambda *arrays: jnp.stack(arrays, axis=axis), *dataclasses)


def take(dataclass_instance: T, indices: jnp.ndarray, axis: int = 0) -> T:
    """Take elements along an axis from every field."""
    return jax.tree_util.tree_map(lambda x: jnp.take(x, indices, axis=axis), dataclass_instance)


def take_along_axis(dataclass_instance: T, indices: jnp.ndarray, axis: int) -> T:
    """Gather values along a given axis for each field."""
    indices_array = jnp.asarray(indices)

    def _reorder_leaf(leaf: jnp.ndarray) -> jnp.ndarray:
        axis_in_leaf = axis if axis >= 0 else axis + leaf.ndim
        if axis_in_leaf < 0 or axis_in_leaf >= leaf.ndim:
            raise ValueError(f"`axis` {axis} is out of bounds for array with ndim {leaf.ndim}.")

        if indices_array.ndim > leaf.ndim:
            raise ValueError(
                "`indices` must not have more dimensions than the target field. "
                f"indices.ndim={indices_array.ndim}, field.ndim={leaf.ndim}."
            )

        expanded = indices_array
        for _ in range(leaf.ndim - expanded.ndim):
            expanded = expanded[..., None]

        target_shape = list(leaf.shape)
        target_shape[axis_in_leaf] = expanded.shape[axis_in_leaf]
        try:
            expanded = jnp.broadcast_to(expanded, tuple(target_shape))
        except ValueError as err:
            raise ValueError(
                "`indices` shape cannot be broadcast to match field shape "
                f"{leaf.shape} outside axis {axis}. Original indices shape: {indices_array.shape}."
            ) from err

        return jnp.take_along_axis(leaf, expanded, axis=axis_in_leaf)

    return jax.tree_util.tree_map(_reorder_leaf, dataclass_instance)


def tile(dataclass_instance: T, reps: Union[int, tuple[int, ...]]) -> T:
    """Tile every field of the dataclass."""
    if isinstance(reps, int):
        reps = (reps,)
    return jax.tree_util.tree_map(lambda x: jnp.tile(x, reps), dataclass_instance)


def split(
    dataclass_instance: T, indices_or_sections: Union[int, jnp.ndarray], axis: int = 0
) -> List[T]:
    """Split a dataclass along the given axis."""
    leaves, treedef = jax.tree_util.tree_flatten(dataclass_instance)
    split_leaves = [jnp.split(leaf, indices_or_sections, axis=axis) for leaf in leaves]

    if not split_leaves:
        return []

    num_splits = len(split_leaves[0])
    result_dataclasses: List[T] = []
    for i in range(num_splits):
        new_leaves = [sl[i] for sl in split_leaves]
        result_dataclasses.append(jax.tree_util.tree_unflatten(treedef, new_leaves))

    return result_dataclasses
