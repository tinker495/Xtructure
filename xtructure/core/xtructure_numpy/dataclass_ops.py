"""Operations for concatenating and padding xtructure dataclasses.

This module provides operations that complement the existing structure utilities
in xtructure_decorators.structure_util, reusing existing methods where possible.
"""

from typing import Any, List, TypeVar, Union

import jax
import jax.numpy as jnp

from xtructure.core.structuredtype import StructuredType

from ..xtructure_decorators import Xtructurable

T = TypeVar("T")


def concat(dataclasses: List[T], axis: int = 0) -> T:
    """
    Concatenate a list of xtructure dataclasses along the specified axis.

    This function complements the existing reshape/flatten methods by providing
    concatenation functionality for combining multiple dataclass instances.

    Args:
        dataclasses: List of xtructure dataclass instances to concatenate
        axis: Axis along which to concatenate (default: 0)

    Returns:
        A new dataclass instance with concatenated data

    Raises:
        ValueError: If dataclasses list is empty or instances have incompatible structures
    """
    if not dataclasses:
        raise ValueError("Cannot concatenate empty list of dataclasses")

    if len(dataclasses) == 1:
        return dataclasses[0]

    # Verify all dataclasses are of the same type
    first_type = type(dataclasses[0])
    if not all(isinstance(dc, first_type) for dc in dataclasses):
        raise ValueError("All dataclasses must be of the same type")

    # Verify all have compatible structured types
    first_structured_type = dataclasses[0].structured_type
    if not all(dc.structured_type == first_structured_type for dc in dataclasses):
        raise ValueError("All dataclasses must have the same structured type")

    # For SINGLE structured type, convert to batched first
    if first_structured_type == StructuredType.SINGLE:
        # Convert each single instance to a batch of size 1
        batched_dataclasses = []
        for dc in dataclasses:
            # Create a batched version by adding a batch dimension
            batched_dc = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), dc)
            batched_dataclasses.append(batched_dc)

        # Concatenate the batched versions
        result = jax.tree_util.tree_map(
            lambda *arrays: jnp.concatenate(arrays, axis=axis), *batched_dataclasses
        )
        return result

    # For BATCHED structured type, concatenate directly
    elif first_structured_type == StructuredType.BATCHED:
        # Verify batch dimensions are compatible (all except the concatenation axis)
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

        # Concatenate along the specified axis
        result = jax.tree_util.tree_map(
            lambda *arrays: jnp.concatenate(arrays, axis=axis), *dataclasses
        )
        return result

    else:
        raise ValueError(
            f"Concatenation not supported for structured type: {first_structured_type}"
        )


def pad(
    dataclass_instance: T,
    pad_width: Union[int, tuple[int, ...], tuple[tuple[int, int], ...]],
    mode: str = "constant",
    **kwargs,
) -> T:
    """
    Pad an xtructure dataclass with specified padding widths.

    This function provides jnp.pad-compatible interface for padding dataclasses.
    It supports all jnp.pad padding modes and parameter formats.

    Args:
        dataclass_instance: The xtructure dataclass instance to pad
        pad_width: Padding width specification, following jnp.pad convention:
            - int: Same padding (before, after) for all axes
            - sequence of int: Padding for each axis (before, after)
            - sequence of pairs: (before, after) padding for each axis
        mode: Padding mode ('constant', 'edge', 'linear_ramp', 'maximum', 'mean', 'median',
        'minimum', 'reflect', 'symmetric', 'wrap'). See jnp.pad for more details.
        **kwargs: Additional arguments passed to jnp.pad (e.g., constant_values for 'constant' mode)

    Returns:
        A new dataclass instance with padded data

    Raises:
        ValueError: If pad_width is incompatible with dataclass structure
    """
    structured_type = dataclass_instance.structured_type

    def _normalize_pad_width(pad_width, ndim):
        """Normalize pad_width to list of (before, after) tuples for each axis."""
        if isinstance(pad_width, int):
            # Same padding for all axes
            return [(pad_width, pad_width)] * ndim
        elif isinstance(pad_width, (list, tuple)):
            if len(pad_width) == 0:
                raise ValueError("pad_width cannot be empty")

            # Check if it's a single (before, after) pair for the first axis
            if len(pad_width) == 2 and all(isinstance(x, (int, float)) for x in pad_width):
                # Single (before, after) pair for first axis, rest get (0, 0)
                result = [(int(pad_width[0]), int(pad_width[1]))]
                result.extend([(0, 0)] * (ndim - 1))
                return result

            # Check if it's a sequence of pairs
            if all(isinstance(x, (list, tuple)) and len(x) == 2 for x in pad_width):
                # Sequence of (before, after) pairs
                if len(pad_width) != ndim:
                    raise ValueError(
                        f"pad_width length {len(pad_width)} must match number of batch dimensions {ndim}"
                    )
                return [(int(before), int(after)) for before, after in pad_width]
            else:
                # Sequence of single values - treat as (before, after) for each axis
                if len(pad_width) != ndim:
                    raise ValueError(
                        f"pad_width length {len(pad_width)} must match number of batch dimensions {ndim}"
                    )
                return [(int(x), int(x)) for x in pad_width]
        else:
            raise ValueError("pad_width must be int, sequence of int, or sequence of pairs")

    if structured_type == StructuredType.SINGLE:
        # For SINGLE type, create a batch dimension and pad it
        if isinstance(pad_width, int):
            pad_before, pad_after = pad_width, pad_width
        elif isinstance(pad_width, (list, tuple)):
            if len(pad_width) == 1:
                if isinstance(pad_width[0], (list, tuple)) and len(pad_width[0]) == 2:
                    pad_before, pad_after = pad_width[0]
                else:
                    pad_before, pad_after = pad_width[0], pad_width[0]
            elif len(pad_width) == 2:
                pad_before, pad_after = pad_width
            else:
                raise ValueError(
                    "For SINGLE structured type, pad_width must specify padding for single dimension"
                )
        else:
            raise ValueError("Invalid pad_width format for SINGLE structured type")

        # Create batch dimension with padding
        total_size = 1 + pad_before + pad_after

        # For simple constant padding with default values, use tiling
        if mode == "constant" and kwargs.get("constant_values", 0) == 0:
            result = jax.tree_util.tree_map(
                lambda x: jnp.tile(jnp.expand_dims(x, axis=0), (total_size,) + (1,) * x.ndim),
                dataclass_instance,
            )
            return result
        else:
            # For other modes, expand dims and pad
            expanded = jax.tree_util.tree_map(
                lambda x: jnp.expand_dims(x, axis=0), dataclass_instance
            )
            pad_width_spec = [(pad_before, pad_after)]

            result = jax.tree_util.tree_map(
                lambda x: jnp.pad(x, pad_width_spec + [(0, 0)] * (x.ndim - 1), mode=mode, **kwargs),
                expanded,
            )
            return result

    elif structured_type == StructuredType.BATCHED:
        batch_shape = dataclass_instance.shape.batch
        batch_ndim = len(batch_shape)

        # Normalize pad_width to list of (before, after) tuples
        normalized_pad_width = _normalize_pad_width(pad_width, batch_ndim)

        # Check if we can use the existing padding_as_batch method
        # This is possible if: 1D batch, axis 0 padding, constant mode with default values
        if (
            batch_ndim == 1
            and len(normalized_pad_width) == 1
            and mode == "constant"
            and kwargs.get("constant_values", 0) == 0
        ):
            pad_before, pad_after = normalized_pad_width[0]
            target_size = batch_shape[0] + pad_before + pad_after

            # Use existing padding_as_batch method
            padded = dataclass_instance.padding_as_batch((target_size,))

            # If we need padding before, shift the data
            if pad_before > 0:
                # Create padding values (using default values)
                padding_instance = type(dataclass_instance).default((pad_before,))
                # Concatenate padding before the data
                result = jax.tree_util.tree_map(
                    lambda pad_val, data_val: jnp.concatenate([pad_val, data_val], axis=0),
                    padding_instance,
                    padded,
                )
                return result
            else:
                return padded

        # General case: create pad_width specification for jnp.pad
        pad_width_spec = normalized_pad_width

        # Apply padding to each field
        result = jax.tree_util.tree_map(
            lambda x: jnp.pad(
                x, pad_width_spec + [(0, 0)] * (x.ndim - batch_ndim), mode=mode, **kwargs
            ),
            dataclass_instance,
        )
        return result

    else:
        raise ValueError(f"Padding not supported for structured type: {structured_type}")


def stack(dataclasses: List[T], axis: int = 0) -> T:
    """
    Stack a list of xtructure dataclasses along a new axis.

    This function complements the existing reshape/flatten methods by providing
    stacking functionality for creating new dimensions from multiple instances.

    Args:
        dataclasses: List of xtructure dataclass instances to stack
        axis: Axis along which to stack (default: 0)

    Returns:
        A new dataclass instance with stacked data

    Raises:
        ValueError: If dataclasses list is empty or instances have incompatible structures
    """
    if not dataclasses:
        raise ValueError("Cannot stack empty list of dataclasses")

    if len(dataclasses) == 1:
        return jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=axis), dataclasses[0])

    # Verify all dataclasses are of the same type
    first_type = type(dataclasses[0])
    if not all(isinstance(dc, first_type) for dc in dataclasses):
        raise ValueError("All dataclasses must be of the same type")

    # Verify all have compatible structured types and shapes
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

    # Stack along the specified axis
    result = jax.tree_util.tree_map(lambda *arrays: jnp.stack(arrays, axis=axis), *dataclasses)
    return result


# Utility functions that wrap existing methods for consistency
def reshape(dataclass_instance: T, new_shape: tuple[int, ...]) -> T:
    """
    Reshape the batch dimensions of a BATCHED dataclass instance.

    This is a wrapper around the existing reshape method for consistency
    with the xtructure_numpy API.
    """
    return dataclass_instance.reshape(new_shape)


def flatten(dataclass_instance: T) -> T:
    """
    Flatten the batch dimensions of a BATCHED dataclass instance.

    This is a wrapper around the existing flatten method for consistency
    with the xtructure_numpy API.
    """
    return dataclass_instance.flatten()


def where(condition: jnp.ndarray, x: Xtructurable, y: Union[Xtructurable, Any]) -> Xtructurable:
    """
    Apply jnp.where to each field of a dataclass.

    This function is equivalent to:
    jax.tree_util.tree_map(lambda field: jnp.where(condition, field, y_field), x)

    Args:
        condition: Boolean array condition for selection
        x: Xtructurable to select from when condition is True
        y: Xtructurable or scalar to select from when condition is False

    Returns:
        Xtructurable with fields selected based on condition

    Examples:
        >>> condition = jnp.array([True, False, True])
        >>> result = xnp.where(condition, dataclass_a, dataclass_b)
        >>> # Equivalent to:
        >>> # jax.tree_util.tree_map(lambda a, b: jnp.where(condition, a, b), dataclass_a, dataclass_b)

        >>> # With scalar fallback
        >>> result = xnp.where(condition, dataclass_a, -1)
        >>> # Equivalent to:
        >>> # jax.tree_util.tree_map(lambda a: jnp.where(condition, a, -1), dataclass_a)
    """
    # Check if y is a pytree (dataclass) by checking if it has multiple leaves
    y_leaves = jax.tree_util.tree_leaves(y)
    if len(y_leaves) > 1 or (len(y_leaves) == 1 and hasattr(y, "__dataclass_fields__")):
        # y is a dataclass with tree structure
        return jax.tree_util.tree_map(
            lambda x_field, y_field: jnp.where(condition, x_field, y_field), x, y
        )
    else:
        # y is a scalar value
        return jax.tree_util.tree_map(lambda x_field: jnp.where(condition, x_field, y), x)


def unique_mask(
    val: Xtructurable,
    key: jnp.ndarray | None = None,
    batch_len: int | None = None,
    return_index: bool = False,
    return_inverse: bool = False,
) -> Union[jnp.ndarray, tuple]:
    """
    Creates a boolean mask identifying unique values in a batched Xtructurable tensor,
    keeping only the entry with the minimum cost for each unique state.
    This function is used to filter out duplicate states in batched operations,
    ensuring only the cheapest path to a state is considered.

    Args:
        val (Xtructurable): The values to check for uniqueness. Must have a uint32ed attribute.
        key (jnp.ndarray | None): The cost/priority values used for tie-breaking when multiple
            entries have the same unique identifier. If None, returns mask for first occurrence.
        batch_len (int | None): The length of the batch. If None, inferred from val.shape.batch[0].
        return_index (bool): Whether to return the indices of the unique values.
        return_inverse (bool): Whether to return the inverse indices of the unique values.

    Returns:
        - jnp.ndarray: Boolean mask if all return flags are False.
        - tuple: A tuple containing the mask and other requested arrays (index, inverse).

    Raises:
        ValueError: If val doesn't have the required uint32ed attribute.

    Examples:
        >>> # Simple unique filtering without cost consideration
        >>> mask = unique_mask(batched_states)

        >>> # With return values
        >>> mask, index, inverse = unique_mask(batched_states, return_index=True, return_inverse=True)

        >>> # Unique filtering with cost-based selection
        >>> mask, index = unique_mask(batched_states, costs, return_index=True)
        >>> unique_states = jax.tree_util.tree_map(lambda x: x[mask], batched_states)
    """
    # Verify that val has the required uint32ed attribute first
    try:
        hash_bytes = jax.vmap(lambda x: x.uint32ed)(val)
    except AttributeError:
        raise ValueError("val must have a uint32ed attribute")

    if batch_len is None:
        batch_len = val.shape.batch[0]

    # Validate key array if provided
    if key is not None and len(key) != batch_len:
        raise ValueError(f"key length {len(key)} must match batch_len {batch_len}")

    # 2. Group by Hash
    # The size argument is crucial for JIT compilation
    _, unique_indices, inv = jnp.unique(
        hash_bytes,
        axis=0,
        size=batch_len,
        return_index=True,
        return_inverse=True,
    )

    batch_idx = jnp.arange(batch_len, dtype=jnp.int32)

    if key is None:
        # Find the first occurrence of each unique group
        final_mask = jnp.zeros(batch_len, dtype=jnp.bool_).at[unique_indices].set(True)
    else:
        # 1. Isolate Keys
        # 3. Find Minimum Cost per Group
        min_costs_per_group = jnp.full((batch_len,), jnp.inf, dtype=key.dtype)
        min_costs_per_group = min_costs_per_group.at[inv].min(key)

        # 4. Primary Mask (Cost Criterion)
        min_cost_for_each_item = min_costs_per_group[inv]
        is_min_cost = key == min_cost_for_each_item

        # 5. Tie-Breaking (Index Criterion)
        indices_to_consider = jnp.where(is_min_cost, batch_idx, batch_len)
        winning_indices_per_group = jnp.full((batch_len,), batch_len, dtype=jnp.int32)
        winning_indices_per_group = winning_indices_per_group.at[inv].min(indices_to_consider)

        # 6. Final Mask
        winning_index_for_each_item = winning_indices_per_group[inv]
        final_mask = batch_idx == winning_index_for_each_item

        # Ensure that invalid (padding) entries with infinite cost are not selected.
        is_valid = key < jnp.inf
        final_mask = jnp.logical_and(final_mask, is_valid)

        if return_index:
            unique_indices = winning_indices_per_group[
                jnp.where(
                    jnp.arange(batch_len) < len(jnp.unique(inv, size=batch_len)[0]),
                    jnp.unique(inv, size=batch_len)[0],
                    0,
                )
            ]

    # Prepare return values
    if not return_index and not return_inverse:
        return final_mask

    returns = (final_mask,)
    if return_index:
        returns += (unique_indices,)
    if return_inverse:
        returns += (inv,)

    return returns
