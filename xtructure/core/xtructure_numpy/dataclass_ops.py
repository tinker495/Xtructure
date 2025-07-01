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
    target_size: Union[int, tuple[int, ...]],
    axis: int = 0,
    mode: str = "constant",
    **kwargs,
) -> T:
    """
    Pad an xtructure dataclass to a target size along the specified axis.

    This function extends the existing padding_as_batch method to support more
    flexible padding options including different modes and custom values.

    Args:
        dataclass_instance: The xtructure dataclass instance to pad
        target_size: Target size for the specified axis, or target shape for all batch dimensions
        axis: Axis along which to pad (default: 0)
        mode: Padding mode ('constant', 'edge', 'linear_ramp', 'maximum', 'mean', 'median',
        'minimum', 'reflect', 'symmetric', 'wrap'). See jnp.pad for more details.
        **kwargs: Additional arguments passed to jnp.pad (e.g., constant_values for 'constant' mode)

    Returns:
        A new dataclass instance with padded data

    Raises:
        ValueError: If target_size is smaller than current size or incompatible with structure
    """
    structured_type = dataclass_instance.structured_type

    if structured_type == StructuredType.SINGLE:
        if isinstance(target_size, int):
            if target_size <= 0:
                raise ValueError("Target size must be positive for SINGLE structured type")

            # For simple cases where we're just replicating, use the existing pattern
            if mode == "constant" and kwargs.get("constant_values", 0) == 0:
                # Use similar approach as padding_as_batch but for replication
                result = jax.tree_util.tree_map(
                    lambda x: jnp.tile(jnp.expand_dims(x, axis=0), (target_size,) + (1,) * x.ndim),
                    dataclass_instance,
                )
                return result
            else:
                # For other modes, expand dims and then pad
                expanded = jax.tree_util.tree_map(
                    lambda x: jnp.expand_dims(x, axis=0), dataclass_instance
                )
                pad_width = target_size - 1
                pad_width_spec = [(0, pad_width)]

                result = jax.tree_util.tree_map(
                    lambda x: jnp.pad(
                        x, pad_width_spec + [(0, 0)] * (x.ndim - 1), mode=mode, **kwargs
                    ),
                    expanded,
                )
                return result
        else:
            raise ValueError("For SINGLE structured type, target_size must be an integer")

    elif structured_type == StructuredType.BATCHED:
        batch_shape = dataclass_instance.shape.batch

        if isinstance(target_size, int):
            # Check if we can use the existing padding_as_batch method
            if (
                axis == 0
                and len(batch_shape) == 1
                and mode == "constant"
                and kwargs.get("constant_values", 0) == 0
            ):
                # Use existing padding_as_batch method for simple case
                if target_size < batch_shape[0]:
                    raise ValueError(
                        f"Target size {target_size} is smaller than current size {batch_shape[0]}"
                    )
                return dataclass_instance.padding_as_batch((target_size,))

            # For other cases, use general padding
            axis_adjusted = axis if axis >= 0 else len(batch_shape) + axis
            if axis_adjusted >= len(batch_shape):
                raise ValueError(
                    f"Padding axis {axis} is out of bounds for batch shape {batch_shape}"
                )

            current_size = batch_shape[axis_adjusted]
            if target_size < current_size:
                raise ValueError(
                    f"Target size {target_size} is smaller than current size {current_size}"
                )

            if target_size == current_size:
                return dataclass_instance

            # Calculate padding width
            pad_width = target_size - current_size

            # Create pad_width tuple for jnp.pad
            pad_width_spec = [(0, 0)] * len(batch_shape)
            pad_width_spec[axis_adjusted] = (0, pad_width)

            # Apply padding to each field
            result = jax.tree_util.tree_map(
                lambda x: jnp.pad(
                    x, pad_width_spec + [(0, 0)] * (x.ndim - len(batch_shape)), mode=mode, **kwargs
                ),
                dataclass_instance,
            )
            return result

        elif isinstance(target_size, tuple):
            # Check if we can use existing padding_as_batch method
            if (
                len(target_size) == 1
                and len(batch_shape) == 1
                and mode == "constant"
                and kwargs.get("constant_values", 0) == 0
            ):
                # Use existing method for simple case
                if target_size[0] < batch_shape[0]:
                    raise ValueError(
                        f"Target size {target_size[0]} is smaller than current size {batch_shape[0]}"
                    )
                return dataclass_instance.padding_as_batch(target_size)

            # General case for multi-dimensional padding
            if len(target_size) != len(batch_shape):
                raise ValueError(
                    f"Target shape {target_size} must have same number of dimensions as batch shape {batch_shape}"
                )

            # Check that target is larger or equal in all dimensions
            for i, (current, target) in enumerate(zip(batch_shape, target_size)):
                if target < current:
                    raise ValueError(
                        f"Target size {target} at axis {i} is smaller than current size {current}"
                    )

            if target_size == batch_shape:
                return dataclass_instance

            # Calculate padding for each dimension
            pad_width_spec = [
                (0, target - current) for current, target in zip(batch_shape, target_size)
            ]

            # Apply padding to each field
            result = jax.tree_util.tree_map(
                lambda x: jnp.pad(
                    x, pad_width_spec + [(0, 0)] * (x.ndim - len(batch_shape)), mode=mode, **kwargs
                ),
                dataclass_instance,
            )
            return result
        else:
            raise ValueError("target_size must be an integer or tuple of integers")

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
    val: Xtructurable, key: jnp.ndarray | None = None, batch_len: int | None = None
) -> jnp.ndarray:
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

    Returns:
        jnp.ndarray: Boolean mask where True indicates the single, cheapest unique value.

    Raises:
        ValueError: If val doesn't have the required uint32ed attribute.

    Examples:
        >>> # Simple unique filtering without cost consideration
        >>> mask = unique_mask(batched_states)

        >>> # Unique filtering with cost-based selection
        >>> mask = unique_mask(batched_states, costs)
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
    _unique_hashes, inv = jnp.unique(hash_bytes, axis=0, size=batch_len, return_inverse=True)
    if key is None:
        # Find the first occurrence of each unique group
        # inv[i] tells us which group item i belongs to
        # We want the first index for each group
        batch_idx = jnp.arange(batch_len, dtype=jnp.int32)
        first_occurrence_per_group = jnp.full((batch_len,), batch_len, dtype=jnp.int32)
        first_occurrence_per_group = first_occurrence_per_group.at[inv].min(batch_idx)
        first_occurrence_for_each_item = first_occurrence_per_group[inv]
        return batch_idx == first_occurrence_for_each_item

    # 1. Isolate Keys
    batch_idx = jnp.arange(batch_len, dtype=jnp.int32)

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

    return final_mask
