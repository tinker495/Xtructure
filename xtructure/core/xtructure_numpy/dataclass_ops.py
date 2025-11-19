"""Operations for concatenating and padding xtructure dataclasses.

This module provides operations that complement the existing structure utilities
in xtructure_decorators.structure_util, reusing existing methods where possible.
"""

from typing import Any, Callable, List, TypeVar, Union

import jax
import jax.numpy as jnp

from xtructure.core.structuredtype import StructuredType

from ..xtructure_decorators import Xtructurable
from .array_ops import _update_array_on_condition, _where_no_broadcast

T = TypeVar("T")


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

    # For SINGLE structured type, this operation is equivalent to stacking
    if first_structured_type == StructuredType.SINGLE:
        return stack(dataclasses, axis=axis)

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

    # Check for no-op case (zero padding)
    if structured_type == StructuredType.SINGLE:
        if isinstance(pad_width, int) and pad_width == 0:
            return dataclass_instance
        elif isinstance(pad_width, (list, tuple)):
            if len(pad_width) == 1:
                if isinstance(pad_width[0], (list, tuple)) and len(pad_width[0]) == 2:
                    if pad_width[0] == (0, 0):
                        return dataclass_instance
                elif pad_width[0] == 0:
                    return dataclass_instance
            elif len(pad_width) == 2 and pad_width == (0, 0):
                return dataclass_instance
    elif structured_type == StructuredType.BATCHED:
        batch_ndim = len(dataclass_instance.shape.batch)
        normalized_pad_width = _normalize_pad_width(pad_width, batch_ndim)
        if all(before == 0 and after == 0 for before, after in normalized_pad_width):
            return dataclass_instance

    if structured_type == StructuredType.SINGLE:
        # For SINGLE type, expand to batch dimension of size 1 and apply BATCHED logic
        expanded = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), dataclass_instance)
        # Apply padding to the expanded instance using the BATCHED logic directly
        # to avoid infinite recursion
        batch_shape = expanded.shape.batch
        batch_ndim = len(batch_shape)
        normalized_pad_width = _normalize_pad_width(pad_width, batch_ndim)

        # Check if we can use the existing padding_as_batch method
        if (
            batch_ndim == 1
            and len(normalized_pad_width) == 1
            and mode == "constant"
            and "constant_values" not in kwargs
        ):
            pad_before, pad_after = normalized_pad_width[0]
            target_size = batch_shape[0] + pad_before + pad_after
            padded = expanded.padding_as_batch((target_size,))

            if pad_before > 0:
                padding_instance = type(expanded).default((pad_before,))
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

        if mode == "constant" and "constant_values" not in kwargs:
            default_instance = type(expanded).default()
            result = jax.tree_util.tree_map(
                lambda x, default_val: jnp.pad(
                    x,
                    pad_width_spec + [(0, 0)] * (x.ndim - batch_ndim),
                    mode=mode,
                    constant_values=default_val,
                ),
                expanded,
                default_instance,
            )
            return result
        else:
            result = jax.tree_util.tree_map(
                lambda x: jnp.pad(
                    x, pad_width_spec + [(0, 0)] * (x.ndim - batch_ndim), mode=mode, **kwargs
                ),
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
            and "constant_values" not in kwargs
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

        # For constant mode without explicit constant_values, use field-specific defaults
        if mode == "constant" and "constant_values" not in kwargs:
            # Create a default instance to get field-specific default values
            default_instance = type(dataclass_instance).default()

            # Apply padding with field-specific constant values
            result = jax.tree_util.tree_map(
                lambda x, default_val: jnp.pad(
                    x,
                    pad_width_spec + [(0, 0)] * (x.ndim - batch_ndim),
                    mode=mode,
                    constant_values=default_val,
                ),
                dataclass_instance,
                default_instance,
            )
            return result
        else:
            # Apply padding to each field with provided kwargs
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
    condition_array = jnp.asarray(condition, dtype=jnp.bool_)

    def _align_condition(target_shape: tuple[int, ...]) -> jnp.ndarray:
        if condition_array.shape == target_shape:
            return condition_array
        try:
            return jnp.broadcast_to(condition_array, target_shape)
        except ValueError as err:
            raise ValueError(
                f"`condition` with shape {condition_array.shape} cannot be broadcast to target shape {target_shape}."
            ) from err

    # Check if y is a pytree (dataclass) by checking if it has multiple leaves
    y_leaves = jax.tree_util.tree_leaves(y)
    if len(y_leaves) > 1 or (len(y_leaves) == 1 and hasattr(y, "__dataclass_fields__")):
        # y is a dataclass with tree structure
        def _apply_dataclass_where(x_field, y_field):
            cond = _align_condition(x_field.shape)
            y_array = jnp.asarray(y_field)
            if y_array.shape != x_field.shape:
                try:
                    y_array = jnp.broadcast_to(y_array, x_field.shape)
                except ValueError as err:
                    raise ValueError(
                        f"`y` field with shape {y_array.shape} cannot be"
                        "broadcast to match `x` field shape {x_field.shape}."
                        f"Original `y` shape: {y_array.shape}, `x` shape: {x_field.shape}."
                    ) from err
            target_dtype = jnp.result_type(x_field.dtype, y_array.dtype)
            return _where_no_broadcast(
                cond,
                jnp.asarray(x_field, dtype=target_dtype),
                jnp.asarray(y_array, dtype=target_dtype),
            )

        return jax.tree_util.tree_map(_apply_dataclass_where, x, y)
    else:
        # y is a scalar value
        scalar_value = jnp.asarray(y)

        def _apply_scalar_where(x_field):
            cond = _align_condition(x_field.shape)
            try:
                y_array = jnp.broadcast_to(scalar_value, x_field.shape)
            except ValueError as err:
                raise ValueError(
                    f"`y` value with shape {scalar_value.shape} cannot be"
                    "broadcast to match `x` field shape {x_field.shape}."
                    f"Original `y` shape: {scalar_value.shape}, `x` shape: {x_field.shape}."
                ) from err
            target_dtype = jnp.result_type(x_field.dtype, y_array.dtype)
            return _where_no_broadcast(
                cond,
                jnp.asarray(x_field, dtype=target_dtype),
                jnp.asarray(y_array, dtype=target_dtype),
            )

        return jax.tree_util.tree_map(_apply_scalar_where, x)


def where_no_broadcast(
    condition: Union[jnp.ndarray, Xtructurable],
    x: Xtructurable,
    y: Xtructurable,
) -> Xtructurable:
    """
    Variant of where that forbids implicit broadcasting by enforcing shape/dtype equality.

    Args:
        condition: Boolean mask with the same tree structure and shapes as the dataclass fields,
            or a single boolean array that exactly matches every field's shape.
        x: Dataclass instance providing values where condition is True.
        y: Dataclass instance providing values where condition is False. Must match the structure
            and dtypes of `x`.

    Returns:
        Dataclass with values selected without relying on broadcasting.

    Raises:
        TypeError: If dataclass structures do not match.
        ValueError: If any field requires broadcasting or implicit dtype casting.
    """

    if type(x) is not type(y):
        raise TypeError(
            "`x` and `y` must be instances of the same dataclass for where_no_broadcast."
        )

    condition_is_dataclass = hasattr(condition, "__dataclass_fields__")

    if condition_is_dataclass:
        condition_structure = jax.tree_util.tree_structure(condition)
        x_structure = jax.tree_util.tree_structure(x)
        if condition_structure != x_structure:
            raise TypeError(
                "`condition` must share the same dataclass structure as `x` and `y` "
                "when provided as a dataclass."
            )

        return jax.tree_util.tree_map(
            lambda cond_field, x_field, y_field: _where_no_broadcast(cond_field, x_field, y_field),
            condition,
            x,
            y,
        )

    condition_array = jnp.asarray(condition, dtype=jnp.bool_)
    return jax.tree_util.tree_map(
        lambda x_field, y_field: _where_no_broadcast(condition_array, x_field, y_field),
        x,
        y,
    )


def unique_mask(
    val: Xtructurable,
    key: jnp.ndarray | None = None,
    filled: jnp.ndarray | None = None,
    key_fn: Callable[[Any], jnp.ndarray] | None = None,
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
        val (Xtructurable): The values to check for uniqueness.
        key (jnp.ndarray | None): The cost/priority values used for tie-breaking when multiple
            entries have the same unique identifier. If None, returns mask for first occurrence.
        key_fn (Callable[[Any], jnp.ndarray] | None): Function to generate hashable keys from
            dataclass instances. If None, defaults to lambda x: x.uint32ed for backward compatibility.
        batch_len (int | None): The length of the batch. If None, inferred from val.shape.batch[0].
        return_index (bool): Whether to return the indices of the unique values.
        return_inverse (bool): Whether to return the inverse indices of the unique values.

    Returns:
        - jnp.ndarray: Boolean mask if all return flags are False.
        - tuple: A tuple containing the mask and other requested arrays (index, inverse).

    Raises:
        ValueError: If val doesn't have the required attributes or key_fn fails.

    Examples:
        >>> # Simple unique filtering without cost consideration
        >>> mask = unique_mask(batched_states)

        >>> # With custom key function
        >>> mask = unique_mask(batched_states, key_fn=lambda x: x.position)

        >>> # With return values
        >>> mask, index, inverse = unique_mask(batched_states, return_index=True, return_inverse=True)

        >>> # Unique filtering with cost-based selection
        >>> mask, index = unique_mask(batched_states, costs, return_index=True)
        >>> unique_states = jax.tree_util.tree_map(lambda x: x[mask], batched_states)
    """
    # Use default key_fn for backward compatibility
    if key_fn is None:

        def key_fn(x):
            return x.uint32ed

    # Generate hashable keys from dataclass instances
    try:
        hash_bytes = jax.vmap(key_fn)(val)
    except Exception as e:
        raise ValueError(f"key_fn failed to generate hashable keys: {e}")

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
        # Apply filled mask if provided
        if filled is not None:
            final_mask = jnp.logical_and(final_mask, filled)
    else:
        # When 'filled' is provided, we can avoid computation on non-filled items
        if filled is not None:
            # Set non-filled items to have infinite cost to exclude them from consideration
            inf_fill = jnp.full_like(key, jnp.inf)
            masked_key = _where_no_broadcast(filled, key, inf_fill)
        else:
            masked_key = key

        # 1. Isolate Keys
        # 3. Find Minimum Cost per Group using masked key
        min_costs_per_group = jnp.full((batch_len,), jnp.inf, dtype=key.dtype)
        min_costs_per_group = min_costs_per_group.at[inv].min(masked_key)

        # 4. Primary Mask (Cost Criterion)
        min_cost_for_each_item = min_costs_per_group[inv]
        is_min_cost = masked_key == min_cost_for_each_item

        # 5. Tie-Breaking (Index Criterion) - only consider filled items
        if filled is not None:
            # Only consider items that have min cost AND are filled
            can_be_considered = jnp.logical_and(is_min_cost, filled)
            fallback_idx = jnp.full_like(batch_idx, batch_len)
            indices_to_consider = _where_no_broadcast(can_be_considered, batch_idx, fallback_idx)
        else:
            fallback_idx = jnp.full_like(batch_idx, batch_len)
            indices_to_consider = _where_no_broadcast(is_min_cost, batch_idx, fallback_idx)

        winning_indices_per_group = jnp.full((batch_len,), batch_len, dtype=jnp.int32)
        winning_indices_per_group = winning_indices_per_group.at[inv].min(indices_to_consider)

        # 6. Final Mask
        winning_index_for_each_item = winning_indices_per_group[inv]
        final_mask = batch_idx == winning_index_for_each_item

        # Ensure that invalid (padding) entries with infinite cost are not selected
        # When filled is provided, this check is redundant since we already masked with inf
        if filled is None:
            is_valid = key < jnp.inf
            final_mask = jnp.logical_and(final_mask, is_valid)

        if return_index:
            unique_group_ids, _ = jnp.unique(inv, size=batch_len, return_index=True)
            unique_indices = winning_indices_per_group[unique_group_ids]

    # Prepare return values
    if not return_index and not return_inverse:
        return final_mask

    returns = (final_mask,)
    if return_index:
        returns += (unique_indices,)
    if return_inverse:
        returns += (inv,)

    return returns


def take(dataclass_instance: T, indices: jnp.ndarray, axis: int = 0) -> T:
    """
    Take elements from a dataclass along the specified axis.

    This function extracts elements at the given indices from each field of the dataclass,
    similar to jnp.take but applied to all fields of a dataclass.

    Args:
        dataclass_instance: The dataclass instance to take elements from
        indices: Array of indices to take
        axis: Axis along which to take elements (default: 0)

    Returns:
        A new dataclass instance with elements taken from the specified indices

    Examples:
        >>> # Take specific elements from a batched dataclass
        >>> data = MyData.default((5,))
        >>> result = xnp.take(data, jnp.array([0, 2, 4]))
        >>> # result will have batch shape (3,) with elements at indices 0, 2, 4

        >>> # Take elements along a different axis
        >>> data = MyData.default((3, 4))
        >>> result = xnp.take(data, jnp.array([1, 3]), axis=1)
        >>> # result will have batch shape (3, 2) with elements at indices 1, 3 along axis 1
    """
    return jax.tree_util.tree_map(lambda x: jnp.take(x, indices, axis=axis), dataclass_instance)


def take_along_axis(dataclass_instance: T, indices: jnp.ndarray, axis: int) -> T:
    """
    Take values from a dataclass along an axis using indices whose shape matches the result.

    This mirrors jnp.take_along_axis by applying it to every leaf array in the dataclass.
    The indices array must have the same shape as the output and match the input shape
    everywhere except at the specified axis.

    Args:
        dataclass_instance: Dataclass to gather values from.
        indices: Index array broadcastable to the output shape (see jnp.take_along_axis).
        axis: Axis along which values are gathered.

    Returns:
        Dataclass instance with gathered values along the requested axis.

    Examples:
        >>> data = MyData.default((3, 4))
        >>> idx = jnp.array([[0, 2, 1, 3]]).T  # shape (4, 1)
        >>> result = xnp.take_along_axis(data, idx, axis=1)
    """
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
        # Grow trailing singleton axes so expanded.ndim matches the leaf ndim.
        for _ in range(leaf.ndim - expanded.ndim):
            expanded = expanded[..., None]

        # Broadcast indices over the extra field dimensions (pattern from user snippet).
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
    """
    Construct an array by repeating a dataclass instance the number of times given by reps.

    This function replicates a dataclass instance along specified axes, similar to jnp.tile
    but applied to all fields of a dataclass.

    Args:
        dataclass_instance: The dataclass instance to tile
        reps: The number of repetitions of dataclass_instance along each axis.
              If reps has length d, the result will have that dimension.
              If reps is an int, it is treated as a 1-tuple.

    Returns:
        A new dataclass instance with tiled data

    Examples:
        >>> # Tile a single dataclass to create a batch
        >>> data = MyData.default()
        >>> result = xnp.tile(data, 3)
        >>> # result will have batch shape (3,) with repeated data

        >>> # Tile a batched dataclass along multiple axes
        >>> data = MyData.default((2,))
        >>> result = xnp.tile(data, (2, 3))
        >>> # result will have batch shape (4, 3) with tiled data

        >>> # Tile along specific dimensions
        >>> data = MyData.default((2, 3))
        >>> result = xnp.tile(data, (1, 2, 1))
        >>> # result will have batch shape (2, 6, 3) with tiled data
    """
    # Normalize reps to a tuple
    if isinstance(reps, int):
        reps = (reps,)

    # Apply tile to each field
    return jax.tree_util.tree_map(lambda x: jnp.tile(x, reps), dataclass_instance)


def update_on_condition(
    dataclass_instance: T,
    indices: Union[jnp.ndarray, tuple[jnp.ndarray, ...]],
    condition: jnp.ndarray,
    values_to_set: Union[T, Any],
) -> T:
    """
    Update values in a dataclass based on a condition, ensuring "first True wins"
    for duplicate indices.

    This function applies conditional updates to all fields of a dataclass,
    similar to how jnp.where works but with support for duplicate index handling.

    Args:
        dataclass_instance: The dataclass instance to update
        indices: Indices where updates should be applied
        condition: Boolean array indicating which updates should be applied
        values_to_set: Values to set when condition is True. Can be a dataclass
            instance (compatible with dataclass_instance) or a scalar value.

    Returns:
        A new dataclass instance with updated values

    Examples:
        >>> # Update with scalar value
        >>> updated = update_on_condition(dataclass, indices, condition, -1)

        >>> # Update with another dataclass
        >>> updated = update_on_condition(dataclass, indices, condition, new_values)
    """
    # Check if values_to_set is a dataclass (has multiple leaves)
    values_leaves = jax.tree_util.tree_leaves(values_to_set)
    if len(values_leaves) > 1 or (
        len(values_leaves) == 1 and hasattr(values_to_set, "__dataclass_fields__")
    ):
        # values_to_set is a dataclass - apply update to each field
        return jax.tree_util.tree_map(
            lambda field, values_field: _update_array_on_condition(
                field, indices, condition, values_field
            ),
            dataclass_instance,
            values_to_set,
        )
    else:
        # values_to_set is a scalar - apply to all fields
        return jax.tree_util.tree_map(
            lambda field: _update_array_on_condition(field, indices, condition, values_to_set),
            dataclass_instance,
        )


def transpose(dataclass_instance: T, axes: Union[tuple[int, ...], None] = None) -> T:
    """
    Transpose the batch dimensions of a dataclass instance.

    This function applies transpose only to the batch dimensions of each field,
    preserving the field-specific dimensions (like vector dimensions).

    Args:
        dataclass_instance: The dataclass instance to transpose
        axes: Tuple or list of ints, a permutation of [0,1,..,N-1] where N is the number of batch axes.
              If None, batch axes are reversed.

    Returns:
        A new dataclass instance with transposed batch dimensions

    Examples:
        >>> # Transpose a 2D batched dataclass
        >>> data = MyData.default((3, 4))
        >>> result = xnp.transpose(data)
        >>> # result will have batch shape (4, 3)

        >>> # Transpose with specific axes order
        >>> data = MyData.default((2, 3, 4))
        >>> result = xnp.transpose(data, axes=(2, 0, 1))
        >>> # result will have batch shape (4, 2, 3)

        >>> # For vector dataclass, only batch dimensions are transposed
        >>> data = VectorData.default((2, 3))  # batch shape (2, 3), vector shape (3,)
        >>> result = xnp.transpose(data)
        >>> # result will have batch shape (3, 2), vector shape remains (3,)
    """
    # Get the batch shape to determine how many batch dimensions we have
    batch_shape = dataclass_instance.shape.batch
    if isinstance(batch_shape, int):
        # Single dimension batch
        batch_ndim = 1
    else:
        batch_ndim = len(batch_shape)

    # If no axes specified, reverse the batch axes
    if axes is None:
        axes = tuple(range(batch_ndim - 1, -1, -1))

    # Apply transpose only to the batch dimensions
    def transpose_batch_only(field):
        # For fields with more dimensions than batch, we need to transpose only the batch part
        field_ndim = field.ndim
        if field_ndim <= batch_ndim:
            # Field has same or fewer dimensions than batch, transpose all
            return jnp.transpose(field, axes=axes)
        else:
            # Field has more dimensions than batch (e.g., vector fields)
            # We need to transpose only the first batch_ndim dimensions
            # Create a full axes permutation that keeps non-batch dimensions in place
            full_axes = list(axes) + list(range(batch_ndim, field_ndim))
            return jnp.transpose(field, axes=full_axes)

    return jax.tree_util.tree_map(transpose_batch_only, dataclass_instance)


def swap_axes(dataclass_instance: T, axis1: int, axis2: int) -> T:
    """
    Swap two batch axes of a dataclass instance.

    This function applies swap_axes only to the batch dimensions of each field,
    preserving the field-specific dimensions (like vector dimensions).

    Args:
        dataclass_instance: The dataclass instance to swap axes for
        axis1: First batch axis to swap
        axis2: Second batch axis to swap

    Returns:
        A new dataclass instance with swapped batch axes

    Examples:
        >>> # Swap first and second batch axes
        >>> data = MyData.default((3, 4, 5))
        >>> result = xnp.swap_axes(data, 0, 1)
        >>> # result will have batch shape (4, 3, 5)

        >>> # Swap last two batch axes
        >>> data = MyData.default((2, 3, 4))
        >>> result = xnp.swap_axes(data, -1, -2)
        >>> # result will have batch shape (2, 4, 3)

        >>> # For vector dataclass, only batch dimensions are swapped
        >>> data = VectorData.default((2, 3))  # batch shape (2, 3), vector shape (3,)
        >>> result = xnp.swap_axes(data, 0, 1)
        >>> # result will have batch shape (3, 2), vector shape remains (3,)
    """
    # Get the batch shape to determine how many batch dimensions we have
    batch_shape = dataclass_instance.shape.batch
    if isinstance(batch_shape, int):
        # Single dimension batch
        batch_ndim = 1
    else:
        batch_ndim = len(batch_shape)

    # Normalize negative indices to positive indices within batch dimensions
    def normalize_axis(axis):
        if axis < 0:
            return batch_ndim + axis
        return axis

    axis1_norm = normalize_axis(axis1)
    axis2_norm = normalize_axis(axis2)

    # Validate that axes are within batch dimensions
    if axis1_norm < 0 or axis1_norm >= batch_ndim:
        raise ValueError(f"Axis {axis1} is out of bounds for batch dimensions {batch_shape}")
    if axis2_norm < 0 or axis2_norm >= batch_ndim:
        raise ValueError(f"Axis {axis2} is out of bounds for batch dimensions {batch_shape}")

    # Apply swap_axes only to the batch dimensions
    def swap_batch_axes_only(field):
        # For fields with more dimensions than batch, we need to swap only the batch part
        field_ndim = field.ndim
        if field_ndim <= batch_ndim:
            # Field has same or fewer dimensions than batch, swap directly
            return jnp.swapaxes(field, axis1_norm, axis2_norm)
        else:
            # Field has more dimensions than batch (e.g., vector fields)
            # We need to swap only the batch dimensions, keeping non-batch dimensions in place
            return jnp.swapaxes(field, axis1_norm, axis2_norm)

    return jax.tree_util.tree_map(swap_batch_axes_only, dataclass_instance)


def expand_dims(dataclass_instance: T, axis: int) -> T:
    """
    Insert a new axis that will appear at the axis position in the expanded array shape.

    Args:
        dataclass_instance: The dataclass instance to expand dimensions.
        axis: Position in the expanded axes where the new axis (or axes) is placed.

    Returns:
        A new dataclass instance with expanded dimensions.
    """
    return jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=axis), dataclass_instance)


def squeeze(dataclass_instance: T, axis: Union[int, tuple[int, ...], None] = None) -> T:
    """
    Remove axes of length one from the dataclass.

    Args:
        dataclass_instance: The dataclass instance to squeeze.
        axis: Selects a subset of the single-dimensional entries in the shape.

    Returns:
        A new dataclass instance with squeezed dimensions.
    """
    return jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=axis), dataclass_instance)


def repeat(dataclass_instance: T, repeats: Union[int, jnp.ndarray], axis: int = None) -> T:
    """
    Repeat elements of a dataclass.

    Args:
        dataclass_instance: The dataclass instance to repeat.
        repeats: The number of repetitions for each element.
        axis: The axis along which to repeat values.

    Returns:
        A new dataclass instance with repeated elements.
    """
    return jax.tree_util.tree_map(lambda x: jnp.repeat(x, repeats, axis=axis), dataclass_instance)


def split(
    dataclass_instance: T, indices_or_sections: Union[int, jnp.ndarray], axis: int = 0
) -> List[T]:
    """
    Split a dataclass into multiple sub-dataclasses as specified by indices_or_sections.

    Args:
        dataclass_instance: The dataclass instance to split.
        indices_or_sections: If an integer, N, the array will be divided into N equal arrays
            along axis. If an 1-D array of sorted integers, the entries indicate where along
            axis the array is split.
        axis: The axis along which to split.

    Returns:
        A list of sub-dataclasses.
    """
    leaves, treedef = jax.tree_util.tree_flatten(dataclass_instance)
    # Split each leaf array
    split_leaves = [jnp.split(leaf, indices_or_sections, axis=axis) for leaf in leaves]

    # Transpose: list of splits of leaves -> list of leaves (for each split)
    # split_leaves is [[part1_leaf1, part2_leaf1], [part1_leaf2, part2_leaf2]]
    # We want: [[part1_leaf1, part1_leaf2], [part2_leaf1, part2_leaf2]]
    if not split_leaves:
        return []

    num_splits = len(split_leaves[0])
    result_dataclasses = []
    for i in range(num_splits):
        new_leaves = [sl[i] for sl in split_leaves]
        result_dataclasses.append(jax.tree_util.tree_unflatten(treedef, new_leaves))

    return result_dataclasses


def full_like(dataclass_instance: T, fill_value: Any) -> T:
    """
    Return a new dataclass with the same shape and type as a given dataclass, filled with fill_value.

    Args:
        dataclass_instance: The prototype dataclass instance.
        fill_value: Fill value.

    Returns:
        A new dataclass instance filled with fill_value.
    """
    return jax.tree_util.tree_map(lambda x: jnp.full_like(x, fill_value), dataclass_instance)


def zeros_like(dataclass_instance: T) -> T:
    """
    Return a new dataclass with the same shape and type as a given dataclass, filled with zeros.

    Args:
        dataclass_instance: The prototype dataclass instance.

    Returns:
        A new dataclass instance filled with zeros.
    """
    return jax.tree_util.tree_map(jnp.zeros_like, dataclass_instance)


def ones_like(dataclass_instance: T) -> T:
    """
    Return a new dataclass with the same shape and type as a given dataclass, filled with ones.

    Args:
        dataclass_instance: The prototype dataclass instance.

    Returns:
        A new dataclass instance filled with ones.
    """
    return jax.tree_util.tree_map(jnp.ones_like, dataclass_instance)
