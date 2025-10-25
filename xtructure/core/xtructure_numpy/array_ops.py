from typing import Any, Union

import jax.numpy as jnp
import numpy as np


def _update_array_on_condition(
    original_array: jnp.ndarray,
    indices: Union[jnp.ndarray, tuple[jnp.ndarray, ...]],
    condition: jnp.ndarray,
    values_to_set: Any,
) -> jnp.ndarray:
    """
    Sets values in an array based on a condition, ensuring "first True wins"
    for duplicate indices.

    This is an internal utility function for array-level conditional updates.
    """
    # For advanced indexing, reshape the array to flatten the batch dimensions
    # and compute on the flattened version, then reshape back.
    if isinstance(indices, tuple) and hasattr(original_array, "shape"):
        batch_rank = len(indices)
        batch_shape = original_array.shape[:batch_rank]
        item_shape = original_array.shape[batch_rank:]

        flat_batch_size = np.prod(batch_shape).item()
        reshaped_array = original_array.reshape((flat_batch_size, *item_shape))

        raveled_indices = jnp.ravel_multi_index(indices, batch_shape, mode="clip")

        result = _update_array_on_condition(
            reshaped_array, raveled_indices, condition, values_to_set
        )
        return result.reshape(original_array.shape)

    condition = jnp.asarray(condition, dtype=jnp.bool_)
    num_updates = condition.size
    if num_updates == 0:
        return original_array

    indices_array = jnp.asarray(indices)
    indices_array = jnp.reshape(indices_array, (num_updates,))
    index_dtype = indices_array.dtype
    invalid_index = jnp.array(original_array.shape[0], dtype=index_dtype)
    sentinel_value = jnp.array(num_updates, dtype=index_dtype)

    update_order = jnp.arange(num_updates, dtype=index_dtype)
    invalid_fill = jnp.full_like(indices_array, invalid_index)
    true_indices = _where_no_broadcast(condition, indices_array, invalid_fill)

    sentinel_fill = jnp.full_like(update_order, sentinel_value)
    true_positions = _where_no_broadcast(condition, update_order, sentinel_fill)

    first_true_pos = jnp.full((original_array.shape[0] + 1,), sentinel_value, dtype=index_dtype)
    first_true_pos = first_true_pos.at[true_indices].min(true_positions, mode="drop")

    first_true_for_updates = jnp.take(first_true_pos, indices_array, mode="clip")
    apply_mask = condition & (first_true_for_updates == update_order)

    apply_mask = jnp.asarray(apply_mask, dtype=jnp.bool_)
    safe_indices = _where_no_broadcast(apply_mask, indices_array, invalid_fill)

    value_array = jnp.asarray(values_to_set, dtype=original_array.dtype)

    return original_array.at[safe_indices].set(value_array, mode="drop")


def _where_no_broadcast(
    condition: jnp.ndarray,
    true_values: jnp.ndarray,
    false_values: jnp.ndarray,
) -> jnp.ndarray:
    """Apply jnp.where while enforcing identical shapes to avoid implicit broadcasting."""
    condition = jnp.asarray(condition, dtype=jnp.bool_)
    true_values = jnp.asarray(true_values)
    false_values = jnp.asarray(false_values)

    if condition.shape != true_values.shape:
        raise ValueError(
            f"`condition` shape {condition.shape} must match `true_values` shape {true_values.shape} "
            "to avoid broadcasting."
        )
    if true_values.shape != false_values.shape:
        raise ValueError(
            f"`true_values` shape {true_values.shape} must match `false_values` shape "
            f"{false_values.shape} to avoid broadcasting."
        )

    if true_values.dtype != false_values.dtype:
        raise ValueError(
            f"`true_values` dtype {true_values.dtype} must match `false_values` dtype "
            f"{false_values.dtype} to avoid implicit casting."
        )

    return jnp.where(condition, true_values, false_values)
