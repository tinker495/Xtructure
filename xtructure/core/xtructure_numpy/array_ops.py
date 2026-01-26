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
    if condition.shape != indices_array.shape:
        raise ValueError(
            f"`condition` shape {condition.shape} must match `indices` shape {indices_array.shape}."
        )

    indices_array = jnp.reshape(indices_array, (num_updates,))
    index_dtype = indices_array.dtype
    invalid_index = jnp.array(original_array.shape[0], dtype=index_dtype)

    # Enforce drop semantics for out-of-bounds 1D indices.
    # Avoid int64 casts; many environments run with x64 disabled.
    if not jnp.issubdtype(index_dtype, jnp.integer):
        raise TypeError(f"indices must be an integer array, got dtype={index_dtype}")

    n0 = jnp.array(original_array.shape[0], dtype=index_dtype)
    if jnp.issubdtype(index_dtype, jnp.unsignedinteger):
        in_bounds = indices_array < n0
    else:
        zero = jnp.array(0, dtype=index_dtype)
        in_bounds = jnp.logical_and(indices_array >= zero, indices_array < n0)

    cond_flat = jnp.reshape(condition, (num_updates,))
    cond_valid = jnp.logical_and(cond_flat, in_bounds)

    true_indices = jnp.where(cond_valid, indices_array, invalid_index)

    # Stable sort by index; within each index group, earlier update_order comes first.
    perm = jnp.argsort(true_indices, stable=True)
    sorted_true_indices = true_indices[perm]

    if num_updates == 1:
        winners_sorted = sorted_true_indices != invalid_index
    else:
        is_first = jnp.concatenate(
            [jnp.array([True]), sorted_true_indices[1:] != sorted_true_indices[:-1]], axis=0
        )
        winners_sorted = jnp.logical_and(is_first, sorted_true_indices != invalid_index)

    # Map winner mask back to original update order (perm is a bijection).
    winners = jnp.zeros((num_updates,), dtype=jnp.bool_).at[perm].set(winners_sorted)

    safe_indices = jnp.where(winners, indices_array, invalid_index)
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
