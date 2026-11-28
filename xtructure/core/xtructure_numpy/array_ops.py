from typing import Any, Union

import jax.numpy as jnp
import numpy as np


def _update_array_on_condition(
    original_array: jnp.ndarray,
    indices: Union[jnp.ndarray, tuple[jnp.ndarray, ...]],
    condition: jnp.ndarray,
    values_to_set: Any,
    *,
    unique_indices: bool = False,
) -> jnp.ndarray:
    """
    Sets values in an array based on a condition, ensuring "first True wins"
    for duplicate indices.

    This is an internal utility function for array-level conditional updates.

    ``unique_indices=True`` is the caller's promise that no index is selected
    twice by ``condition``; the update is then a single masked scatter and skips
    the first-true-wins resolution. Sparse unsigned row updates elect the first
    active writer in batch-sized storage.
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
            reshaped_array,
            raveled_indices,
            condition,
            values_to_set,
            unique_indices=unique_indices,
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
    if index_dtype == jnp.uint64:
        # JAX can narrow scatter indices for small tables. Drop wide invalid
        # keys before their low bits can alias a writable row.
        indices_array = jnp.minimum(indices_array, invalid_index)

    invalid_fill = jnp.full_like(indices_array, invalid_index)
    keys = _where_no_broadcast(condition, indices_array, invalid_fill)
    if unique_indices:
        safe_indices = keys
    elif (
        jnp.issubdtype(index_dtype, jnp.unsignedinteger)
        and original_array.shape[0] >= 4 * num_updates
    ):
        # A stable key/lane sort uses batch-sized storage. Keep the dense
        # scatter-min path when its destination map is smaller than the sort buffers.
        lanes = jnp.argsort(keys, stable=True)
        sorted_keys = keys[lanes]
        first = jnp.concatenate(
            (jnp.ones((1,), dtype=jnp.bool_), sorted_keys[1:] != sorted_keys[:-1])
        )
        first_in_batch = jnp.zeros_like(condition).at[lanes].set(first)
        safe_indices = _where_no_broadcast(condition & first_in_batch, indices_array, invalid_fill)
    else:
        update_order = jnp.arange(num_updates, dtype=index_dtype)
        true_positions = _where_no_broadcast(
            condition, update_order, jnp.full_like(update_order, sentinel_value)
        )
        first_true_pos = jnp.full((original_array.shape[0] + 1,), sentinel_value, dtype=index_dtype)
        first_true_pos = first_true_pos.at[keys].min(true_positions, mode="drop")
        first_true_for_updates = jnp.take(first_true_pos, indices_array, mode="clip")
        apply_mask = condition & (first_true_for_updates == update_order)
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
