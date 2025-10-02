from typing import Any, Union

import jax
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

    num_updates = len(condition)
    if num_updates == 0:
        return original_array

    def _apply_updates(_):
        # Identify the earliest True entry for each index while keeping all tensors
        # bounded by the number of updates instead of the full table size.
        true_positions = jnp.nonzero(condition, size=num_updates, fill_value=-1)[0]
        safe_positions = jnp.where(true_positions >= 0, true_positions, 0)

        gathered_indices = indices[safe_positions]
        gathered_indices = jnp.where(true_positions >= 0, gathered_indices, -1)

        unique_indices, first_pos = jnp.unique(
            gathered_indices,
            size=num_updates,
            fill_value=-1,
            return_index=True,
        )

        valid_unique = unique_indices >= 0
        safe_first_pos = jnp.where(valid_unique, first_pos, 0)
        selected_positions = safe_positions[safe_first_pos]

        target_indices = jnp.where(valid_unique, unique_indices, 0).astype(indices.dtype)

        value_array = jnp.asarray(values_to_set)
        if value_array.ndim == 0:
            trailing_shape = original_array.shape[1:]
            value_array = jnp.broadcast_to(
                value_array.astype(original_array.dtype),
                (num_updates, *trailing_shape),
            )

        updates = jnp.take(value_array, selected_positions, axis=0, mode="clip")
        fallback = jnp.take(original_array, target_indices, axis=0, mode="clip")

        mask = valid_unique
        while mask.ndim < updates.ndim:
            mask = mask[..., None]

        updates = jnp.where(mask, updates, fallback)

        return original_array.at[target_indices].set(updates)

    return jax.lax.cond(
        jnp.any(condition),
        _apply_updates,
        lambda _: original_array,
        operand=None,
    )
