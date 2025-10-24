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

    condition = jnp.asarray(condition, dtype=bool)
    num_updates = len(condition)
    if num_updates == 0:
        return original_array

    indices_array = jnp.asarray(indices)
    indices_array = jnp.reshape(indices_array, (num_updates,))
    invalid_index = jnp.array(original_array.shape[0], dtype=indices_array.dtype)
    # Drop False updates by mapping them to an out-of-bounds position.
    safe_indices = jnp.where(condition, indices_array, invalid_index)
    # Reverse so that earlier True entries are applied last ("first True wins").
    safe_indices = jnp.flip(safe_indices, axis=0)

    value_array = jnp.asarray(values_to_set, dtype=original_array.dtype)
    if value_array.ndim > 0 and value_array.shape[0] == num_updates:
        value_array = jnp.flip(value_array, axis=0)

    return original_array.at[safe_indices].set(value_array, mode="drop")
