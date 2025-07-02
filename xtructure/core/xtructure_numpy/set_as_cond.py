from typing import Any, Union

import jax.numpy as jnp
import numpy as np
from jax.ops import segment_max


def set_as_condition_on_array(
    original_array: jnp.ndarray,
    indices: Union[jnp.ndarray, tuple[jnp.ndarray, ...]],
    condition: jnp.ndarray,
    values_to_set: Any,
) -> jnp.ndarray:
    """
    Sets values in an array based on a condition, ensuring "first True wins"
    for duplicate indices.
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

        result = set_as_condition_on_array(
            reshaped_array, raveled_indices, condition, values_to_set
        )
        return result.reshape(original_array.shape)

    num_updates = len(condition)
    if num_updates == 0:
        return original_array

    # Use segment_max to find the first update for each index where condition is True.
    # To find the minimum timestamp (first True), we use a trick with segment_max:
    # we negate the timestamps, find the maximum (which corresponds to the minimum
    # absolute value), and then negate the result back. A very large negative number
    # is used for False conditions to ensure they are ignored in the max operation.
    timestamps = jnp.arange(num_updates)
    masked_timestamps = jnp.where(condition, -(timestamps + 1), -jnp.inf)

    num_segments = original_array.shape[0]
    # The result of segment_max will be -(t+1) for the first t, or -inf.
    first_true_timestamps_neg = segment_max(masked_timestamps, indices, num_segments=num_segments)
    # Undo the negation to get t+1.
    first_true_timestamps = -first_true_timestamps_neg

    update_indices = first_true_timestamps - 1

    if jnp.ndim(values_to_set) == 0:
        updates = jnp.full(original_array.shape, values_to_set, dtype=original_array.dtype)
    else:
        safe_update_indices = jnp.where(jnp.isfinite(update_indices), update_indices, 0).astype(
            jnp.int32
        )
        updates = values_to_set[safe_update_indices]

    condition_mask = jnp.isfinite(first_true_timestamps)
    while len(condition_mask.shape) < len(original_array.shape):
        condition_mask = jnp.expand_dims(condition_mask, -1)

    return jnp.where(condition_mask, updates, original_array)
