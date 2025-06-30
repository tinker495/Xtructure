from typing import Any, Union

import jax.numpy as jnp


def set_as_condition_on_array(
    original_array: jnp.ndarray,
    indices: Union[jnp.ndarray, tuple[jnp.ndarray, ...]],
    condition: jnp.ndarray,
    values_to_set: Any,
) -> jnp.ndarray:

    original_array = original_array.at[indices].set(
        jnp.where(condition, values_to_set, original_array[indices])
    )
    return original_array
