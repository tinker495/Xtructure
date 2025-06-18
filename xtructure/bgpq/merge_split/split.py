from typing import Tuple

import jax
import jax.numpy as jnp


@jax.jit
def merge_sort_split_idx(
    ak: jax.Array, bk: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    key_concat = jnp.concatenate([ak, bk])
    indices_payload = jnp.arange(key_concat.shape[0], dtype=jnp.int32)
    sorted_key_full, sorted_idx_full = jax.lax.sort_key_val(key_concat, indices_payload)
    return sorted_key_full, sorted_idx_full
