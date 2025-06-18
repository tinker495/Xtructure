from typing import Tuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from .common import binary_search_partition

BLOCK_SIZE = 64


def merge_parallel_kernel(ak_ref, bk_ref, merged_keys_ref, merged_indices_ref):
    """
    Pallas kernel that merges two sorted arrays in parallel using the
    Merge Path algorithm for block-level partitioning.
    """
    block_idx = pl.program_id(axis=0)

    n, m = ak_ref.shape[0], bk_ref.shape[0]
    total_len = n + m

    k_start = block_idx * BLOCK_SIZE
    k_end = jnp.minimum(k_start + BLOCK_SIZE, total_len)

    a_start, b_start = binary_search_partition(k_start, ak_ref, bk_ref)
    a_end, b_end = binary_search_partition(k_end, ak_ref, bk_ref)

    initial_main_loop_state = (a_start, b_start, k_start)

    def main_loop_cond(state):
        idx_a, idx_b, _ = state
        return jnp.logical_and(idx_a < a_end, idx_b < b_end)

    def main_loop_body(state):
        idx_a, idx_b, out_ptr = state
        val_a = pl.load(ak_ref, (idx_a,))
        val_b = pl.load(bk_ref, (idx_b,))
        is_a_le_b = val_a <= val_b

        key_to_store = jnp.where(is_a_le_b, val_a, val_b)
        idx_to_store = jnp.where(is_a_le_b, idx_a, n + idx_b)

        key_casted = key_to_store.astype(merged_keys_ref.dtype)
        pl.store(
            merged_keys_ref, (pl.ds(out_ptr, 1),), jnp.expand_dims(key_casted, 0)
        )
        pl.store(
            merged_indices_ref,
            (pl.ds(out_ptr, 1),),
            jnp.expand_dims(idx_to_store, 0),
        )

        next_idx_a = jnp.where(is_a_le_b, idx_a + 1, idx_a)
        next_idx_b = jnp.where(is_a_le_b, idx_b, idx_b + 1)
        return next_idx_a, next_idx_b, out_ptr + 1

    idx_a, idx_b, out_ptr = jax.lax.while_loop(
        main_loop_cond, main_loop_body, initial_main_loop_state
    )

    initial_ak_loop_state = (idx_a, out_ptr)

    def ak_loop_cond(state):
        current_idx_a, _ = state
        return current_idx_a < a_end

    def ak_loop_body(state):
        current_idx_a, current_out_ptr = state
        val_to_store = pl.load(ak_ref, (current_idx_a,))
        val_casted = val_to_store.astype(merged_keys_ref.dtype)
        pl.store(
            merged_keys_ref,
            (pl.ds(current_out_ptr, 1),),
            jnp.expand_dims(val_casted, 0),
        )
        pl.store(
            merged_indices_ref,
            (pl.ds(current_out_ptr, 1),),
            jnp.expand_dims(current_idx_a, 0),
        )
        return current_idx_a + 1, current_out_ptr + 1

    idx_a, out_ptr = jax.lax.while_loop(ak_loop_cond, ak_loop_body, initial_ak_loop_state)

    initial_bk_loop_state = (idx_b, out_ptr)

    def bk_loop_cond(state):
        current_idx_b, _ = state
        return current_idx_b < b_end

    def bk_loop_body(state):
        current_idx_b, current_out_ptr = state
        val_to_store = pl.load(bk_ref, (current_idx_b,))
        val_casted = val_to_store.astype(merged_keys_ref.dtype)
        pl.store(
            merged_keys_ref,
            (pl.ds(current_out_ptr, 1),),
            jnp.expand_dims(val_casted, 0),
        )
        pl.store(
            merged_indices_ref,
            (pl.ds(current_out_ptr, 1),),
            jnp.expand_dims(n + current_idx_b, 0),
        )
        return current_idx_b + 1, current_out_ptr + 1

    jax.lax.while_loop(bk_loop_cond, bk_loop_body, initial_bk_loop_state)


@jax.jit
def merge_arrays_parallel(ak: jax.Array, bk: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    Merges two sorted JAX arrays using the parallel Merge Path Pallas kernel.
    """
    if ak.ndim != 1 or bk.ndim != 1:
        raise ValueError("Input arrays ak and bk must be 1D.")

    n, m = ak.shape[0], bk.shape[0]
    total_len = n + m
    if total_len == 0:
        key_dtype = jnp.result_type(ak.dtype, bk.dtype)
        return jnp.array([], dtype=key_dtype), jnp.array([], dtype=jnp.int32)

    key_dtype = jnp.result_type(ak.dtype, bk.dtype)
    out_keys_shape_dtype = jax.ShapeDtypeStruct((total_len,), key_dtype)
    out_idx_shape_dtype = jax.ShapeDtypeStruct((total_len,), jnp.int32)

    grid_size = (total_len + BLOCK_SIZE - 1) // BLOCK_SIZE

    return pl.pallas_call(
        merge_parallel_kernel,
        grid=(grid_size,),
        out_shape=(out_keys_shape_dtype, out_idx_shape_dtype),
    )(ak, bk)
