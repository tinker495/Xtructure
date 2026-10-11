from typing import Tuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pltriton

from xtructure.core.xtructure_numpy.array_ops import _where_no_broadcast

from .common import binary_search_partition

BLOCK_SIZE = 64


def _merge_interleaved_kernel(ak_ref, bk_ref, merged_keys_ref, merged_indices_ref):
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
        val_a = ak_ref[idx_a]
        val_b = bk_ref[idx_b]
        is_a_le_b = val_a <= val_b

        cond = jnp.asarray(is_a_le_b, dtype=jnp.bool_)

        target_key_dtype = jnp.result_type(val_a.dtype, val_b.dtype)
        val_a_cast = jnp.asarray(val_a, dtype=target_key_dtype)
        val_b_cast = jnp.asarray(val_b, dtype=target_key_dtype)
        key_to_store = _where_no_broadcast(cond, val_a_cast, val_b_cast)

        idx_true = jnp.asarray(idx_a)
        idx_dtype = idx_true.dtype
        idx_false = jnp.asarray(n + idx_b, dtype=idx_dtype)
        idx_to_store = _where_no_broadcast(cond, idx_true, idx_false)

        key_casted = key_to_store.astype(merged_keys_ref.dtype)
        merged_keys_ref[out_ptr] = key_casted
        merged_indices_ref[out_ptr] = idx_to_store

        next_idx_a = _where_no_broadcast(
            cond,
            jnp.asarray(idx_a + 1, dtype=idx_dtype),
            jnp.asarray(idx_a, dtype=idx_dtype),
        )
        next_idx_b = _where_no_broadcast(
            cond,
            jnp.asarray(idx_b, dtype=idx_dtype),
            jnp.asarray(idx_b + 1, dtype=idx_dtype),
        )
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
        val_to_store = ak_ref[current_idx_a]
        val_casted = val_to_store.astype(merged_keys_ref.dtype)
        merged_keys_ref[current_out_ptr] = val_casted
        merged_indices_ref[current_out_ptr] = current_idx_a
        return current_idx_a + 1, current_out_ptr + 1

    idx_a, out_ptr = jax.lax.while_loop(ak_loop_cond, ak_loop_body, initial_ak_loop_state)

    initial_bk_loop_state = (idx_b, out_ptr)

    def bk_loop_cond(state):
        current_idx_b, _ = state
        return current_idx_b < b_end

    def bk_loop_body(state):
        current_idx_b, current_out_ptr = state
        val_to_store = bk_ref[current_idx_b]
        val_casted = val_to_store.astype(merged_keys_ref.dtype)
        merged_keys_ref[current_out_ptr] = val_casted
        merged_indices_ref[current_out_ptr] = n + current_idx_b
        return current_idx_b + 1, current_out_ptr + 1

    jax.lax.while_loop(bk_loop_cond, bk_loop_body, initial_bk_loop_state)


def merge_parallel_kernel(ak_ref, bk_ref, merged_keys_ref, merged_indices_ref):
    n, m = ak_ref.shape[0], bk_ref.shape[0]
    if not n or not m:
        return _merge_interleaved_kernel(ak_ref, bk_ref, merged_keys_ref, merged_indices_ref)

    def concatenate(b_first):
        lane = pl.program_id(0) * BLOCK_SIZE + jnp.arange(BLOCK_SIZE)
        valid = lane < n + m
        a_index = lane - b_first.astype(jnp.int32) * m
        b_index = lane - (~b_first).astype(jnp.int32) * n
        from_a = jnp.where(b_first, lane >= m, lane < n)
        a_value = pltriton.load(ak_ref.at[a_index], mask=valid & from_a, other=0)
        b_value = pltriton.load(bk_ref.at[b_index], mask=valid & ~from_a, other=0)
        keys = jnp.where(from_a, a_value, b_value).astype(merged_keys_ref.dtype)
        indices = jnp.where(from_a, a_index, n + b_index).astype(jnp.int32)
        pltriton.store(merged_keys_ref.at[lane], keys, mask=valid)
        pltriton.store(merged_indices_ref.at[lane], indices, mask=valid)

    def merge(_):
        _merge_interleaved_kernel(ak_ref, bk_ref, merged_keys_ref, merged_indices_ref)

    # Only A-first concatenation permits equality, preserving stable ties.
    # These device-local branches avoid a host predicate readback.
    a_first = ak_ref[n - 1] <= bk_ref[0]
    b_first = bk_ref[m - 1] < ak_ref[0]
    jax.lax.cond(a_first | b_first, concatenate, merge, operand=b_first)


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
