from typing import Tuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def merge_indices_kernel_loop(ak_ref, bk_ref, merged_keys_ref, merged_indices_ref):
    """
    Pallas kernel to merge two sorted arrays (ak, bk) and write the indices
    of the merged elements (relative to a conceptual [ak, bk] concatenation)
    into merged_indices_ref. Uses explicit loops and Pallas memory operations.
    """
    n = ak_ref.shape[0]
    m = bk_ref.shape[0]

    def true_branch_body_fn(cond_operands):
        (
            current_idx_a,
            current_idx_b,
            current_out_ptr_val,
            val_a_to_store,
            _,
            _,
            merged_keys_ref_from_cond,
            merged_indices_ref_from_cond,
        ) = cond_operands
        val_a_casted = val_a_to_store.astype(merged_keys_ref_from_cond.dtype)
        pl.store(
            merged_keys_ref_from_cond,
            (current_out_ptr_val,),
            val_a_casted,
            eviction_policy="evict_last",
        )
        pl.store(
            merged_indices_ref_from_cond,
            (current_out_ptr_val,),
            current_idx_a,
            eviction_policy="evict_last",
        )
        return current_idx_a + 1, current_idx_b

    def false_branch_body_fn(cond_operands):
        (
            current_idx_a,
            current_idx_b,
            current_out_ptr_val,
            _,
            val_b_to_store,
            _,
            merged_keys_ref_from_cond,
            merged_indices_ref_from_cond,
        ) = cond_operands
        val_b_casted = val_b_to_store.astype(merged_keys_ref_from_cond.dtype)
        pl.store(
            merged_keys_ref_from_cond,
            (current_out_ptr_val,),
            val_b_casted,
            eviction_policy="evict_last",
        )
        pl.store(
            merged_indices_ref_from_cond,
            (current_out_ptr_val,),
            n + current_idx_b,
            eviction_policy="evict_last",
        )
        return current_idx_a, current_idx_b + 1

    initial_main_loop_state = (0, 0, 0, ak_ref, bk_ref, merged_keys_ref, merged_indices_ref)

    def main_loop_condition(state):
        idx_a, idx_b, _, _, _, _, _ = state
        return jnp.logical_and(idx_a < n, idx_b < m)

    def main_loop_body(state):
        (
            idx_a,
            idx_b,
            out_ptr,
            loop_ak_ref,
            loop_bk_ref,
            loop_merged_keys_ref,
            loop_merged_indices_ref,
        ) = state
        val_a = pl.load(loop_ak_ref, (idx_a,))
        val_b = pl.load(loop_bk_ref, (idx_b,))
        pred = val_a <= val_b

        updated_idx_a, updated_idx_b = jax.lax.cond(
            pred,
            true_branch_body_fn,
            false_branch_body_fn,
            (
                idx_a,
                idx_b,
                out_ptr,
                val_a,
                val_b,
                loop_ak_ref,
                loop_merged_keys_ref,
                loop_merged_indices_ref,
            ),
        )
        return (
            updated_idx_a,
            updated_idx_b,
            out_ptr + 1,
            loop_ak_ref,
            loop_bk_ref,
            loop_merged_keys_ref,
            loop_merged_indices_ref,
        )

    final_state_after_main_loop = jax.lax.while_loop(
        main_loop_condition, main_loop_body, initial_main_loop_state
    )

    (
        idx_a,
        idx_b,
        out_ptr,
        _,
        _,
        final_loop_merged_keys_ref,
        final_loop_merged_indices_ref,
    ) = final_state_after_main_loop

    initial_ak_loop_state = (
        idx_a,
        out_ptr,
        ak_ref,
        final_loop_merged_keys_ref,
        final_loop_merged_indices_ref,
    )

    def ak_loop_condition(state):
        current_idx_a, _, _, _, _ = state
        return current_idx_a < n

    def ak_loop_body(state):
        (
            current_idx_a,
            current_out_ptr,
            loop_ak_ref,
            loop_merged_keys_ref,
            loop_merged_indices_ref,
        ) = state
        val_to_store = pl.load(loop_ak_ref, (current_idx_a,))
        val_casted = val_to_store.astype(loop_merged_keys_ref.dtype)
        pl.store(
            loop_merged_keys_ref,
            (current_out_ptr,),
            val_casted,
            eviction_policy="evict_last",
        )
        pl.store(
            loop_merged_indices_ref, (current_out_ptr,), current_idx_a, eviction_policy="evict_last"
        )
        return (
            current_idx_a + 1,
            current_out_ptr + 1,
            loop_ak_ref,
            loop_merged_keys_ref,
            loop_merged_indices_ref,
        )

    final_state_after_ak_loop = jax.lax.while_loop(
        ak_loop_condition, ak_loop_body, initial_ak_loop_state
    )
    (
        idx_a,
        out_ptr,
        _,
        final_loop_merged_keys_ref,
        final_loop_merged_indices_ref,
    ) = final_state_after_ak_loop

    initial_bk_loop_state = (
        idx_b,
        out_ptr,
        bk_ref,
        final_loop_merged_keys_ref,
        final_loop_merged_indices_ref,
    )

    def bk_loop_condition(state):
        current_idx_b, _, _, _, _ = state
        return current_idx_b < m

    def bk_loop_body(state):
        (
            current_idx_b,
            current_out_ptr,
            loop_bk_ref,
            loop_merged_keys_ref,
            loop_merged_indices_ref,
        ) = state
        val_to_store = pl.load(loop_bk_ref, (current_idx_b,))
        val_casted = val_to_store.astype(loop_merged_keys_ref.dtype)
        pl.store(
            loop_merged_keys_ref,
            (current_out_ptr,),
            val_casted,
            eviction_policy="evict_last",
        )
        pl.store(
            loop_merged_indices_ref,
            (current_out_ptr,),
            n + current_idx_b,
            eviction_policy="evict_last",
        )
        return (
            current_idx_b + 1,
            current_out_ptr + 1,
            loop_bk_ref,
            loop_merged_keys_ref,
            loop_merged_indices_ref,
        )

    jax.lax.while_loop(bk_loop_condition, bk_loop_body, initial_bk_loop_state)


@jax.jit
def merge_arrays_indices_loop(ak: jax.Array, bk: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    Merges two sorted JAX arrays ak and bk using a loop-based Pallas kernel
    and returns a tuple containing:
    - merged_keys: The sorted merged array of keys.
    - merged_indices: An array of indices representing the merged order.
                      The indices refer to the positions in a conceptual concatenation [ak, bk].
    """
    if ak.ndim != 1 or bk.ndim != 1:
        raise ValueError("Input arrays ak and bk must be 1D.")

    n = ak.shape[0]
    m = bk.shape[0]

    key_dtype = jnp.result_type(ak.dtype, bk.dtype)
    out_keys_shape_dtype = jax.ShapeDtypeStruct((n + m,), key_dtype)
    out_idx_shape_dtype = jax.ShapeDtypeStruct((n + m,), jnp.int32)

    return pl.pallas_call(
        merge_indices_kernel_loop, out_shape=(out_keys_shape_dtype, out_idx_shape_dtype)
    )(ak, bk)
