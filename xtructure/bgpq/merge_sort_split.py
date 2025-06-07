from typing import Tuple  # Added for type hinting

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def merge_sort_split_kernel(
    ak_ref,
    bk_ref,  # Input refs
    res_key0_ref,
    res_idx0_ref,
    res_key1_ref,
    res_idx1_ref,  # Output refs
):
    """
    Merge and split two sorted arrays while maintaining their relative order.
    This version is Pallas-compliant: writes to output refs, no return.
    Outputs are:
    - res_key0: First N keys of the merged and sorted array.
    - res_idx0: Corresponding original indices for res_key0.
    - res_key1: Remaining keys of the merged and sorted array.
    - res_idx1: Corresponding original indices for res_key1.
    """
    ak_val, bk_val = ak_ref[...], bk_ref[...]

    # n_split determines the size of the first chunk of the output.
    # It's based on the length of the original 'ak' array's last dimension.
    n_split = ak_val.shape[-1]

    # Concatenate input arrays. For 1D inputs, this concatenates along axis 0.
    # If inputs were N-D, one might need to specify axis, e.g., axis=-1.
    # For this problem, inputs ak, bk are treated as primary sequences.
    key_concat = jnp.concatenate([ak_val, bk_val])

    # Create a payload of original indices (0 to L+M-1 for concatenated array)
    # Ensure dtype is suitable for indices.
    indices_payload = jnp.arange(key_concat.shape[0], dtype=jnp.int32)

    # Sort the concatenated keys, carrying along the original indices payload.
    # jax.lax.sort_key_val sorts along the last dimension by default.
    sorted_key_full, sorted_idx_full = jax.lax.sort_key_val(key_concat, indices_payload)

    # Write results to the output references
    res_key0_ref[...] = sorted_key_full[:n_split]
    res_idx0_ref[...] = sorted_idx_full[:n_split]
    res_key1_ref[...] = sorted_key_full[n_split:]
    res_idx1_ref[...] = sorted_idx_full[n_split:]
    # No return statement, as per Pallas convention for kernels writing to refs


@jax.jit
def merge_sort_split_idx(
    ak: jax.Array, bk: jax.Array
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    # Determine the length of the first part of the split (from ak)
    # and the second part (from bk). Assumes working with the last dimension.
    len_ak_part = ak.shape[-1]
    len_bk_part = bk.shape[-1]  # This is the length of the "rest" after splitting by len_ak_part

    # Determine the data type for the output keys (can be a promotion of input types)
    key_dtype = jnp.result_type(ak.dtype, bk.dtype)

    # Define shapes and dtypes for the four outputs
    # Output 1: Keys for the first part (length from ak)
    shape_key0 = jax.ShapeDtypeStruct(shape=ak.shape[:-1] + (len_ak_part,), dtype=key_dtype)
    # Output 2: Indices for the first part
    shape_idx0 = jax.ShapeDtypeStruct(shape=ak.shape[:-1] + (len_ak_part,), dtype=jnp.int32)
    # Output 3: Keys for the second part (length from bk)
    shape_key1 = jax.ShapeDtypeStruct(shape=bk.shape[:-1] + (len_bk_part,), dtype=key_dtype)
    # Output 4: Indices for the second part
    shape_idx1 = jax.ShapeDtypeStruct(shape=bk.shape[:-1] + (len_bk_part,), dtype=jnp.int32)

    return pl.pallas_call(
        merge_sort_split_kernel,
        out_shape=(shape_key0, shape_idx0, shape_key1, shape_idx1),  # Tuple of ShapeDtypeStructs
    )(ak, bk)


# New kernel and wrapper function for merging with loops and returning indices
def merge_indices_kernel_loop(ak_ref, bk_ref, merged_keys_ref, merged_indices_ref):
    """
    Pallas kernel to merge two sorted arrays (ak, bk) and write the indices
    of the merged elements (relative to a conceptual [ak, bk] concatenation)
    into merged_indices_ref. Uses explicit loops and Pallas memory operations.
    """
    # ak, bk = ak_ref[...], bk_ref[...] # Removed: We'll use pl.load directly on refs
    n = ak_ref.shape[0]  # Assuming shape is available on Ref, or pass n, m as args
    m = bk_ref.shape[0]

    # Helper functions for lax.cond need to use pl.store for side-effects
    # They will capture n and the refs: ak_ref, bk_ref, merged_indices_ref
    def true_branch_body_fn(cond_operands):
        (
            current_idx_a,
            current_idx_b,
            current_out_ptr_val,
            val_a_to_store,
            _,
            _,  # Corresponds to loop_ak_ref from cond
            merged_keys_ref_from_cond,  # Newly added
            merged_indices_ref_from_cond,  # Corresponds to loop_merged_indices_ref from cond
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
            _,  # Corresponds to loop_ak_ref from cond
            merged_keys_ref_from_cond,  # Newly added
            merged_indices_ref_from_cond,  # Corresponds to loop_merged_indices_ref from cond
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

    # Main merge loop using jax.lax.while_loop
    # Loop state: (idx_a, idx_b, out_ptr, ak_ref, bk_ref, merged_keys_ref, merged_indices_ref)
    # merged_indices_ref is also passed to ensure it's part of the state if needed by JAX,
    # though pl.store uses the captured one.
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

        # Pass val_a and val_b to be stored by the conditional branches
        # Also pass relevant refs (loop_ak_ref, loop_merged_keys_ref, loop_merged_indices_ref)
        updated_idx_a, updated_idx_b = jax.lax.cond(
            pred,
            true_branch_body_fn,
            false_branch_body_fn,
            # Operands to cond: current indices, out_ptr, actual values, and relevant refs
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
    # Unpack state. Refs are carried through.
    (
        idx_a,
        idx_b,
        out_ptr,
        _,
        _,
        final_loop_merged_keys_ref,
        final_loop_merged_indices_ref,
    ) = final_state_after_main_loop

    # Loop for remaining elements from ak
    # Loop state: (current_idx_a, current_out_ptr, ak_ref_ignored, merged_keys_ref, merged_indices_ref_loop_state)
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

    # Loop for remaining elements from bk
    # Loop state: (current_idx_b, current_out_ptr, bk_ref_ignored, merged_keys_ref, merged_indices_ref_loop_state)
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
        # Store index from bk, offset by n
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
    # Assuming ak and bk are 1D arrays.
    # If you need to handle batched inputs, this function (or the kernel) would need modification,
    # e.g., by using jax.vmap on this function or making the kernel batch-aware.
    if ak.ndim != 1 or bk.ndim != 1:
        raise ValueError("Input arrays ak and bk must be 1D.")

    n = ak.shape[0]
    m = bk.shape[0]

    # The output will be an array of indices, with length n + m.
    # Indices are integers, jnp.int32 is usually sufficient.

    # Determine the common dtype for keys by promoting types of ak and bk.
    key_dtype = jnp.result_type(ak.dtype, bk.dtype)
    out_keys_shape_dtype = jax.ShapeDtypeStruct((n + m,), key_dtype)
    out_idx_shape_dtype = jax.ShapeDtypeStruct((n + m,), jnp.int32)

    return pl.pallas_call(
        merge_indices_kernel_loop, out_shape=(out_keys_shape_dtype, out_idx_shape_dtype)
    )(ak, bk)


# ### Implementation of Merge Path Algorithm ###


def _get_sentinels(dtype):
    """Returns the min and max sentinel values for a given dtype."""
    if jnp.issubdtype(dtype, jnp.integer):
        return jnp.iinfo(dtype).min, jnp.iinfo(dtype).max
    if jnp.issubdtype(dtype, jnp.floating):
        return dtype.type(-jnp.inf), dtype.type(jnp.inf)
    raise TypeError(f"Unsupported dtype for sentinel values: {dtype}")


def binary_search_partition(k, a, b):
    """
    Finds the partition of k elements between sorted arrays a and b.

    This function implements the core logic of the "Merge Path" algorithm. It
    uses binary search to find a split point (i, j) such that i elements from
    array `a` and j elements from array `b` constitute the first k elements
    of the merged array. Thus, i + j = k.

    The search finds an index `i` in `[0, n]` that satisfies the condition:
    `a[i-1] <= b[j]` and `b[j-1] <= a[i]`, where `j = k - i`. These checks
    define a valid merge partition. The binary search below finds the
    largest `i` that satisfies `a[i-1] <= b[k-i]`.

    Args:
      k: The total number of elements in the target partition (the "diagonal"
         of the merge path grid).
      a: A sorted JAX array or a Pallas Ref to one.
      b: A sorted JAX array or a Pallas Ref to one.

    Returns:
      A tuple (i, j) where i is the number of elements to take from a and j
      is the number of elements from b, satisfying i + j = k.
    """
    n = a.shape[0]
    m = b.shape[0]

    # The number of elements from `a`, `i`, must be in the range [low, high].
    low = jnp.maximum(0, k - m)
    high = jnp.minimum(n, k)

    # Binary search for the correct partition index `i`. We are looking for the
    # largest `i` in `[low, high]` such that `a[i-1] <= b[k-i]`.
    def cond_fn(state):
        low_i, high_i = state
        return low_i < high_i

    def body_fn(state):
        low_i, high_i = state
        # Bias the midpoint to the right to ensure the loop terminates when
        # searching for the "last true" condition.
        i = low_i + (high_i - low_i + 1) // 2
        j = k - i

        min_val, max_val = _get_sentinels(a.dtype)
        is_a_safe = i > 0
        is_b_safe = j < m

        # A more robust way to handle conditional loading in Pallas to avoid
        # the `scf.yield` lowering error.
        # 1. Select a safe index to load from (0 if out of bounds).
        # 2. Perform the load unconditionally.
        # 3. Use `where` to replace the loaded value with a sentinel if the
        #    original index was out of bounds.
        safe_a_idx = jnp.where(is_a_safe, i - 1, 0)
        a_val_loaded = pl.load(a, (safe_a_idx,))
        a_val = jnp.where(is_a_safe, a_val_loaded, min_val)

        safe_b_idx = jnp.where(is_b_safe, j, 0)
        b_val_loaded = pl.load(b, (safe_b_idx,))
        b_val = jnp.where(is_b_safe, b_val_loaded, max_val)

        # The condition for a valid partition from `a`'s perspective.
        # If `a[i-1] <= b[j]`, then `i` is a valid candidate, and we can
        # potentially take even more from `a`. So, we search in `[i, high]`.
        # Otherwise, `i` is too high, and we must search in `[low, i-1]`.
        is_partition_valid = a_val <= b_val
        new_low = jnp.where(is_partition_valid, i, low_i)
        new_high = jnp.where(is_partition_valid, high_i, i - 1)
        return new_low, new_high

    # The loop terminates when low == high, and `final_low` is our desired `i`.
    final_low, _ = jax.lax.while_loop(cond_fn, body_fn, (low, high))
    return final_low, k - final_low


def merge_parallel_kernel(ak_ref, bk_ref, merged_keys_ref, merged_indices_ref):
    """
    Pallas kernel that merges two sorted arrays in parallel using the
    Merge Path algorithm for block-level partitioning.
    """
    # Each block processes a fixed-size chunk of the output array.
    BLOCK_SIZE = 256  # The number of output elements each block handles.
    block_idx = pl.program_id(axis=0)

    n, m = ak_ref.shape[0], bk_ref.shape[0]
    total_len = n + m

    # 1. Block-level Partitioning using Merge Path
    # Determine the start and end of the output slice for this block.
    k_start = block_idx * BLOCK_SIZE
    k_end = jnp.minimum(k_start + BLOCK_SIZE, total_len)

    # Use the binary search helper to find where this slice begins and ends
    # in the input arrays. This is the "coarse-grained" partitioning.
    a_start, b_start = binary_search_partition(k_start, ak_ref, bk_ref)
    a_end, b_end = binary_search_partition(k_end, ak_ref, bk_ref)

    # 2. Local Merge
    # This block now performs a serial merge on the determined slices:
    # ak_ref[a_start:a_end] and bk_ref[b_start:b_end].
    # The logic is identical to `merge_indices_kernel_loop` but operates on
    # these smaller, independent slices.

    # State for the loops: (current_idx_a, current_idx_b, current_out_ptr)
    initial_main_loop_state = (a_start, b_start, k_start)

    def main_loop_cond(state):
        idx_a, idx_b, _ = state
        return jnp.logical_and(idx_a < a_end, idx_b < b_end)

    def main_loop_body(state):
        idx_a, idx_b, out_ptr = state
        val_a = pl.load(ak_ref, (idx_a,))
        val_b = pl.load(bk_ref, (idx_b,))
        is_a_le_b = val_a <= val_b

        # Store the smaller value and its original index.
        key_to_store = jnp.where(is_a_le_b, val_a, val_b)
        # The original index is offset by `n` if it comes from `bk`.
        idx_to_store = jnp.where(is_a_le_b, idx_a, n + idx_b)

        key_casted = key_to_store.astype(merged_keys_ref.dtype)
        pl.store(merged_keys_ref, (out_ptr,), key_casted)
        pl.store(merged_indices_ref, (out_ptr,), idx_to_store)

        # Advance the index of the array from which the value was taken.
        next_idx_a = jnp.where(is_a_le_b, idx_a + 1, idx_a)
        next_idx_b = jnp.where(is_a_le_b, idx_b, idx_b + 1)
        return next_idx_a, next_idx_b, out_ptr + 1

    idx_a, idx_b, out_ptr = jax.lax.while_loop(
        main_loop_cond, main_loop_body, initial_main_loop_state
    )

    # Cleanup loop for remaining elements from ak's slice.
    initial_ak_loop_state = (idx_a, out_ptr)

    def ak_loop_cond(state):
        current_idx_a, _ = state
        return current_idx_a < a_end

    def ak_loop_body(state):
        current_idx_a, current_out_ptr = state
        val_to_store = pl.load(ak_ref, (current_idx_a,))
        val_casted = val_to_store.astype(merged_keys_ref.dtype)
        pl.store(merged_keys_ref, (current_out_ptr,), val_casted)
        pl.store(merged_indices_ref, (current_out_ptr,), current_idx_a)
        return current_idx_a + 1, current_out_ptr + 1

    idx_a, out_ptr = jax.lax.while_loop(ak_loop_cond, ak_loop_body, initial_ak_loop_state)

    # Cleanup loop for remaining elements from bk's slice.
    initial_bk_loop_state = (idx_b, out_ptr)

    def bk_loop_cond(state):
        current_idx_b, _ = state
        return current_idx_b < b_end

    def bk_loop_body(state):
        current_idx_b, current_out_ptr = state
        val_to_store = pl.load(bk_ref, (current_idx_b,))
        val_casted = val_to_store.astype(merged_keys_ref.dtype)
        pl.store(merged_keys_ref, (current_out_ptr,), val_casted)
        pl.store(merged_indices_ref, (current_out_ptr,), n + current_idx_b)
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
        # Handle empty case to avoid grid computation errors.
        key_dtype = jnp.result_type(ak.dtype, bk.dtype)
        return jnp.array([], dtype=key_dtype), jnp.array([], dtype=jnp.int32)

    # Determine the common dtype for keys by promoting types of ak and bk.
    key_dtype = jnp.result_type(ak.dtype, bk.dtype)
    out_keys_shape_dtype = jax.ShapeDtypeStruct((total_len,), key_dtype)
    out_idx_shape_dtype = jax.ShapeDtypeStruct((total_len,), jnp.int32)

    # Each block handles a fixed number of output elements.
    BLOCK_SIZE = 32
    # Grid size is the number of blocks needed to cover the whole output.
    grid_size = (total_len + BLOCK_SIZE - 1) // BLOCK_SIZE

    return pl.pallas_call(
        merge_parallel_kernel,
        grid=(grid_size,),
        out_shape=(out_keys_shape_dtype, out_idx_shape_dtype),
    )(ak, bk)


if __name__ == "__main__":
    # Example from the original file for merge_sort_split_idx
    x_orig = jnp.array([1, 2, 3, 4, 5, 6, 7, 8])
    y_orig = jnp.array([1, 2, 3, 4, 5, 6, 7, 8])
    # print("Original merge_sort_split_idx output (structure might differ from request):")
    # This call might error if the out_shape of merge_sort_split_idx is indeed misconfigured for its kernel
    # The following line is commented out because merge_sort_split_kernel uses jnp.concatenate,
    # which has limitations in Pallas Triton lowering for 1D arrays not ending in shape [..., 1].
    # print(merge_sort_split_idx(x_orig, y_orig))

    print("\nTesting new merge_arrays_indices_loop:")
    # Test case 1: Identical arrays (as in user's initial example)
    a1 = jnp.array([1, 2, 3, 4])
    b1 = jnp.array([1, 2, 3, 4])
    # Concatenated: [1,2,3,4, | 1,2,3,4] (indices 0-3 for a1, 4-7 for b1)
    # Merged values: [1,1,2,2,3,3,4,4]
    # Expected indices: [0 (a1[0]), 4 (b1[0]), 1 (a1[1]), 5 (b1[1]), 2 (a1[2]), 6 (b1[2]), 3 (a1[3]), 7 (b1[3])]
    merged_keys1, merged_indices1 = merge_arrays_indices_loop(a1, b1)
    print(f"a1: {a1}")
    print(f"b1: {b1}")
    print(f"Merged keys for a1, b1: {merged_keys1}")
    print(f"Merged indices for a1, b1: {merged_indices1}")
    # To verify, let's reconstruct the sorted array using these indices
    concatenated1 = jnp.concatenate([a1, b1])
    print(f"Reconstructed sorted array (using indices): {concatenated1[merged_indices1]}")
    assert jnp.array_equal(merged_keys1, concatenated1[merged_indices1])

    # Test case 2: Interleaved arrays
    a2 = jnp.array([1, 5, 9])
    b2 = jnp.array([2, 6, 10])
    # Concatenated: [1,5,9, | 2,6,10] (indices 0-2 for a2, 3-5 for b2)
    # Merged values: [1,2,5,6,9,10]
    # Expected indices: [0 (a2[0]), 3 (b2[0]), 1 (a2[1]), 4 (b2[1]), 2 (a2[2]), 5 (b2[2])]
    merged_keys2, merged_indices2 = merge_arrays_indices_loop(a2, b2)
    print(f"\na2: {a2}")
    print(f"b2: {b2}")
    print(f"Merged keys for a2, b2: {merged_keys2}")
    print(f"Merged indices for a2, b2: {merged_indices2}")
    concatenated2 = jnp.concatenate([a2, b2])
    print(f"Reconstructed sorted array (using indices): {concatenated2[merged_indices2]}")
    assert jnp.array_equal(merged_keys2, concatenated2[merged_indices2])

    # Test case 3: One array exhausted first
    a3 = jnp.array([1, 2])
    b3 = jnp.array([3, 4, 5, 6])
    # Concatenated: [1,2, | 3,4,5,6] (indices 0-1 for a3, 2-5 for b3)
    # Merged values: [1,2,3,4,5,6]
    # Expected indices: [0,1,2,3,4,5] (0,1 from a3; 2+0, 2+1, 2+2, 2+3 from b3)
    merged_keys3, merged_indices3 = merge_arrays_indices_loop(a3, b3)
    print(f"\na3: {a3}")
    print(f"b3: {b3}")
    print(f"Merged keys for a3, b3: {merged_keys3}")
    print(f"Merged indices for a3, b3: {merged_indices3}")
    concatenated3 = jnp.concatenate([a3, b3])
    print(f"Reconstructed sorted array (using indices): {concatenated3[merged_indices3]}")
    assert jnp.array_equal(merged_keys3, concatenated3[merged_indices3])

    # Test case 4: Empty array
    a4 = jnp.array([])
    b4 = jnp.array([1, 2, 3])
    merged_keys4a, merged_indices4a = merge_arrays_indices_loop(a4, b4)  # ak empty
    # Expected: [0+0, 0+1, 0+2] -> [0,1,2] (relative to b4, offset by n=0)
    print(f"\na4: {a4}")
    print(f"b4: {b4}")
    print(f"Merged keys for a4, b4: {merged_keys4a}")
    print(f"Merged indices for a4, b4: {merged_indices4a}")
    concatenated4a = jnp.concatenate([a4, b4])
    # Check if concatenated4a is empty, if so, merged_keys4a should also be empty.
    # Otherwise, proceed with reconstruction and assertion.
    if concatenated4a.size > 0:
        print(f"Reconstructed sorted array (using indices): {concatenated4a[merged_indices4a]}")
        assert jnp.array_equal(merged_keys4a, concatenated4a[merged_indices4a])
    else:
        print("Reconstructed sorted array (using indices): [] (Inputs were empty)")
        assert merged_keys4a.size == 0

    merged_keys4b, merged_indices4b = merge_arrays_indices_loop(b4, a4)  # bk empty
    # Expected: [0,1,2] (relative to b4)
    print(f"\nb4: {b4}")
    print(f"a4: {a4}")
    print(f"Merged keys for b4, a4: {merged_keys4b}")
    print(f"Merged indices for b4, a4: {merged_indices4b}")
    concatenated4b = jnp.concatenate([b4, a4])
    if concatenated4b.size > 0:
        print(f"Reconstructed sorted array (using indices): {concatenated4b[merged_indices4b]}")
        assert jnp.array_equal(merged_keys4b, concatenated4b[merged_indices4b])
    else:
        print("Reconstructed sorted array (using indices): [] (Inputs were empty)")
        assert merged_keys4b.size == 0

    # Test case 5: Arrays with duplicate values across them
    a5 = jnp.array([10, 20, 30])
    b5 = jnp.array([10, 25, 30])
    # Concat: [10,20,30 | 10,25,30] (idx 0-2 for a5, 3-5 for b5)
    # Merge: [10(a5), 10(b5), 20(a5), 25(b5), 30(a5), 30(b5)] (stable sort behavior: take from first array if equal)
    # Expect: [0 (a5[0]), 3 (b5[0]), 1 (a5[1]), 4 (b5[1]), 2 (a5[2]), 5 (b5[2])]
    merged_keys5, merged_indices5 = merge_arrays_indices_loop(a5, b5)
    print(f"\na5: {a5}")
    print(f"b5: {b5}")
    print(f"Merged keys for a5, b5: {merged_keys5}")
    print(f"Merged indices for a5, b5: {merged_indices5}")
    concatenated5 = jnp.concatenate([a5, b5])
    print(f"Reconstructed sorted array (using indices): {concatenated5[merged_indices5]}")
    assert jnp.array_equal(merged_keys5, concatenated5[merged_indices5])

    print("\n\n--- Testing new merge_arrays_indices_loop with random values and timing ---")
    import time

    import jax.random as jr

    def verify_and_time_merge(key, size_ak, size_bk, dtype=jnp.int32):
        print(f"\nTesting with ak_size={size_ak}, bk_size={size_bk}, dtype={dtype}")
        key_ak, key_bk = jr.split(key)

        # Generate random arrays
        if jnp.issubdtype(dtype, jnp.integer):
            ak_rand = jr.randint(
                key_ak, (size_ak,), minval=0, maxval=max(1, size_ak * 10), dtype=dtype
            )
            bk_rand = jr.randint(
                key_bk, (size_bk,), minval=0, maxval=max(1, size_bk * 10), dtype=dtype
            )
        elif jnp.issubdtype(dtype, jnp.floating):
            # For float types, use uniform. maxval should be > minval for uniform.
            # Using a simple range, can be adjusted if a specific float range is needed.
            ak_rand = jr.uniform(
                key_ak, (size_ak,), dtype=dtype, minval=0.0, maxval=float(max(1, size_ak * 10))
            )
            bk_rand = jr.uniform(
                key_bk, (size_bk,), dtype=dtype, minval=0.0, maxval=float(max(1, size_bk * 10))
            )
        else:
            raise TypeError(f"Unsupported dtype for random generation: {dtype}")

        # Sort them to be used as inputs
        ak_sorted = jnp.sort(ak_rand)
        bk_sorted = jnp.sort(bk_rand)

        if size_ak < 10 and size_bk < 10:  # Print smaller arrays for inspection
            print(f"  Sorted ak: {ak_sorted}")
            print(f"  Sorted bk: {bk_sorted}")

        # --- Correctness Verification ---
        # Select implementation to test based on a condition, e.g., size
        # For this example, let's test both and maybe a new parallel one.
        implementations_to_test = {
            "pallas_loop": merge_arrays_indices_loop,
            "pallas_parallel": merge_arrays_parallel,
        }

        reference_merged_jax = jnp.sort(jnp.concatenate([ak_sorted, bk_sorted]))
        reference_merged_jax.block_until_ready()
        concatenated_inputs = jnp.concatenate([ak_sorted, bk_sorted])

        all_passed = True
        for name, merge_fn in implementations_to_test.items():
            print(f"  Verifying implementation: {name}")
            try:
                merged_keys, merged_indices = merge_fn(ak_sorted, bk_sorted)
                merged_keys.block_until_ready()
                merged_indices.block_until_ready()

                reconstructed = concatenated_inputs[merged_indices]
                reconstructed.block_until_ready()

                # Verify merged keys directly and via reconstructed indices
                assert jnp.array_equal(merged_keys, reference_merged_jax)
                assert jnp.array_equal(reconstructed, reference_merged_jax)
                print("    ✅ Correctness check PASSED.")
            except AssertionError as e:
                all_passed = False
                print(f"    ❌ Correctness check FAILED for {name}.")
                if size_ak < 20 and size_bk < 20:
                    print(f"      Pallas merged keys:   {merged_keys}")
                    print(f"      Pallas reconstructed: {reconstructed}")
                    print(f"      JAX reference sorted: {reference_merged_jax}")
                    print(f"      Pallas indices:       {merged_indices}")
                # Optional: raise e to stop on failure
                print(f"    ❌ {e}")
            except Exception as e:
                all_passed = False
                print(f"    ❌ An exception occurred during {name} execution: {e}")

        # --- Timing Comparison ---
        if not all_passed:
            print("  Skipping timing due to correctness failure.")
            return

        # JAX baseline version timing (jnp.sort on concatenated array)
        # Warm-up run for JAX baseline
        _ = jnp.sort(jnp.concatenate([ak_sorted, bk_sorted])).block_until_ready()

        start_time_jax = time.perf_counter()
        for _ in range(10):
            jax_output_timing = jnp.sort(jnp.concatenate([ak_sorted, bk_sorted]))
        jax_output_timing.block_until_ready()
        end_time_jax = time.perf_counter()
        jax_time = (end_time_jax - start_time_jax) / 10
        print(f"  ⏱️ JAX jnp.sort(concatenate) avg time:      {jax_time*1000:.4f} ms")

        # Pallas versions timing
        for name, merge_fn in implementations_to_test.items():
            # Warm-up run for Pallas
            _keys, _indices = merge_fn(ak_sorted, bk_sorted)
            _keys.block_until_ready()
            _indices.block_until_ready()

            start_time_pallas = time.perf_counter()
            for _ in range(10):
                pallas_keys_timing, pallas_indices_timing = merge_fn(ak_sorted, bk_sorted)
            pallas_keys_timing.block_until_ready()
            pallas_indices_timing.block_until_ready()
            end_time_pallas = time.perf_counter()
            pallas_time = (end_time_pallas - start_time_pallas) / 10
            print(f"  ⏱️ Pallas '{name}' avg time: {pallas_time*1000:.4f} ms")

    # Run tests with different sizes and types
    master_key = jr.PRNGKey(42)

    key, subkey = jr.split(master_key)
    verify_and_time_merge(subkey, size_ak=8, size_bk=8, dtype=jnp.int32)

    key, subkey = jr.split(key)
    verify_and_time_merge(subkey, size_ak=200, size_bk=200, dtype=jnp.int32)

    key, subkey = jr.split(key)
    verify_and_time_merge(subkey, size_ak=1000, size_bk=1000, dtype=jnp.int32)

    key, subkey = jr.split(key)
    verify_and_time_merge(subkey, size_ak=5000, size_bk=5000, dtype=jnp.int32)

    key, subkey = jr.split(key)
    verify_and_time_merge(subkey, size_ak=200, size_bk=200, dtype=jnp.float32)

    key, subkey = jr.split(key)
    verify_and_time_merge(subkey, size_ak=1000, size_bk=1000, dtype=jnp.float32)

    key, subkey = jr.split(key)
    verify_and_time_merge(subkey, size_ak=5000, size_bk=5000, dtype=jnp.float32)
