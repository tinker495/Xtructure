import jax.numpy as jnp
from jax import lax

from xtructure.core.xtructure_numpy.array_ops import _where_no_broadcast


def _get_sentinels(dtype):
    """Returns the min and max sentinel values for a given dtype."""
    finfo = jnp.finfo(dtype)
    return finfo.min, finfo.max


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
        cond_safe_a = jnp.asarray(is_a_safe, dtype=jnp.bool_)
        i_minus_one = jnp.asarray(i - 1)
        safe_a_idx = _where_no_broadcast(
            cond_safe_a,
            i_minus_one,
            jnp.zeros_like(i_minus_one),
        )
        a_val_loaded = a[safe_a_idx]
        min_val_array = jnp.full_like(a_val_loaded, min_val)
        a_val = _where_no_broadcast(cond_safe_a, a_val_loaded, min_val_array)

        cond_safe_b = jnp.asarray(is_b_safe, dtype=jnp.bool_)
        j_array = jnp.asarray(j)
        safe_b_idx = _where_no_broadcast(
            cond_safe_b,
            j_array,
            jnp.zeros_like(j_array),
        )
        b_val_loaded = b[safe_b_idx]
        max_val_array = jnp.full_like(b_val_loaded, max_val)
        b_val = _where_no_broadcast(cond_safe_b, b_val_loaded, max_val_array)

        # The condition for a valid partition from `a`'s perspective.
        # If `a[i-1] <= b[j]`, then `i` is a valid candidate, and we can
        # potentially take even more from `a`. So, we search in `[i, high]`.
        # Otherwise, `i` is too high, and we must search in `[low, i-1]`.
        is_partition_valid = a_val <= b_val
        cond_partition = jnp.asarray(is_partition_valid, dtype=jnp.bool_)
        new_low = _where_no_broadcast(cond_partition, jnp.asarray(i), jnp.asarray(low_i))
        new_high = _where_no_broadcast(
            cond_partition,
            jnp.asarray(high_i),
            i_minus_one,
        )
        return new_low, new_high

    # The loop terminates when low == high, and `final_low` is our desired `i`.
    final_low, _ = lax.while_loop(cond_fn, body_fn, (low, high))
    return final_low, k - final_low
