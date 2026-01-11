"""Small helpers for BGPQ operations."""

import chex
import jax
import jax.numpy as jnp

from ..core import Xtructurable
from ._constants import SIZE_DTYPE, SORT_STABLE


def sort_arrays(k: chex.Array, v: Xtructurable):
    sorted_k, sorted_idx = jax.lax.sort_key_val(k, jnp.arange(k.shape[0]), is_stable=SORT_STABLE)
    sorted_v = v[sorted_idx]
    return sorted_k, sorted_v


@jax.jit
def _next(current, target):
    """
    Calculate the next index in the heap traversal path.
    Uses leading zero count (clz) for efficient binary tree navigation.
    This implementation handles the 0-indexed heap structure by temporarily
    converting to 1-based indices for the underlying bitwise logic.

    Args:
        current: Current index in the heap
        target: Target index to reach

    Returns:
        Next index in the path from current to target
    """
    current_1based = current.astype(SIZE_DTYPE) + 1
    target_1based = target.astype(SIZE_DTYPE) + 1

    clz_current = jax.lax.clz(current_1based)
    clz_target = jax.lax.clz(target_1based)
    shift_amount = clz_current - clz_target - 1

    next_index_1based = target_1based >> shift_amount
    return next_index_1based - 1
