"""Merge/split operations used by BGPQ."""

import jax

from ..core.protocol import Xtructurable
from ..core.xtructure_numpy.dataclass_ops.batch_ops import concat
from ._backend import merge_array_backend


@jax.jit
def merge_sort_split(
    ak: jax.Array, av: Xtructurable, bk: jax.Array, bv: Xtructurable
) -> tuple[jax.Array, Xtructurable, jax.Array, Xtructurable]:
    """
    Merge and split two sorted arrays while maintaining their relative order.
    This is a key operation for maintaining heap property in batched operations.

    Args:
        ak: First array of keys
        av: First array of values
        bk: Second array of keys
        bv: Second array of values

    Returns:
        tuple containing:
            - First half of merged and sorted keys
            - First half of corresponding values
            - Second half of merged and sorted keys
            - Second half of corresponding values
    """
    n = ak.shape[-1]  # size of group
    val = concat([av, bv], axis=0)
    sorted_key, sorted_idx = merge_array_backend(ak, bk)
    sorted_val = val[sorted_idx]
    return sorted_key[:n], sorted_val[:n], sorted_key[n:], sorted_val[n:]
