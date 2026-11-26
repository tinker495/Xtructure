"""Merge/split operations used by BGPQ."""

import jax
import jax.numpy as jnp

from ..core.protocol import Xtructurable
from ..core.xtructure_numpy.dataclass_ops.batch_ops import concat
from ._backend import PARALLEL_MERGE_MIN_ELEMENTS, merge_array_backend
from .merge_split.parallel import RANKED_MERGE_MAX_ROW, merge_scalar_payloads


def _index_merge(ak, av, bk, bv):
    n = ak.shape[0]
    val = concat([av, bv], axis=0)
    sorted_key, sorted_idx = merge_array_backend(ak, bk)
    sorted_val = val[sorted_idx]
    return sorted_key[:n], sorted_val[:n], sorted_key[n:], sorted_val[n:]


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
    n, m = ak.shape[0], bk.shape[0]
    if (
        n
        and m
        and n + m >= PARALLEL_MERGE_MIN_ELEMENTS
        and max(n, m) <= RANKED_MERGE_MAX_ROW
        and ak.dtype == bk.dtype
        and ak.dtype in (jnp.float16, jnp.float32)
        and av.shape == (n,)
        and bv.shape == (m,)
    ):
        a_leaves, a_tree = jax.tree.flatten(av)
        b_leaves, b_tree = jax.tree.flatten(bv)
        # Limit register pressure and keep the generic gather for structured or
        # mixed-dtype payloads. The search queues use two or four scalar fields.
        if (
            a_tree == b_tree
            and 0 < len(a_leaves) <= 4
            and all(
                a.shape == (n,)
                and b.shape == (m,)
                and a.dtype == b.dtype
                and a.dtype in (jnp.uint8, jnp.uint32, jnp.float16, jnp.float32)
                for a, b in zip(a_leaves, b_leaves)
            )
        ):
            return jax.lax.platform_dependent(
                ak, av, bk, bv, cuda=merge_scalar_payloads, default=_index_merge
            )
    return _index_merge(ak, av, bk, bv)
