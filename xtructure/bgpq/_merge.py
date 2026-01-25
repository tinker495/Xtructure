"""Merge/split operations used by BGPQ."""

import os
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp

from ..core import Xtructurable
from ..core import xtructure_numpy as xnp
from ._constants import merge_array_backend
from .merge_split.parallel import merge_arrays_parallel_kv


def _batch_size(values: Xtructurable) -> int:
    leaf = jax.tree_util.tree_leaves(values)[0]
    return int(leaf.shape[0])


def _gather_sorted_values(av: Xtructurable, bv: Xtructurable, sorted_idx: chex.Array):
    reorder_mode = os.environ.get("XTRUCTURE_BGPQ_VALUE_REORDER", "gather").strip().lower()
    if reorder_mode in {"concat", "concat_gather"}:
        val = xnp.concatenate([av, bv], axis=0)
        return val[sorted_idx]
    if reorder_mode not in {"gather", "direct"}:
        raise ValueError("Invalid XTRUCTURE_BGPQ_VALUE_REORDER. Expected gather or concat.")

    n = _batch_size(av)
    m = _batch_size(bv)
    if n == 0:
        return bv[sorted_idx]
    if m == 0:
        return av[sorted_idx]

    idx = jnp.asarray(sorted_idx, dtype=jnp.int32)
    idx_a = jnp.clip(idx, 0, n - 1)
    idx_b = jnp.clip(idx - n, 0, m - 1)
    take_a = idx < n
    val_a = av[idx_a]
    val_b = bv[idx_b]

    def _select_leaf(leaf_a, leaf_b):
        cond = take_a
        if leaf_a.ndim > 1:
            cond = cond.reshape((cond.shape[0],) + (1,) * (leaf_a.ndim - 1))
        return jnp.where(cond, leaf_a, leaf_b)

    return jax.tree_util.tree_map(_select_leaf, val_a, val_b)


def _use_kv_backend(backend: str, batch_size: int) -> bool:
    backend = backend.strip().lower()
    if backend in {"", "off", "0", "false", "none"}:
        return False
    if backend in {"on", "true", "parallel", "pallas", "kv", "kv_parallel"}:
        return True
    if backend in {"auto", "kv_auto"}:
        threshold = os.environ.get("XTRUCTURE_BGPQ_MERGE_VALUE_AUTO_MIN_BATCH", "0")
        try:
            threshold_value = int(threshold)
        except ValueError as exc:
            raise ValueError(
                "XTRUCTURE_BGPQ_MERGE_VALUE_AUTO_MIN_BATCH must be an integer."
            ) from exc
        return threshold_value > 0 and batch_size >= threshold_value
    raise ValueError(
        "Invalid XTRUCTURE_BGPQ_MERGE_VALUE_BACKEND_SORTSPLIT. "
        "Expected off/auto/parallel/kv_parallel."
    )


@jax.jit
def merge_sort_split(
    ak: chex.Array, av: Xtructurable, bk: chex.Array, bv: Xtructurable
) -> tuple[chex.Array, Xtructurable, chex.Array, Xtructurable]:
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
    n = ak.shape[0]  # type: ignore
    merge_backend = os.environ.get("XTRUCTURE_BGPQ_MERGE_VALUE_BACKEND_SORTSPLIT")
    if merge_backend is None:
        merge_backend = os.environ.get("XTRUCTURE_BGPQ_MERGE_VALUE_BACKEND", "")
    use_parallel_values = _use_kv_backend(merge_backend, int(n))

    if use_parallel_values and jax.default_backend() == "gpu":
        sorted_key, sorted_val = merge_arrays_parallel_kv(ak, av, bk, bv)
        sorted_key_any = cast(Any, sorted_key)
        sorted_val_any = cast(Any, sorted_val)
        return (
            cast(chex.Array, sorted_key_any[:n]),  # type: ignore[index]
            cast(Xtructurable, sorted_val_any[:n]),
            cast(chex.Array, sorted_key_any[n:]),  # type: ignore[index]
            cast(Xtructurable, sorted_val_any[n:]),
        )

    sorted_key, sorted_idx = merge_array_backend(ak, bk)
    sorted_key_any = cast(Any, sorted_key)
    sorted_val = cast(Xtructurable, _gather_sorted_values(av, bv, sorted_idx))
    sorted_val_any = cast(Any, sorted_val)
    return (
        cast(chex.Array, sorted_key_any[:n]),  # type: ignore[index]
        cast(Xtructurable, sorted_val_any[:n]),
        cast(chex.Array, sorted_key_any[n:]),  # type: ignore[index]
        cast(Xtructurable, sorted_val_any[n:]),
    )
