"""Small helpers for BGPQ operations."""

import os

import chex
import jax
import jax.numpy as jnp

from ..core import Xtructurable
from ._constants import SIZE_DTYPE, SORT_STABLE


def _use_kv_backend(backend: str, batch_size: int, context: str = "BACKEND") -> bool:
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
        f"Invalid XTRUCTURE_BGPQ_MERGE_VALUE_{context}. Expected off/auto/parallel/kv_parallel."
    )


def sort_arrays(k: chex.Array, v: Xtructurable):
    sorted_k, sorted_idx = jax.lax.sort_key_val(
        k, jnp.arange(k.shape[0]), is_stable=SORT_STABLE
    )
    sorted_v = v[sorted_idx]
    return sorted_k, sorted_v


def _scatter_update_rows(
    operand: chex.Array,
    indices: chex.Array,
    updates: chex.Array,
    *,
    indices_are_sorted: bool,
) -> chex.Array:
    scatter_indices = indices.astype(jnp.int32)[:, None]
    update_window_dims = tuple(range(1, updates.ndim))
    dimension_numbers = jax.lax.ScatterDimensionNumbers(
        update_window_dims=update_window_dims,
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    return jax.lax.scatter(
        operand,
        scatter_indices,
        updates,
        dimension_numbers,
        indices_are_sorted=indices_are_sorted,
        unique_indices=True,
        mode=jax.lax.GatherScatterMode.FILL_OR_DROP,
    )


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
