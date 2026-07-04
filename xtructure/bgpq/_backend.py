"""BGPQ Merge Backend Policy."""

from __future__ import annotations

from collections.abc import Callable
from typing import Tuple

import jax

from .merge_split import merge_arrays_parallel, merge_sort_split_idx

MergeArrayBackend = Callable[[jax.Array, jax.Array], Tuple[jax.Array, jax.Array]]
PARALLEL_MERGE_MIN_ELEMENTS = 1024


def merge_arrays_adaptive(ak: jax.Array, bk: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Merge arrays with a shape-specialized backend choice.

    The Pallas Merge Path kernel wins once the launch overhead is amortized,
    but ``lax.sort_key_val`` is consistently faster for the small BGPQ branch
    sizes that dominate unit workloads and shallow heaps.  The array shape is
    static under JIT, so this Python branch is resolved during tracing rather
    than becoming data-dependent device control flow.
    """
    total_len = ak.shape[0] + bk.shape[0]
    if total_len < PARALLEL_MERGE_MIN_ELEMENTS:
        return merge_sort_split_idx(ak, bk)
    return merge_arrays_parallel(ak, bk)


merge_array_backend = (
    merge_sort_split_idx if jax.default_backend() == "tpu" else merge_arrays_adaptive
)

__all__ = [
    "MergeArrayBackend",
    "PARALLEL_MERGE_MIN_ELEMENTS",
    "merge_array_backend",
    "merge_arrays_adaptive",
]
