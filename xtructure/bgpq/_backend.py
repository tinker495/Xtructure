"""BGPQ Merge Backend Policy."""

from __future__ import annotations

from collections.abc import Callable
from typing import Tuple

import jax

from .merge_split import merge_arrays_parallel, merge_sort_split_idx

MergeArrayBackend = Callable[[jax.Array, jax.Array], Tuple[jax.Array, jax.Array]]


def select_merge_array_backend(backend_name: str | None = None) -> MergeArrayBackend:
    """Return the merge backend for a JAX backend name.

    TPU currently uses the sort/split implementation because the parallel
    Pallas Merge Path kernel is not TPU-ready.  GPU/CPU keep the parallel
    backend selected from the existing GPU microbench result: float32 trials=10000
    showed merge_arrays_parallel ~6-14x faster than loop/split at n in
    [1024, 16384].
    """
    backend = jax.default_backend() if backend_name is None else backend_name
    if backend == "tpu":
        return merge_sort_split_idx
    return merge_arrays_parallel


merge_array_backend = select_merge_array_backend()

__all__ = ["MergeArrayBackend", "merge_array_backend", "select_merge_array_backend"]
