"""Shared constants and backend selection for BGPQ."""

import jax

from .merge_split import merge_arrays_parallel, merge_sort_split_idx

SORT_STABLE = True  # Use stable sorting to maintain insertion order for equal keys

# Decision note: GPU microbench (float32, trials=10000) shows merge_arrays_parallel
# is ~6-14x faster than loop/split at n in [1024, 16384], so we default to it.
# TODO: Make merge_arrays_parallel for TPU.
merge_array_backend = (
    merge_sort_split_idx if jax.default_backend() == "tpu" else merge_arrays_parallel
)
