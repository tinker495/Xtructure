"""Shared constants and backend selection for BGPQ."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp

from .merge_split import merge_arrays_parallel, merge_sort_split_idx

SORT_STABLE = True  # Use stable sorting to maintain insertion order for equal keys
SIZE_DTYPE = jnp.uint32

# Backend selection
#
# `merge_arrays_parallel` uses Pallas (mosaic_gpu) and is significantly faster on
# GPU. For CPU/TPU we default to the pure XLA sort-based implementation.
#
# You can force a backend via `XTRUCTURE_BGPQ_MERGE_BACKEND`:
# - `parallel` / `pallas`
# - `sort` / `xla`
_DEFAULT_MERGE_BACKEND = "parallel" if jax.default_backend() == "gpu" else "sort"
_MERGE_BACKEND = os.environ.get("XTRUCTURE_BGPQ_MERGE_BACKEND", _DEFAULT_MERGE_BACKEND).lower()

if _MERGE_BACKEND in {"sort", "xla"}:
    merge_array_backend = merge_sort_split_idx
elif _MERGE_BACKEND in {"parallel", "pallas"}:
    merge_array_backend = merge_arrays_parallel
else:
    raise ValueError(
        "Invalid XTRUCTURE_BGPQ_MERGE_BACKEND. " "Expected one of: sort, xla, parallel, pallas"
    )
