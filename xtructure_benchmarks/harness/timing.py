from __future__ import annotations

import time
from typing import Any, Callable

import jax
import jax.numpy as jnp


def estimate_bytes(tree: Any) -> int:
    leaves = jax.tree_util.tree_leaves(tree)
    total = 0
    for leaf in leaves:
        if hasattr(leaf, "size") and hasattr(leaf, "dtype"):
            total += int(leaf.size) * int(jnp.dtype(leaf.dtype).itemsize)
    return total


def timed_trials(
    fn: Callable[[], Any],
    mode: str,
    transfer_policy: str,
    warmup_iters: int,
    measure_iters: int,
) -> tuple[list[float], float, float]:
    for _ in range(max(1, warmup_iters)):
        out = fn()
        jax.block_until_ready(out)
        if mode in {"transfer", "e2e"} and transfer_policy != "none":
            _ = jax.device_get(out)

    durations: list[float] = []
    transferred_samples: list[float] = []
    for _ in range(measure_iters):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)

        host_out = None
        if mode in {"transfer", "e2e"} and transfer_policy != "none":
            host_out = jax.device_get(out)

        t1 = time.perf_counter()
        durations.append(t1 - t0)

        # Keep byte estimation outside the timed region to avoid skewing e2e/transfer timing.
        if host_out is not None:
            transferred_samples.append(float(estimate_bytes(host_out)))
        else:
            transferred_samples.append(0.0)

    bytes_per_iter = float(sum(transferred_samples) / max(1, len(transferred_samples)))
    bytes_per_sec = (
        bytes_per_iter / max(1e-12, float(sum(durations) / max(1, len(durations))))
        if durations
        else 0.0
    )
    return durations, bytes_per_iter, bytes_per_sec
