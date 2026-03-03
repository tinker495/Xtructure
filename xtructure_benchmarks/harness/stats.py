from __future__ import annotations

from typing import Iterable

import numpy as np


def summarize(values: Iterable[float], prefix: str) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {f"{prefix}_median": 0.0, f"{prefix}_iqr": 0.0, f"{prefix}_p99": 0.0}
    q75, q25 = np.percentile(arr, [75, 25])
    return {
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_p99": float(np.percentile(arr, 99)),
    }


def throughput(items: int, durations_s: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(durations_s), dtype=np.float64)
    ops = items / np.maximum(arr, 1e-12)
    return summarize(ops, "items_per_sec")
