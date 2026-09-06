from collections import deque
from typing import List, Literal, Optional

import tyro

from xtructure import Queue
from xtructure_benchmarks.common import (
    PythonBenchmarkValue,
    parse_batch_sizes,
    run_linear_container_benchmarks,
)


def _python_enqueue(items: List[PythonBenchmarkValue]):
    d = deque()
    d.extend(items)
    return d


def _python_dequeue(items: List[PythonBenchmarkValue], batch_size: int):
    d = deque(items)
    results = []
    for _ in range(batch_size):
        if d:
            results.append(d.popleft())
    return results


def run_benchmarks(mode: str = "kernel", trials: int = 10, batch_sizes: Optional[List[int]] = None):
    """Runs the Queue benchmarks and saves the results."""
    run_linear_container_benchmarks(
        container_name="Queue",
        container_class=Queue,
        add_method="enqueue",
        remove_method="dequeue",
        python_add=_python_enqueue,
        python_remove=_python_dequeue,
        mode=mode,
        trials=trials,
        batch_sizes=batch_sizes,
        output_path="xtructure_benchmarks/results/queue_results.json",
        title="Queue Performance Results",
    )


def main(
    mode: Literal["kernel", "e2e"] = "kernel",
    trials: int = 10,
    batch_sizes: str = "",
) -> None:
    """Run Queue benchmarks.

    Args:
        mode: Measure kernel execution or end-to-end transfer costs.
        trials: Number of timing trials.
        batch_sizes: Comma-separated batch sizes (e.g. 1024,4096,16384).
    """
    run_benchmarks(mode=mode, trials=trials, batch_sizes=parse_batch_sizes(batch_sizes))


if __name__ == "__main__":
    tyro.cli(main)
