import argparse
from collections import deque
from typing import List, Optional

from xtructure import Queue
from xtructure_benchmarks.common import (
    PythonBenchmarkValue,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Queue benchmarks")
    parser.add_argument("--mode", choices=["kernel", "e2e"], default="kernel")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="",
        help="Comma-separated batch sizes (e.g. 1024,4096,16384)",
    )
    args = parser.parse_args()

    batch_sizes_arg: Optional[List[int]] = None
    if args.batch_sizes:
        batch_sizes_arg = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]

    run_benchmarks(mode=args.mode, trials=args.trials, batch_sizes=batch_sizes_arg)
