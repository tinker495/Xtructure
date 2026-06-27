import argparse
from typing import List, Optional

from xtructure import Stack
from xtructure_benchmarks.common import (
    PythonBenchmarkValue,
    run_linear_container_benchmarks,
)


def _python_push(items: List[PythonBenchmarkValue]):
    stack = []
    stack.extend(items)
    return stack


def _python_pop(items: List[PythonBenchmarkValue], batch_size: int):
    stack = list(items)
    results = []
    for _ in range(batch_size):
        if stack:
            results.append(stack.pop())
    return results


def run_benchmarks(mode: str = "kernel", trials: int = 10, batch_sizes: Optional[List[int]] = None):
    """Runs the Stack benchmarks and saves the results."""
    run_linear_container_benchmarks(
        container_name="Stack",
        container_class=Stack,
        add_method="push",
        remove_method="pop",
        python_add=_python_push,
        python_remove=_python_pop,
        mode=mode,
        trials=trials,
        batch_sizes=batch_sizes,
        output_path="xtructure_benchmarks/results/stack_results.json",
        title="Stack Performance Results",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stack benchmarks")
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
