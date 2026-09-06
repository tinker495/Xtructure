from typing import List, Literal, Optional

import tyro

from xtructure import Stack
from xtructure_benchmarks.common import (
    PythonBenchmarkValue,
    parse_batch_sizes,
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


def main(
    mode: Literal["kernel", "e2e"] = "kernel",
    trials: int = 10,
    batch_sizes: str = "",
) -> None:
    """Run Stack benchmarks.

    Args:
        mode: Measure kernel execution or end-to-end transfer costs.
        trials: Number of timing trials.
        batch_sizes: Comma-separated batch sizes (e.g. 1024,4096,16384).
    """
    run_benchmarks(mode=mode, trials=trials, batch_sizes=parse_batch_sizes(batch_sizes))


if __name__ == "__main__":
    tyro.cli(main)
