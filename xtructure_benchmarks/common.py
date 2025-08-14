import time
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from rich.console import Console
from rich.table import Table

from xtructure import FieldDescriptor, xtructure_dataclass


def human_format(num, pos=None):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude]
    )


def validate_results_schema(results: Dict[str, Any]) -> None:
    """
    Validates that the results dictionary has consistent shapes and symmetric ops.

    Requirements:
      - keys: batch_sizes, xtructure, python
      - operations present under xtructure and python are identical
      - for every operation, list lengths equal len(batch_sizes)
      - entries are either numbers or dicts with {median, iqr}
    Raises AssertionError on violation.
    """
    assert isinstance(results, dict), "results must be a dict"
    assert "batch_sizes" in results and isinstance(
        results["batch_sizes"], list
    ), "results must contain a list 'batch_sizes'"
    batch_sizes = results["batch_sizes"]
    assert "xtructure" in results and isinstance(
        results["xtructure"], dict
    ), "results must contain dict 'xtructure'"
    assert "python" in results and isinstance(
        results["python"], dict
    ), "results must contain dict 'python'"

    x_ops = set(results["xtructure"].keys())
    p_ops = set(results["python"].keys())
    assert (
        x_ops == p_ops
    ), f"operation keys mismatch between xtructure and python: {x_ops} vs {p_ops}"

    for op in x_ops:
        x_list = results["xtructure"][op]
        p_list = results["python"][op]
        assert isinstance(x_list, list) and isinstance(
            p_list, list
        ), f"'{op}' entries must be lists"
        assert len(x_list) == len(
            batch_sizes
        ), f"xtructure['{op}'] length {len(x_list)} != len(batch_sizes) {len(batch_sizes)}"
        assert len(p_list) == len(
            batch_sizes
        ), f"python['{op}'] length {len(p_list)} != len(batch_sizes) {len(batch_sizes)}"

        def _validate_entry(e: Any) -> None:
            if isinstance(e, (int, float)):
                return
            assert (
                isinstance(e, dict) and "median" in e and "iqr" in e
            ), "each entry must be a number or a dict with 'median' and 'iqr'"

        for e in x_list:
            _validate_entry(e)
        for e in p_list:
            _validate_entry(e)


@xtructure_dataclass
class BenchmarkValue:
    """
    A standard data structure for use across all benchmarks.
    It contains a mix of different data types and shapes to provide a
    representative workload.
    """

    id: FieldDescriptor[jnp.uint32]
    timestamp: FieldDescriptor[jnp.uint32]
    position: FieldDescriptor[jnp.float32, (3,)]
    embedding: FieldDescriptor[jnp.float16, (128,)]


def jax_timer(func: Callable[[], Any], trials: int = 10) -> Tuple[float, float]:
    """
    A timer for JAX functions that ensures accurate measurement by waiting for
    computation to complete. Runs multiple trials and returns median and IQR.

    Args:
        func: The JAX function to time.
        trials: Number of timing trials to run (default 10).

    Returns:
        Tuple of (median_time, iqr_time) in seconds.
    """
    # JIT compile the function first if it's not already
    jitted_func = jax.jit(func)

    # Run once to warm up and ensure compilation is done
    result = jitted_func()
    jax.block_until_ready(result)

    # Time multiple trials
    times = []
    for _ in range(trials):
        start_time = time.perf_counter()
        result = jitted_func()
        jax.block_until_ready(result)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    times = np.array(times)
    median_time = np.median(times)
    q75, q25 = np.percentile(times, [75, 25])
    iqr_time = q75 - q25

    return median_time, iqr_time


def python_timer(func: Callable[[], Any], trials: int = 10) -> Tuple[float, float]:
    """
    A timer for standard Python functions with multiple trials.

    Args:
        func: The Python function to time.
        trials: Number of timing trials to run (default 10).

    Returns:
        Tuple of (median_time, iqr_time) in seconds.
    """
    times = []
    for _ in range(trials):
        start_time = time.perf_counter()
        func()
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    times = np.array(times)
    median_time = np.median(times)
    q75, q25 = np.percentile(times, [75, 25])
    iqr_time = q75 - q25

    return median_time, iqr_time


def print_results_table(results: Dict[str, Any], title: str):
    """
    Displays benchmark results in a formatted table using the rich library.
    """
    console = Console()
    table = Table(title=title, show_header=True, header_style="bold magenta")

    table.add_column("Batch Size", justify="right", style="cyan")
    table.add_column("Operation", style="green")
    table.add_column("Implementation", style="yellow")
    table.add_column("Ops/Sec (Median)", justify="right", style="bold blue")
    table.add_column("IQR", justify="right", style="dim blue")

    batch_sizes = results.get("batch_sizes", [])
    xtructure_results = results.get("xtructure", {})
    python_results = results.get("python", {})

    operations = list(xtructure_results.keys())

    for i, size in enumerate(batch_sizes):
        for op in operations:
            op_name = op.replace("_ops_per_sec", "")

            # Add row for xtructure
            xtructure_data = xtructure_results.get(op, [])[i]
            if isinstance(xtructure_data, dict):
                xtructure_perf = xtructure_data["median"]
                xtructure_iqr = xtructure_data["iqr"]
            else:
                # Backward compatibility with old format
                xtructure_perf = xtructure_data
                xtructure_iqr = 0

            table.add_row(
                f"{size:,}",
                op_name,
                "xtructure",
                human_format(xtructure_perf),
                f"±{human_format(xtructure_iqr)}",
            )

            # Add row for python
            python_data = python_results.get(op, [])[i]
            if isinstance(python_data, dict):
                python_perf = python_data["median"]
                python_iqr = python_data["iqr"]
            else:
                # Backward compatibility with old format
                python_perf = python_data
                python_iqr = 0

            table.add_row(
                "", op_name, "python", human_format(python_perf), f"±{human_format(python_iqr)}"
            )
        if i < len(batch_sizes) - 1:
            table.add_row("", "", "", "", "", end_section=True)

    console.print(table)
