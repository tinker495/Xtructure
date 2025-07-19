import time
from typing import Any, Callable, Dict

import jax
import jax.numpy as jnp
from rich.console import Console
from rich.table import Table

from xtructure import FieldDescriptor, xtructure_dataclass


def human_format(num):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude]
    )


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


def jax_timer(func: Callable[[], Any]) -> float:
    """
    A timer for JAX functions that ensures accurate measurement by waiting for
    computation to complete.

    Args:
        func: The JAX function to time.

    Returns:
        The execution time in seconds.
    """
    # JIT compile the function first if it's not already
    jitted_func = jax.jit(func)

    # Run once to warm up and ensure compilation is done
    result = jitted_func()
    jax.block_until_ready(result)

    # Time the actual execution
    start_time = time.perf_counter()
    result = jitted_func()
    jax.block_until_ready(result)
    end_time = time.perf_counter()

    return end_time - start_time


def python_timer(func: Callable[[], Any]) -> float:
    """
    A simple timer for standard Python functions.

    Args:
        func: The Python function to time.

    Returns:
        The execution time in seconds.
    """
    start_time = time.perf_counter()
    func()
    end_time = time.perf_counter()
    return end_time - start_time


def print_results_table(results: Dict[str, Any], title: str):
    """
    Displays benchmark results in a formatted table using the rich library.
    """
    console = Console()
    table = Table(title=title, show_header=True, header_style="bold magenta")

    table.add_column("Batch Size", justify="right", style="cyan")
    table.add_column("Operation", style="green")
    table.add_column("Implementation", style="yellow")
    table.add_column("Operations per Second", justify="right", style="bold blue")

    batch_sizes = results.get("batch_sizes", [])
    xtructure_results = results.get("xtructure", {})
    python_results = results.get("python", {})

    operations = list(xtructure_results.keys())

    for i, size in enumerate(batch_sizes):
        for op in operations:
            op_name = op.replace("_ops_per_sec", "")

            # Add row for xtructure
            xtructure_perf = xtructure_results.get(op, [])[i]
            table.add_row(f"{size:,}", op_name, "xtructure", human_format(xtructure_perf))

            # Add row for python
            python_perf = python_results.get(op, [])[i]
            table.add_row(
                "", op_name, "python", human_format(python_perf)
            )  # Don't repeat batch size
        if i < len(batch_sizes) - 1:
            table.add_row("", "", "", "", end_section=True)

    console.print(table)
