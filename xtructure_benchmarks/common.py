import argparse
import gc
import json
import os
import platform
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from rich.console import Console
from rich.table import Table

from xtructure import FieldDescriptor, xtructure_dataclass


def enable_jax_float64():
    """
    Enables JAX float64 mode (double precision).
    Must be called before any JAX operations.
    """
    from jax import config

    config.update("jax_enable_x64", True)


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
    assert "batch_sizes" in results and isinstance(results["batch_sizes"], list), (
        "results must contain a list 'batch_sizes'"
    )
    batch_sizes = results["batch_sizes"]
    assert "xtructure" in results and isinstance(results["xtructure"], dict), (
        "results must contain dict 'xtructure'"
    )
    assert "python" in results and isinstance(results["python"], dict), (
        "results must contain dict 'python'"
    )

    # Optional environment info
    if "environment" in results:
        assert isinstance(results["environment"], dict), "environment must be a dict"

    x_ops = set(results["xtructure"].keys())
    p_ops = set(results["python"].keys())
    assert x_ops == p_ops, (
        f"operation keys mismatch between xtructure and python: {x_ops} vs {p_ops}"
    )

    for op in x_ops:
        x_list = results["xtructure"][op]
        p_list = results["python"][op]
        assert isinstance(x_list, list) and isinstance(p_list, list), (
            f"'{op}' entries must be lists"
        )
        assert len(x_list) == len(batch_sizes), (
            f"xtructure['{op}'] length {len(x_list)} != len(batch_sizes) {len(batch_sizes)}"
        )
        assert len(p_list) == len(batch_sizes), (
            f"python['{op}'] length {len(p_list)} != len(batch_sizes) {len(batch_sizes)}"
        )

        def _validate_entry(e: Any) -> None:
            if isinstance(e, (int, float)):
                return
            assert isinstance(e, dict) and "median" in e and "iqr" in e, (
                "each entry must be a number or a dict with 'median' and 'iqr'"
            )

        for e in x_list:
            _validate_entry(e)
        for e in p_list:
            _validate_entry(e)


def _ops_stats(
    num_ops: int, durations: Iterable[float], peak_memories: Optional[Iterable[float]] = None
) -> Dict[str, float]:
    """Compute median/IQR/P99 of per-trial throughput and peak memory."""
    durations_arr = np.asarray(list(durations), dtype=np.float64)
    ops = num_ops / durations_arr

    median_ops = float(np.median(ops))
    q75, q25 = np.percentile(ops, [75, 25])
    # P99 throughput corresponds to the slowest 1% of trials,
    # but usually P99 latency is desired. Here we report P99 ops/sec (which is the 1st percentile of speed).
    # To be less confusing,
    # let's just store p99_ops which is the performance of the 99th percentile slowest run (1st percentile speed).
    p99_ops = float(np.percentile(ops, 1))

    stats = {"median": median_ops, "iqr": float(q75 - q25), "p99_ops": p99_ops}

    if peak_memories:
        mem_arr = np.asarray(list(peak_memories), dtype=np.float64)
        stats["peak_memory_median"] = float(np.median(mem_arr))

    return stats


def get_system_info() -> Dict[str, Any]:
    """Captures system environment information for reproducibility."""
    info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }
    try:
        import jax

        info["jax_version"] = jax.__version__
        info["jax_backend"] = jax.default_backend()
        devices = jax.devices()
        if devices:
            # Simplify device list to avoid clutter
            device_types = [f"{d.platform}:{d.device_kind}" for d in devices]
            info["jax_devices"] = device_types
    except ImportError:
        pass

    if hasattr(os, "getloadavg"):
        try:
            info["load_avg_1min"] = os.getloadavg()[0]
        except OSError:
            pass

    return info


def check_system_load(threshold: float = 2.0) -> None:
    """Warns if system load is high."""
    if hasattr(os, "getloadavg"):
        try:
            load = os.getloadavg()[0]
            if load > threshold:
                print(
                    f"\n⚠️  WARNING: High system load detected ({load:.2f} > {threshold}). Results may be noisy.\n"
                )
        except OSError:
            pass


@xtructure_dataclass
class BenchmarkValue:
    """
    A standard data structure for use across all benchmarks.
    It contains a mix of different data types and shapes to provide a
    representative workload.
    """

    id: FieldDescriptor.scalar(dtype=jnp.uint32)
    timestamp: FieldDescriptor.scalar(dtype=jnp.uint32)
    position: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))
    embedding: FieldDescriptor.tensor(dtype=jnp.float16, shape=(128,))


@dataclass(frozen=True, slots=True)
class PythonBenchmarkValue:
    """
    Python equivalent of BenchmarkValue for strict fairness.
    Using __slots__ for memory efficiency similar to how JAX arrays are packed,
    though Python objects still have overhead.
    """

    id: int
    timestamp: int
    position: Tuple[float, ...]
    embedding: Tuple[float, ...]

    def __hash__(self):
        # Mimic structural hashing
        return hash((self.id, self.timestamp, self.position))

    def __lt__(self, other):
        # For heap comparisons
        if not isinstance(other, PythonBenchmarkValue):
            return NotImplemented
        return self.id < other.id


def to_python_values(jax_values: BenchmarkValue) -> List[PythonBenchmarkValue]:
    """Converts JAX Struct-of-Arrays to Python List-of-Structs for fair comparison."""
    host_values = jax.device_get(jax_values)
    ids = host_values.id.tolist()
    timestamps = host_values.timestamp.tolist()
    positions = host_values.position.tolist()
    embeddings = host_values.embedding.tolist()

    return [
        PythonBenchmarkValue(id=i, timestamp=t, position=tuple(p), embedding=tuple(e))
        for i, t, p, e in zip(ids, timestamps, positions, embeddings)
    ]


def run_jax_trials(
    func: Callable[..., Any],
    trials: int = 10,
    warmup: int = 5,
    include_device_transfer: bool = False,
    args_supplier: Optional[Callable[[], Tuple[Any, ...]]] = None,
) -> Tuple[List[float], List[float]]:
    """
    JITs, warms up, and measures JAX function latency per trial.

    Returns:
        (durations, peak_memories) - peak_memories is currently a placeholder (0s) for JAX
    """

    jitted = jax.jit(func)

    def _invoke():
        args = args_supplier() if args_supplier else ()
        result = jitted(*args)
        jax.block_until_ready(result)
        if include_device_transfer:
            _ = jax.device_get(result)
        return result

    # Warmup to avoid counting compilation and stabilize caches
    # Interleave GC to ensure clean state
    for _ in range(max(1, warmup)):
        _invoke()
        gc.collect()

    durations: List[float] = []
    for _ in range(trials):
        # Ensure clean state before measurement
        gc.collect()

        start = time.perf_counter()
        _invoke()
        end = time.perf_counter()
        durations.append(end - start)

    # JAX memory tracking is complex and requires external tools usually.
    # For now, returning 0 to indicate "not measured" in this context.
    peak_memories = [0.0] * trials

    return durations, peak_memories


def run_python_trials(
    func: Callable[[], Any], trials: int = 10, warmup: int = 5, disable_gc: bool = False
) -> Tuple[List[float], List[float]]:
    """
    Runs a Python function multiple times.
    Splits timing and memory measurement to avoid observer effect overhead.
    """

    # Warmup
    for _ in range(max(1, warmup)):
        func()
        gc.collect()

    durations: List[float] = []

    # Phase 1: Latency Measurement (No tracemalloc overhead)
    gc_was_enabled = gc.isenabled()
    if disable_gc:
        gc.disable()

    try:
        for _ in range(trials):
            # If disable_gc is True, we force GC between trials to ensure fair start
            if disable_gc:
                if gc_was_enabled:
                    gc.enable()
                    gc.collect()
                    gc.disable()
            else:
                # If GC is enabled (strict mode), we still collect before run
                # to start from clean slate, but let GC run during execution.
                gc.collect()

            start = time.perf_counter()
            func()
            end = time.perf_counter()
            durations.append(end - start)
    finally:
        if disable_gc and gc_was_enabled:
            gc.enable()

    # Phase 2: Peak Memory Measurement
    # We run this separately because tracemalloc slows down execution significantly.
    # We can use fewer trials or the same number. Using same number for consistency.
    peak_memories: List[float] = []

    try:
        for _ in range(trials):
            gc.collect()

            tracemalloc.start()
            # We don't time this run
            func()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            peak_memories.append(peak / (1024 * 1024))  # Convert to MB

    except Exception:
        # Fallback if memory tracking fails or isn't supported
        peak_memories = [0.0] * trials

    return durations, peak_memories


def throughput_stats(num_ops: int, results: Tuple[List[float], List[float]]) -> Dict[str, float]:
    """Public helper to compute ops/sec and memory statistics."""
    durations, peak_memories = results
    return _ops_stats(num_ops, durations, peak_memories)


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
    table.add_column("P99 (Ops/Sec)", justify="right", style="dim cyan")
    table.add_column("Peak Mem (MB)", justify="right", style="bold red")

    batch_sizes = results.get("batch_sizes", [])
    xtructure_results = results.get("xtructure", {})
    python_results = results.get("python", {})

    operations = list(xtructure_results.keys())

    for i, size in enumerate(batch_sizes):
        for op in operations:
            op_name = op.replace("_ops_per_sec", "")

            # Add row for xtructure
            xtructure_data = xtructure_results.get(op, [])[i]
            xtructure_mem = "-"
            xtructure_p99 = "-"
            if isinstance(xtructure_data, dict):
                xtructure_perf = xtructure_data["median"]
                xtructure_iqr = xtructure_data["iqr"]
                if "p99_ops" in xtructure_data:
                    xtructure_p99 = human_format(xtructure_data["p99_ops"])
                if (
                    "peak_memory_median" in xtructure_data
                    and xtructure_data["peak_memory_median"] > 0
                ):
                    xtructure_mem = f"{xtructure_data['peak_memory_median']:.1f}"
            else:
                xtructure_perf = xtructure_data
                xtructure_iqr = 0

            table.add_row(
                f"{size:,}",
                op_name,
                "xtructure",
                human_format(xtructure_perf),
                f"±{human_format(xtructure_iqr)}",
                xtructure_p99,
                xtructure_mem,
            )

            # Add row for python
            python_data = python_results.get(op, [])[i]
            python_mem = "-"
            python_p99 = "-"
            if isinstance(python_data, dict):
                python_perf = python_data["median"]
                python_iqr = python_data["iqr"]
                if "p99_ops" in python_data:
                    python_p99 = human_format(python_data["p99_ops"])
                if "peak_memory_median" in python_data:
                    python_mem = f"{python_data['peak_memory_median']:.1f}"
            else:
                python_perf = python_data
                python_iqr = 0

            table.add_row(
                "",
                op_name,
                "python",
                human_format(python_perf),
                f"±{human_format(python_iqr)}",
                python_p99,
                python_mem,
            )
        if i < len(batch_sizes) - 1:
            table.add_row("", "", "", "", "", "", end_section=True)

    console.print(table)


def init_benchmark_results(batch_sizes: List[int]) -> Dict[str, Any]:
    check_system_load()
    return {
        "batch_sizes": batch_sizes,
        "xtructure": {},
        "python": {},
        "environment": get_system_info(),
    }


def save_and_print_results(results: Dict[str, Any], output_path: str, title: str) -> None:
    validate_results_schema(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"{title} saved to {output_path}")
    print_results_table(results, title)


def add_common_benchmark_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=["kernel", "e2e"], default="kernel")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="",
        help="Comma-separated batch sizes (e.g. 1024,4096,16384)",
    )


def parse_batch_sizes(batch_sizes_str: str) -> Optional[List[int]]:
    if not batch_sizes_str:
        return None
    return [int(x.strip()) for x in batch_sizes_str.split(",") if x.strip()]
