import gc
import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

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

    # Optional environment info
    if "environment" in results:
        assert isinstance(results["environment"], dict), "environment must be a dict"

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


def _ops_stats(num_ops: int, durations: Iterable[float]) -> Dict[str, float]:
    """Compute median/IQR/P99 throughput from per-trial durations."""
    durations_arr = np.asarray(list(durations), dtype=np.float64)
    ops = num_ops / durations_arr

    median_ops = float(np.median(ops))
    q75, q25 = np.percentile(ops, [75, 25])
    p99_ops = float(np.percentile(ops, 1))

    return {"median": median_ops, "iqr": float(q75 - q25), "p99_ops": p99_ops}


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
) -> List[float]:
    """JIT, warm up, and measure JAX function latency per trial."""

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

    return durations


def run_python_trials(
    func: Callable[[], Any], trials: int = 10, warmup: int = 5, disable_gc: bool = False
) -> List[float]:
    """Run a Python function multiple times and return per-trial durations."""

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

    return durations


def throughput_stats(num_ops: int, durations: Iterable[float]) -> Dict[str, float]:
    """Public helper to compute ops/sec statistics."""
    return _ops_stats(num_ops, durations)


def run_linear_container_benchmarks(
    *,
    container_name: str,
    container_class: Any,
    add_method: str,
    remove_method: str,
    python_add: Callable[[List[PythonBenchmarkValue]], Any],
    python_remove: Callable[[List[PythonBenchmarkValue], int], Any],
    mode: str = "kernel",
    trials: int = 10,
    batch_sizes: Optional[List[int]] = None,
    output_path: str,
    title: str,
) -> None:
    """Run the shared Queue/Stack benchmark shape."""
    batch_sizes = batch_sizes or [2**10, 2**12, 2**14]

    check_system_load()

    results: Dict[str, Any] = {
        "batch_sizes": batch_sizes,
        "xtructure": {},
        "python": {},
        "environment": get_system_info(),
    }
    max_size = int(max(batch_sizes) * 2)

    print(f"Running {container_name} Benchmarks...")
    try:
        print(f"JAX backend: {jax.default_backend()}")
        print("JAX devices:", ", ".join([d.platform + ":" + d.device_kind for d in jax.devices()]))
    except Exception:
        pass

    add_key = f"{add_method}_ops_per_sec"
    remove_key = f"{remove_method}_ops_per_sec"

    for batch_size in batch_sizes:
        print(f"  Batch Size: {batch_size}")
        key = jax.random.PRNGKey(batch_size)
        values_device = BenchmarkValue.random(shape=(batch_size,), key=key)
        values_host = jax.device_get(values_device)

        precomputed_items = to_python_values(values_device)

        def materialize_items(include_preprocessing: bool):
            return to_python_values(values_device) if include_preprocessing else precomputed_items

        container = container_class.build(max_size=max_size, value_class=BenchmarkValue)

        def add_args_supplier():
            return (jax.device_put(values_host),) if mode == "e2e" else (values_device,)

        add_durations = run_jax_trials(
            lambda batch: getattr(container, add_method)(batch),
            trials=trials,
            args_supplier=add_args_supplier,
        )
        add_stats = throughput_stats(batch_size, add_durations)

        filled_container = getattr(container, add_method)(values_device)
        jax.block_until_ready(filled_container)
        remove_durations = run_jax_trials(
            lambda: getattr(filled_container, remove_method)(batch_size),
            trials=trials,
            include_device_transfer=(mode == "e2e"),
        )
        remove_stats = throughput_stats(batch_size, remove_durations)

        results["xtructure"].setdefault(add_key, []).append(add_stats)
        results["xtructure"].setdefault(remove_key, []).append(remove_stats)

        def python_add_op():
            return python_add(materialize_items(mode == "e2e"))

        add_durations = run_python_trials(python_add_op, trials=trials)
        add_stats = throughput_stats(batch_size, add_durations)

        def python_remove_op():
            return python_remove(materialize_items(mode == "e2e"), batch_size)

        remove_durations = run_python_trials(python_remove_op, trials=trials)
        remove_stats = throughput_stats(batch_size, remove_durations)

        results["python"].setdefault(add_key, []).append(add_stats)
        results["python"].setdefault(remove_key, []).append(remove_stats)

    validate_results_schema(results)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(results, f, indent=4)

    print(f"{container_name} benchmark results saved to {output_path}")
    print_results_table(results, title)


def print_results_table(results: Dict[str, Any], title: str):
    """Print benchmark results as a plain text table."""
    batch_sizes = results.get("batch_sizes", [])
    xtructure_results = results.get("xtructure", {})
    python_results = results.get("python", {})

    operations = list(xtructure_results.keys())
    rows = []

    for i, size in enumerate(batch_sizes):
        for op in operations:
            op_name = op.replace("_ops_per_sec", "")

            # Add row for xtructure
            xtructure_data = xtructure_results.get(op, [])[i]
            xtructure_p99 = "-"
            if isinstance(xtructure_data, dict):
                xtructure_perf = xtructure_data["median"]
                xtructure_iqr = xtructure_data["iqr"]
                if "p99_ops" in xtructure_data:
                    xtructure_p99 = human_format(xtructure_data["p99_ops"])
            else:
                xtructure_perf = xtructure_data
                xtructure_iqr = 0

            rows.append(
                (
                    f"{size:,}",
                    op_name,
                    "xtructure",
                    human_format(xtructure_perf),
                    f"±{human_format(xtructure_iqr)}",
                    xtructure_p99,
                )
            )

            # Add row for python
            python_data = python_results.get(op, [])[i]
            python_p99 = "-"
            if isinstance(python_data, dict):
                python_perf = python_data["median"]
                python_iqr = python_data["iqr"]
                if "p99_ops" in python_data:
                    python_p99 = human_format(python_data["p99_ops"])
            else:
                python_perf = python_data
                python_iqr = 0

            rows.append(
                (
                    "",
                    op_name,
                    "python",
                    human_format(python_perf),
                    f"±{human_format(python_iqr)}",
                    python_p99,
                )
            )

    headers = ("Batch Size", "Operation", "Implementation", "Ops/Sec (Median)", "IQR", "P99")
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    def format_row(row):
        return "  ".join(value.rjust(width) for value, width in zip(row, widths))

    print(title)
    print(format_row(headers))
    print(format_row(tuple("-" * width for width in widths)))
    for row in rows:
        print(format_row(row))
