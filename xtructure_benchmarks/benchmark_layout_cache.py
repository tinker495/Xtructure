import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import jax.numpy as jnp

from xtructure_benchmarks.common import (
    BenchmarkValue,
    check_system_load,
    get_system_info,
    print_results_table,
    run_python_trials,
    throughput_stats,
    validate_results_schema,
)


@dataclass
class PythonLayoutValue:
    """Small Python baseline for layout-property read and string-format overhead."""

    batch_shape: tuple[int, ...]

    @property
    def shape(self):
        return (self.batch_shape, (), (), (3,), (128,))

    @property
    def dtype(self):
        return (jnp.uint32, jnp.uint32, jnp.float32, jnp.float16)

    @property
    def structured_type(self):
        return "BATCHED" if self.batch_shape else "SINGLE"

    def replace(self):
        return PythonLayoutValue(batch_shape=self.batch_shape)

    def __str__(self):
        return (
            "PythonLayoutValue("
            f"batch_shape={self.batch_shape}, shape={self.shape}, dtype={self.dtype})"
        )


def _read_shape(instance: Any, read_count: int) -> None:
    for _ in range(read_count):
        _ = instance.shape


def _read_dtype(instance: Any, read_count: int) -> None:
    for _ in range(read_count):
        _ = instance.dtype


def _read_structured_type(instance: Any, read_count: int) -> None:
    for _ in range(read_count):
        _ = instance.structured_type


def _replace_xtructure(instance: BenchmarkValue, replace_count: int) -> None:
    current = instance
    for _ in range(replace_count):
        current = current.replace(timestamp=current.timestamp)


def _replace_python(instance: PythonLayoutValue, replace_count: int) -> None:
    current = instance
    for _ in range(replace_count):
        current = current.replace()


def run_benchmarks(
    mode: str = "kernel",
    trials: int = 10,
    batch_sizes: Optional[List[int]] = None,
    read_count: int = 2048,
    replace_count: int = 256,
):
    """Run Layout Cache micro-benchmarks for property reads, formatting, and replace()."""
    del mode  # Layout Cache microbenchmarks are host-side adapter costs.
    batch_sizes = batch_sizes or [2**10, 2**12, 2**14]

    check_system_load()

    results: Dict[str, Any] = {
        "batch_sizes": batch_sizes,
        "xtructure": {},
        "python": {},
        "environment": get_system_info(),
    }

    print("Running Layout Cache Microbenchmarks...")
    for batch_size in batch_sizes:
        print(f"  Batch Size: {batch_size}")
        xtructure_value = BenchmarkValue.default(shape=(batch_size,))
        python_value = PythonLayoutValue(batch_shape=(batch_size,))

        xtructure_shape_stats = throughput_stats(
            read_count,
            run_python_trials(lambda: _read_shape(xtructure_value, read_count), trials=trials),
        )
        python_shape_stats = throughput_stats(
            read_count,
            run_python_trials(lambda: _read_shape(python_value, read_count), trials=trials),
        )

        xtructure_dtype_stats = throughput_stats(
            read_count,
            run_python_trials(lambda: _read_dtype(xtructure_value, read_count), trials=trials),
        )
        python_dtype_stats = throughput_stats(
            read_count,
            run_python_trials(lambda: _read_dtype(python_value, read_count), trials=trials),
        )

        xtructure_structured_type_stats = throughput_stats(
            read_count,
            run_python_trials(
                lambda: _read_structured_type(xtructure_value, read_count),
                trials=trials,
            ),
        )
        python_structured_type_stats = throughput_stats(
            read_count,
            run_python_trials(
                lambda: _read_structured_type(python_value, read_count),
                trials=trials,
            ),
        )

        xtructure_string_format_stats = throughput_stats(
            1,
            run_python_trials(lambda: str(xtructure_value), trials=trials),
        )
        python_string_format_stats = throughput_stats(
            1,
            run_python_trials(lambda: str(python_value), trials=trials),
        )

        xtructure_replace_stats = throughput_stats(
            replace_count,
            run_python_trials(
                lambda: _replace_xtructure(xtructure_value, replace_count),
                trials=trials,
            ),
        )
        python_replace_stats = throughput_stats(
            replace_count,
            run_python_trials(
                lambda: _replace_python(python_value, replace_count),
                trials=trials,
            ),
        )

        results["xtructure"].setdefault("shape_read_ops_per_sec", []).append(xtructure_shape_stats)
        results["python"].setdefault("shape_read_ops_per_sec", []).append(python_shape_stats)
        results["xtructure"].setdefault("dtype_read_ops_per_sec", []).append(xtructure_dtype_stats)
        results["python"].setdefault("dtype_read_ops_per_sec", []).append(python_dtype_stats)
        results["xtructure"].setdefault("structured_type_read_ops_per_sec", []).append(
            xtructure_structured_type_stats
        )
        results["python"].setdefault("structured_type_read_ops_per_sec", []).append(
            python_structured_type_stats
        )
        results["xtructure"].setdefault("string_format_ops_per_sec", []).append(
            xtructure_string_format_stats
        )
        results["python"].setdefault("string_format_ops_per_sec", []).append(
            python_string_format_stats
        )
        results["xtructure"].setdefault("replace_ops_per_sec", []).append(xtructure_replace_stats)
        results["python"].setdefault("replace_ops_per_sec", []).append(python_replace_stats)

    validate_results_schema(results)
    output_path = "xtructure_benchmarks/results/layout_cache_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Layout Cache benchmark results saved to {output_path}")
    print_results_table(results, "Layout Cache Microbenchmark Results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layout Cache microbenchmarks")
    parser.add_argument("--mode", choices=["kernel", "e2e"], default="kernel")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="",
        help="Comma-separated batch sizes (e.g. 1024,4096,16384)",
    )
    parser.add_argument("--read-count", type=int, default=2048)
    parser.add_argument("--replace-count", type=int, default=256)
    args = parser.parse_args()

    batch_sizes_arg: Optional[List[int]] = None
    if args.batch_sizes:
        batch_sizes_arg = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]

    run_benchmarks(
        mode=args.mode,
        trials=args.trials,
        batch_sizes=batch_sizes_arg,
        read_count=args.read_count,
        replace_count=args.replace_count,
    )
