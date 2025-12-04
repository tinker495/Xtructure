import argparse
import json
from collections import deque
from typing import Any, Dict, List, Optional

import jax

from xtructure import Queue
from xtructure_benchmarks.common import (
    BenchmarkValue,
    check_system_load,
    get_system_info,
    print_results_table,
    run_jax_trials,
    run_python_trials,
    throughput_stats,
    to_python_values,
    validate_results_schema,
)


def run_benchmarks(mode: str = "kernel", trials: int = 10, batch_sizes: Optional[List[int]] = None):
    """Runs the full suite of Queue benchmarks and saves the results."""
    batch_sizes = batch_sizes or [2**10, 2**12, 2**14]

    check_system_load()

    results: Dict[str, Any] = {
        "batch_sizes": batch_sizes,
        "xtructure": {},
        "python": {},
        "environment": get_system_info(),
    }
    max_size = int(max(batch_sizes) * 2)

    print("Running Queue Benchmarks...")
    try:
        print(f"JAX backend: {jax.default_backend()}")
        print("JAX devices:", ", ".join([d.platform + ":" + d.device_kind for d in jax.devices()]))
    except Exception:
        pass
    for batch_size in batch_sizes:
        print(f"  Batch Size: {batch_size}")
        key = jax.random.PRNGKey(batch_size)
        values_device = BenchmarkValue.random(shape=(batch_size,), key=key)
        values_host = jax.device_get(values_device)

        precomputed_items = to_python_values(values_device)

        def materialize_items(include_preprocessing: bool):
            return to_python_values(values_device) if include_preprocessing else precomputed_items

        # --- xtructure.Queue Benchmark ---
        xtructure_queue = Queue.build(max_size=max_size, value_class=BenchmarkValue)
        enqueue_args_supplier = (
            lambda: (jax.device_put(values_host),) if mode == "e2e" else (values_device,)
        )
        enqueue_durations = run_jax_trials(
            lambda batch: xtructure_queue.enqueue(batch),
            trials=trials,
            args_supplier=enqueue_args_supplier,
        )
        xtructure_enqueue_stats = throughput_stats(batch_size, enqueue_durations)

        # Create a filled queue for dequeue benchmark
        queue_with_data = xtructure_queue.enqueue(values_device)
        jax.block_until_ready(queue_with_data)
        dequeue_durations = run_jax_trials(
            lambda: queue_with_data.dequeue(batch_size),
            trials=trials,
            include_device_transfer=(mode == "e2e"),
        )
        xtructure_dequeue_stats = throughput_stats(batch_size, dequeue_durations)

        results["xtructure"].setdefault("enqueue_ops_per_sec", []).append(xtructure_enqueue_stats)
        results["xtructure"].setdefault("dequeue_ops_per_sec", []).append(xtructure_dequeue_stats)

        # --- Python collections.deque Benchmark ---
        def python_enqueue_op():
            d = deque()
            items = materialize_items(mode == "e2e")
            d.extend(items)
            return d

        python_enqueue_durations = run_python_trials(python_enqueue_op, trials=trials)
        python_enqueue_stats = throughput_stats(batch_size, python_enqueue_durations)

        def python_dequeue_op():
            items = materialize_items(mode == "e2e")
            d = deque(items)
            results_local = []
            for _ in range(batch_size):
                if d:
                    results_local.append(d.popleft())
            return results_local

        python_dequeue_durations = run_python_trials(python_dequeue_op, trials=trials)
        python_dequeue_stats = throughput_stats(batch_size, python_dequeue_durations)

        results["python"].setdefault("enqueue_ops_per_sec", []).append(python_enqueue_stats)
        results["python"].setdefault("dequeue_ops_per_sec", []).append(python_dequeue_stats)

    # Validate and save results
    validate_results_schema(results)
    output_path = "xtructure_benchmarks/results/queue_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Queue benchmark results saved to {output_path}")
    print_results_table(results, "Queue Performance Results")


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
