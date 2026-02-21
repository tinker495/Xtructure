import argparse
from collections import deque
from typing import List, Optional

import jax

from xtructure import Queue
from xtructure_benchmarks.common import (
    BenchmarkValue,
    add_common_benchmark_args,
    init_benchmark_results,
    parse_batch_sizes,
    run_jax_trials,
    run_python_trials,
    save_and_print_results,
    throughput_stats,
    to_python_values,
)


def run_benchmarks(mode: str = "kernel", trials: int = 10, batch_sizes: Optional[List[int]] = None):
    """Runs the full suite of Queue benchmarks and saves the results."""
    batch_sizes = batch_sizes or [2**10, 2**12, 2**14]
    results = init_benchmark_results(batch_sizes)
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

        def enqueue_args_supplier():
            if mode == "e2e":
                return (jax.device_put(values_host),)
            return (values_device,)

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

    save_and_print_results(
        results,
        "xtructure_benchmarks/results/queue_results.json",
        "Queue Performance Results",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Queue benchmarks")
    add_common_benchmark_args(parser)
    args = parser.parse_args()

    run_benchmarks(
        mode=args.mode,
        trials=args.trials,
        batch_sizes=parse_batch_sizes(args.batch_sizes),
    )
