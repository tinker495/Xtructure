import dataclasses
import json
import time
from collections import deque
from typing import Any, Dict

import jax

from xtructure import Queue
from xtructure_benchmarks.common import BenchmarkValue, print_results_table


def benchmark_xtructure_queue_enqueue(queue: Queue, values: BenchmarkValue, trials: int = 10):
    """Benchmarks the batched enqueue operation for xtructure.Queue."""

    def enqueue_op():
        return queue.enqueue(values)

    # JIT compile and warm up
    jitted_enqueue = jax.jit(enqueue_op)
    new_queue = jitted_enqueue()
    jax.block_until_ready(new_queue)

    # Time multiple trials
    times = []
    for _ in range(trials):
        start_time = time.perf_counter()
        new_queue = jitted_enqueue()
        jax.block_until_ready(new_queue)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    import numpy as np

    times = np.array(times)
    median_time = np.median(times)
    q75, q25 = np.percentile(times, [75, 25])
    iqr_time = q75 - q25

    return median_time, iqr_time


def benchmark_xtructure_queue_dequeue(
    queue: Queue, count: int, trials: int = 10, include_host_transfer: bool = False
):
    """Benchmarks the batched dequeue operation for xtructure.Queue."""

    def dequeue_op():
        return queue.dequeue(count)

    # JIT compile and warm up
    jitted_dequeue = jax.jit(dequeue_op)
    new_queue, _ = jitted_dequeue()
    jax.block_until_ready(new_queue)

    # Time multiple trials
    times = []
    for _ in range(trials):
        start_time = time.perf_counter()
        new_queue, dequeued_values = jitted_dequeue()
        jax.block_until_ready(new_queue)
        if include_host_transfer:
            # Include device-to-host transfer cost outside JIT
            _ = jax.device_get(dequeued_values)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    import numpy as np

    times = np.array(times)
    median_time = np.median(times)
    q75, q25 = np.percentile(times, [75, 25])
    iqr_time = q75 - q25

    return median_time, iqr_time


def benchmark_deque_enqueue(values: BenchmarkValue, trials: int = 10):
    """Benchmarks the extend (batched enqueue) operation for collections.deque."""
    # Convert xtructure data to a list of dicts for deque
    # Use dataclasses.asdict and then convert numpy arrays to lists
    dict_of_arrays = dataclasses.asdict(values)
    items = [dict(zip(dict_of_arrays, t)) for t in zip(*dict_of_arrays.values())]

    times = []
    for _ in range(trials):
        d = deque()  # Fresh deque per trial
        start_time = time.perf_counter()
        d.extend(items)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    import numpy as np

    times = np.array(times)
    median_time = np.median(times)
    q75, q25 = np.percentile(times, [75, 25])
    iqr_time = q75 - q25

    return median_time, iqr_time


def benchmark_deque_dequeue(values: BenchmarkValue, count: int, trials: int = 10):
    """Benchmarks the popleft (dequeue) operation for collections.deque."""
    # Convert xtructure data to a list of dicts for deque
    dict_of_arrays = dataclasses.asdict(values)
    items = [dict(zip(dict_of_arrays, t)) for t in zip(*dict_of_arrays.values())]

    times = []
    for _ in range(trials):
        d = deque(items)  # Fresh filled deque per trial
        start_time = time.perf_counter()
        results = []
        for _ in range(count):
            if d:
                results.append(d.popleft())
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    import numpy as np

    times = np.array(times)
    median_time = np.median(times)
    q75, q25 = np.percentile(times, [75, 25])
    iqr_time = q75 - q25

    return median_time, iqr_time


def run_benchmarks():
    """Runs the full suite of Queue benchmarks and saves the results."""
    batch_sizes = [2**10, 2**12, 2**14]
    results: Dict[str, Any] = {"batch_sizes": batch_sizes, "xtructure": {}, "python": {}}
    max_size = int(max(batch_sizes) * 2)

    print("Running Queue Benchmarks...")
    for batch_size in batch_sizes:
        print(f"  Batch Size: {batch_size}")
        key = jax.random.PRNGKey(batch_size)
        values = BenchmarkValue.random(shape=(batch_size,), key=key)

        # --- xtructure.Queue Benchmark ---
        xtructure_queue = Queue.build(max_size=max_size, value_class=BenchmarkValue)
        xtructure_enqueue_median, xtructure_enqueue_iqr = benchmark_xtructure_queue_enqueue(
            xtructure_queue, values
        )

        # Create a filled queue for dequeue benchmark
        queue_with_data = xtructure_queue.enqueue(values)
        jax.block_until_ready(queue_with_data)
        xtructure_dequeue_median, xtructure_dequeue_iqr = benchmark_xtructure_queue_dequeue(
            queue_with_data, batch_size
        )

        results["xtructure"].setdefault("enqueue_ops_per_sec", []).append(
            {
                "median": batch_size / xtructure_enqueue_median
                if xtructure_enqueue_median > 0
                else 0,
                "iqr": batch_size * xtructure_enqueue_iqr / (xtructure_enqueue_median**2)
                if xtructure_enqueue_median > 0
                else 0,
            }
        )
        results["xtructure"].setdefault("dequeue_ops_per_sec", []).append(
            {
                "median": batch_size / xtructure_dequeue_median
                if xtructure_dequeue_median > 0
                else 0,
                "iqr": batch_size * xtructure_dequeue_iqr / (xtructure_dequeue_median**2)
                if xtructure_dequeue_median > 0
                else 0,
            }
        )

        # --- Python collections.deque Benchmark ---
        python_enqueue_median, python_enqueue_iqr = benchmark_deque_enqueue(values)
        python_dequeue_median, python_dequeue_iqr = benchmark_deque_dequeue(values, batch_size)

        results["python"].setdefault("enqueue_ops_per_sec", []).append(
            {
                "median": batch_size / python_enqueue_median if python_enqueue_median > 0 else 0,
                "iqr": batch_size * python_enqueue_iqr / (python_enqueue_median**2)
                if python_enqueue_median > 0
                else 0,
            }
        )
        results["python"].setdefault("dequeue_ops_per_sec", []).append(
            {
                "median": batch_size / python_dequeue_median if python_dequeue_median > 0 else 0,
                "iqr": batch_size * python_dequeue_iqr / (python_dequeue_median**2)
                if python_dequeue_median > 0
                else 0,
            }
        )

    # Save results
    output_path = "xtructure_benchmarks/results/queue_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Queue benchmark results saved to {output_path}")
    print_results_table(results, "Queue Performance Results")


if __name__ == "__main__":
    run_benchmarks()
