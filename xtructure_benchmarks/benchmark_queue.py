import dataclasses
import json
import time
from collections import deque
from typing import Any, Deque, Dict

import jax

from xtructure import Queue
from xtructure_benchmarks.common import (
    BenchmarkValue,
    print_results_table,
    python_timer,
)


def benchmark_xtructure_queue_enqueue(queue: Queue, values: BenchmarkValue):
    """Benchmarks the batched enqueue operation for xtructure.Queue."""

    def enqueue_op():
        return queue.enqueue(values)

    # JIT compile and warm up
    jitted_enqueue = jax.jit(enqueue_op)
    new_queue = jitted_enqueue()
    jax.block_until_ready(new_queue)

    # Time the actual execution
    start_time = time.perf_counter()
    new_queue = jitted_enqueue()
    jax.block_until_ready(new_queue)
    end_time = time.perf_counter()

    return end_time - start_time


def benchmark_xtructure_queue_dequeue(queue: Queue, count: int):
    """Benchmarks the batched dequeue operation for xtructure.Queue."""

    def dequeue_op():
        return queue.dequeue(count)

    # JIT compile and warm up
    jitted_dequeue = jax.jit(dequeue_op)
    new_queue, _ = jitted_dequeue()
    jax.block_until_ready(new_queue)

    # Time the actual execution
    start_time = time.perf_counter()
    new_queue, _ = jitted_dequeue()
    jax.block_until_ready(new_queue)
    end_time = time.perf_counter()

    return end_time - start_time


def benchmark_deque_enqueue(d: Deque, values: BenchmarkValue):
    """Benchmarks the extend (batched enqueue) operation for collections.deque."""
    # Convert xtructure data to a list of dicts for deque
    # Use dataclasses.asdict and then convert numpy arrays to lists
    dict_of_arrays = dataclasses.asdict(values)
    items = [dict(zip(dict_of_arrays, t)) for t in zip(*dict_of_arrays.values())]

    def enqueue_op():
        d.extend(items)

    return python_timer(enqueue_op)


def benchmark_deque_dequeue(d: Deque, count: int):
    """Benchmarks the popleft (dequeue) operation for collections.deque."""

    def dequeue_op():
        for _ in range(count):
            if d:
                d.popleft()

    return python_timer(dequeue_op)


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
        xtructure_enqueue_time = benchmark_xtructure_queue_enqueue(xtructure_queue, values)

        # Create a filled queue for dequeue benchmark
        queue_with_data = xtructure_queue.enqueue(values)
        jax.block_until_ready(queue_with_data)
        xtructure_dequeue_time = benchmark_xtructure_queue_dequeue(queue_with_data, batch_size)

        results["xtructure"].setdefault("enqueue_ops_per_sec", []).append(
            batch_size / xtructure_enqueue_time if xtructure_enqueue_time > 0 else 0
        )
        results["xtructure"].setdefault("dequeue_ops_per_sec", []).append(
            batch_size / xtructure_dequeue_time if xtructure_dequeue_time > 0 else 0
        )

        # --- Python collections.deque Benchmark ---
        py_deque: Deque[Dict] = deque()
        python_enqueue_time = benchmark_deque_enqueue(py_deque, values)

        # We already have the filled deque for dequeue
        python_dequeue_time = benchmark_deque_dequeue(py_deque, batch_size)

        results["python"].setdefault("enqueue_ops_per_sec", []).append(
            batch_size / python_enqueue_time if python_enqueue_time > 0 else 0
        )
        results["python"].setdefault("dequeue_ops_per_sec", []).append(
            batch_size / python_dequeue_time if python_dequeue_time > 0 else 0
        )

    # Save results
    output_path = "xtructure_benchmarks/results/queue_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Queue benchmark results saved to {output_path}")
    print_results_table(results, "Queue Performance Results")


if __name__ == "__main__":
    run_benchmarks()
