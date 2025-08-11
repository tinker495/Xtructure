import heapq
import json
import time
from typing import Any, Dict, List

import jax
import jax.numpy as jnp

from xtructure import BGPQ
from xtructure_benchmarks.common import BenchmarkValue, print_results_table


# Key generation function adapted from tests/heap_test.py
# Improved to reduce ties by using full hash range
@jax.jit
def key_gen(x: BenchmarkValue) -> float:
    uint32_hash = x.hash()
    # Map to [0,1) with much fewer ties
    key = uint32_hash / jnp.float32(2**32)
    return key.astype(jnp.float32)


vmapped_key_gen = jax.jit(jax.vmap(key_gen))


def benchmark_bgpq_insert(heap: BGPQ, keys: jnp.ndarray, values: BenchmarkValue, trials: int = 10):
    """Benchmarks the batched insertion into the xtructure BGPQ."""

    def insert_op():
        return heap.insert(keys, values)

    # JIT compile and warm up
    jitted_insert = jax.jit(insert_op)
    new_heap = jitted_insert()
    jax.block_until_ready(new_heap)

    # Time multiple trials
    times = []
    for _ in range(trials):
        start_time = time.perf_counter()
        new_heap = jitted_insert()
        jax.block_until_ready(new_heap)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    import numpy as np

    times = np.array(times)
    median_time = np.median(times)
    q75, q25 = np.percentile(times, [75, 25])
    iqr_time = q75 - q25

    return median_time, iqr_time


def benchmark_bgpq_delete(heap: BGPQ, trials: int = 10, include_host_transfer: bool = False):
    """Benchmarks the batched deletion from the xtructure BGPQ."""

    def delete_op():
        return BGPQ.delete_mins(heap)

    # JIT compile and warm up
    jitted_delete = jax.jit(delete_op)
    new_heap, _, _ = jitted_delete()
    jax.block_until_ready(new_heap)

    # Time multiple trials
    times = []
    for _ in range(trials):
        start_time = time.perf_counter()
        new_heap, deleted_keys, deleted_values = jitted_delete()
        jax.block_until_ready(new_heap)
        if include_host_transfer:
            # Include device-to-host transfer cost outside JIT
            _ = jax.device_get((deleted_keys, deleted_values))
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    import numpy as np

    times = np.array(times)
    median_time = np.median(times)
    q75, q25 = np.percentile(times, [75, 25])
    iqr_time = q75 - q25

    return median_time, iqr_time


def benchmark_heapq_insert(
    keys: jnp.ndarray, values: BenchmarkValue, trials: int = 10, bulk_mode: bool = False
):
    """Benchmarks insertion into Python's heapq."""
    # Convert JAX arrays to Python native types for heapq
    native_keys = keys.tolist()
    value_bytes = [v.tobytes() for v in jax.vmap(lambda x: x.bytes)(values)]

    times = []
    for _ in range(trials):
        if bulk_mode:
            # Bulk build using heapify - O(n) instead of O(n log n)
            items = [(native_keys[i], i, value_bytes[i]) for i in range(len(native_keys))]
            start_time = time.perf_counter()
            heapq.heapify(items)
            end_time = time.perf_counter()
        else:
            # Incremental insert with integer tiebreaker to avoid expensive bytes comparison
            data_heap = []  # Fresh heap per trial
            start_time = time.perf_counter()
            for i in range(len(native_keys)):
                heapq.heappush(data_heap, (native_keys[i], i, value_bytes[i]))
            end_time = time.perf_counter()
        times.append(end_time - start_time)

    import numpy as np

    times = np.array(times)
    median_time = np.median(times)
    q75, q25 = np.percentile(times, [75, 25])
    iqr_time = q75 - q25

    return median_time, iqr_time


def benchmark_heapq_delete(prototype_heap: List, count: int, trials: int = 10):
    """Benchmarks deletion from Python's heapq."""

    times = []
    for _ in range(trials):
        data_heap = prototype_heap.copy()  # Fresh copy per trial
        start_time = time.perf_counter()
        results = []
        for _ in range(count):
            if data_heap:
                results.append(heapq.heappop(data_heap))
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    import numpy as np

    times = np.array(times)
    median_time = np.median(times)
    q75, q25 = np.percentile(times, [75, 25])
    iqr_time = q75 - q25

    return median_time, iqr_time


def run_benchmarks():
    """Runs the full suite of Heap benchmarks and saves the results."""
    # Using smaller batch sizes to ensure completion within a reasonable time
    batch_sizes = [2**10, 2**12, 2**14]
    results: Dict[str, Any] = {"batch_sizes": batch_sizes, "xtructure": {}, "python": {}}
    max_size = int(max(batch_sizes) * 1.5)

    print("Running Heap (BGPQ) Benchmarks...")
    for batch_size in batch_sizes:
        print(f"  Batch Size: {batch_size}")
        key = jax.random.PRNGKey(batch_size)
        values = BenchmarkValue.random(shape=(batch_size,), key=key)
        keys = vmapped_key_gen(values)

        # --- xtructure.BGPQ Benchmark ---
        bgpq_heap = BGPQ.build(max_size, batch_size, BenchmarkValue, jnp.float32)

        # We need to make the keys/values batched to the BGPQ batch_size
        padded_keys, padded_values = BGPQ.make_batched(keys, values, batch_size)

        xtructure_insert_median, xtructure_insert_iqr = benchmark_bgpq_insert(
            bgpq_heap, padded_keys, padded_values
        )

        # Create a filled heap for deletion benchmark
        heap_with_data = bgpq_heap.insert(padded_keys, padded_values)
        jax.block_until_ready(heap_with_data)
        xtructure_delete_median, xtructure_delete_iqr = benchmark_bgpq_delete(heap_with_data)

        results["xtructure"].setdefault("insert_ops_per_sec", []).append(
            {
                "median": batch_size / xtructure_insert_median
                if xtructure_insert_median > 0
                else 0,
                "iqr": batch_size * xtructure_insert_iqr / (xtructure_insert_median**2)
                if xtructure_insert_median > 0
                else 0,
            }
        )
        results["xtructure"].setdefault("delete_ops_per_sec", []).append(
            {
                "median": batch_size / xtructure_delete_median
                if xtructure_delete_median > 0
                else 0,
                "iqr": batch_size * xtructure_delete_iqr / (xtructure_delete_median**2)
                if xtructure_delete_median > 0
                else 0,
            }
        )

        # --- Python heapq Benchmark ---
        # Test both incremental and bulk modes, report the better one
        python_insert_incremental_median, python_insert_incremental_iqr = benchmark_heapq_insert(
            keys, values, bulk_mode=False
        )
        python_insert_bulk_median, python_insert_bulk_iqr = benchmark_heapq_insert(
            keys, values, bulk_mode=True
        )

        # Use the better (faster) Python baseline and build prototype heap for deletion
        native_keys = keys.tolist()
        value_bytes = [v.tobytes() for v in jax.vmap(lambda x: x.bytes)(values)]

        if python_insert_incremental_median < python_insert_bulk_median:
            python_insert_median, python_insert_iqr = (
                python_insert_incremental_median,
                python_insert_incremental_iqr,
            )
            # Build prototype heap using incremental method
            py_heap = []
            for i in range(len(native_keys)):
                heapq.heappush(py_heap, (native_keys[i], i, value_bytes[i]))
        else:
            python_insert_median, python_insert_iqr = (
                python_insert_bulk_median,
                python_insert_bulk_iqr,
            )
            # Build prototype heap using bulk method
            py_heap = [(native_keys[i], i, value_bytes[i]) for i in range(len(native_keys))]
            heapq.heapify(py_heap)

        python_delete_median, python_delete_iqr = benchmark_heapq_delete(py_heap, batch_size)

        results["python"].setdefault("insert_ops_per_sec", []).append(
            {
                "median": batch_size / python_insert_median if python_insert_median > 0 else 0,
                "iqr": batch_size * python_insert_iqr / (python_insert_median**2)
                if python_insert_median > 0
                else 0,
            }
        )
        results["python"].setdefault("delete_ops_per_sec", []).append(
            {
                "median": batch_size / python_delete_median if python_delete_median > 0 else 0,
                "iqr": batch_size * python_delete_iqr / (python_delete_median**2)
                if python_delete_median > 0
                else 0,
            }
        )

    # Save results
    output_path = "xtructure_benchmarks/results/heap_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Heap benchmark results saved to {output_path}")
    print_results_table(results, "Heap (BGPQ) Performance Results")


if __name__ == "__main__":
    run_benchmarks()
