import heapq
import json
import time
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp

from xtructure import BGPQ
from xtructure_benchmarks.common import (
    BenchmarkValue,
    print_results_table,
    python_timer,
)


# Key generation function adapted from tests/heap_test.py
@jax.jit
def key_gen(x: BenchmarkValue) -> float:
    uint32_hash = x.hash()
    key = uint32_hash % (2**12) / (2**8)
    return key.astype(jnp.float32)


vmapped_key_gen = jax.jit(jax.vmap(key_gen))


def benchmark_bgpq_insert(heap: BGPQ, keys: jnp.ndarray, values: BenchmarkValue):
    """Benchmarks the batched insertion into the xtructure BGPQ."""

    def insert_op():
        return heap.insert(keys, values)

    # JIT compile and warm up
    jitted_insert = jax.jit(insert_op)
    new_heap = jitted_insert()
    jax.block_until_ready(new_heap)

    # Time the actual execution
    start_time = time.perf_counter()
    new_heap = jitted_insert()
    jax.block_until_ready(new_heap)
    end_time = time.perf_counter()

    return end_time - start_time


def benchmark_bgpq_delete(heap: BGPQ):
    """Benchmarks the batched deletion from the xtructure BGPQ."""

    def delete_op():
        return BGPQ.delete_mins(heap)

    # JIT compile and warm up
    jitted_delete = jax.jit(delete_op)
    new_heap, _, _ = jitted_delete()
    jax.block_until_ready(new_heap)

    # Time the actual execution
    start_time = time.perf_counter()
    new_heap, _, _ = jitted_delete()
    jax.block_until_ready(new_heap)
    end_time = time.perf_counter()

    return end_time - start_time


def benchmark_heapq_insert(data_heap: List, keys: jnp.ndarray, values: BenchmarkValue):
    """Benchmarks insertion into Python's heapq."""
    # Convert JAX arrays to Python native types for heapq
    native_keys = keys.tolist()
    value_bytes = [v.tobytes() for v in jax.vmap(lambda x: x.bytes)(values)]

    def insert_op():
        for i in range(len(native_keys)):
            heapq.heappush(data_heap, (native_keys[i], value_bytes[i]))

    return python_timer(insert_op)


def benchmark_heapq_delete(data_heap: List, count: int):
    """Benchmarks deletion from Python's heapq."""

    def delete_op():
        for _ in range(count):
            if data_heap:
                heapq.heappop(data_heap)

    return python_timer(delete_op)


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

        xtructure_insert_time = benchmark_bgpq_insert(bgpq_heap, padded_keys, padded_values)

        # Create a filled heap for deletion benchmark
        heap_with_data = bgpq_heap.insert(padded_keys, padded_values)
        jax.block_until_ready(heap_with_data)
        xtructure_delete_time = benchmark_bgpq_delete(heap_with_data)

        results["xtructure"].setdefault("insert_ops_per_sec", []).append(
            batch_size / xtructure_insert_time if xtructure_insert_time > 0 else 0
        )
        results["xtructure"].setdefault("delete_ops_per_sec", []).append(
            batch_size / xtructure_delete_time if xtructure_delete_time > 0 else 0
        )

        # --- Python heapq Benchmark ---
        py_heap: List[Tuple[float, bytes]] = []
        python_insert_time = benchmark_heapq_insert(py_heap, keys, values)
        # We already have the filled heap for deletion
        python_delete_time = benchmark_heapq_delete(py_heap, batch_size)

        results["python"].setdefault("insert_ops_per_sec", []).append(
            batch_size / python_insert_time if python_insert_time > 0 else 0
        )
        results["python"].setdefault("delete_ops_per_sec", []).append(
            batch_size / python_delete_time if python_delete_time > 0 else 0
        )

    # Save results
    output_path = "xtructure_benchmarks/results/heap_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Heap benchmark results saved to {output_path}")
    print_results_table(results, "Heap (BGPQ) Performance Results")


if __name__ == "__main__":
    run_benchmarks()
