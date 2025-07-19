import json
import time
from typing import Any, Dict

import jax

from xtructure import HashTable
from xtructure_benchmarks.common import (
    BenchmarkValue,
    print_results_table,
    python_timer,
)


def benchmark_hashtable_insert(table: HashTable, keys: BenchmarkValue):
    """Benchmarks the parallel insertion into the xtructure HashTable."""

    def insert_op():
        # The key and value are the same in this benchmark
        return table.parallel_insert(keys)[0]

    # JIT compile and warm up
    jitted_insert = jax.jit(insert_op)
    new_table = jitted_insert()
    jax.block_until_ready(new_table)

    # Time the actual execution
    start_time = time.perf_counter()
    new_table = jitted_insert()
    jax.block_until_ready(new_table)
    end_time = time.perf_counter()

    return end_time - start_time


def benchmark_hashtable_lookup(table: HashTable, keys: BenchmarkValue):
    """Benchmarks the parallel lookup from the xtructure HashTable."""

    def lookup_op():
        return table.lookup_parallel(keys)

    # JIT compile and warm up
    jitted_lookup = jax.jit(lookup_op)
    result = jitted_lookup()
    jax.block_until_ready(result)

    # Time the actual execution
    start_time = time.perf_counter()
    result = jitted_lookup()
    jax.block_until_ready(result)
    end_time = time.perf_counter()

    return end_time - start_time


def benchmark_dict_insert(data_dict: Dict, keys: BenchmarkValue):
    """Benchmarks insertion into a standard Python dict."""
    # Convert JAX arrays to hashable Python bytes
    key_bytes = [arr.tobytes() for arr in jax.vmap(lambda x: x.bytes)(keys)]

    def insert_op():
        for i, key in enumerate(key_bytes):
            data_dict[key] = i

    return python_timer(insert_op)


def benchmark_dict_lookup(data_dict: Dict, keys: BenchmarkValue):
    """Benchmarks lookup from a standard Python dict."""
    # Convert JAX arrays to hashable Python bytes
    key_bytes = [arr.tobytes() for arr in jax.vmap(lambda x: x.bytes)(keys)]

    def lookup_op():
        for key in key_bytes:
            _ = data_dict.get(key)

    return python_timer(lookup_op)


def run_benchmarks():
    """Runs the full suite of HashTable benchmarks and saves the results."""
    # Using smaller batch sizes to ensure completion within a reasonable time
    batch_sizes = [2**10, 2**12, 2**14]
    results: Dict[str, Any] = {"batch_sizes": batch_sizes, "xtructure": {}, "python": {}}
    table_size = int(max(batch_sizes) * 1.5)

    print("Running HashTable Benchmarks...")
    for batch_size in batch_sizes:
        print(f"  Batch Size: {batch_size}")
        key = jax.random.PRNGKey(0)
        keys = BenchmarkValue.random(shape=(batch_size,), key=key)

        # --- xtructure.HashTable Benchmark ---
        xtructure_table = HashTable.build(BenchmarkValue, 1, table_size)
        xtructure_insert_time = benchmark_hashtable_insert(xtructure_table, keys)
        # Re-build and insert for a fair lookup comparison
        table_with_data, _, _, _ = xtructure_table.parallel_insert(keys)
        jax.block_until_ready(table_with_data)
        xtructure_lookup_time = benchmark_hashtable_lookup(table_with_data, keys)

        results["xtructure"].setdefault("insert_ops_per_sec", []).append(
            batch_size / xtructure_insert_time
        )
        results["xtructure"].setdefault("lookup_ops_per_sec", []).append(
            batch_size / xtructure_lookup_time
        )

        # --- Python dict Benchmark ---
        py_dict: Dict[bytes, int] = {}
        python_insert_time = benchmark_dict_insert(py_dict, keys)
        # We already have the filled dict for lookup
        python_lookup_time = benchmark_dict_lookup(py_dict, keys)

        results["python"].setdefault("insert_ops_per_sec", []).append(
            batch_size / python_insert_time
        )
        results["python"].setdefault("lookup_ops_per_sec", []).append(
            batch_size / python_lookup_time
        )

    # Save results to the correct directory
    output_path = "xtructure_benchmarks/results/hashtable_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"HashTable benchmark results saved to {output_path}")
    print_results_table(results, "HashTable Performance Results")


if __name__ == "__main__":
    run_benchmarks()
