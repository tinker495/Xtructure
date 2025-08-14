import argparse
import json
import time
from typing import Any, Dict, List, Optional

import jax

from xtructure import HashTable
from xtructure_benchmarks.common import (
    BenchmarkValue,
    print_results_table,
    python_timer,
    validate_results_schema,
)


def benchmark_hashtable_insert(table: HashTable, keys: BenchmarkValue, trials: int = 10):
    """Benchmarks the parallel insertion into the xtructure HashTable."""

    def insert_op():
        # The key and value are the same in this benchmark
        return table.parallel_insert(keys)[0]

    # JIT compile and warm up
    jitted_insert = jax.jit(insert_op)
    new_table = jitted_insert()
    jax.block_until_ready(new_table)

    # Time multiple trials
    times = []
    for _ in range(trials):
        start_time = time.perf_counter()
        new_table = jitted_insert()
        jax.block_until_ready(new_table)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    import numpy as np

    times = np.array(times)
    median_time = np.median(times)
    q75, q25 = np.percentile(times, [75, 25])
    iqr_time = q75 - q25

    return median_time, iqr_time


def benchmark_hashtable_lookup(
    table: HashTable, keys: BenchmarkValue, trials: int = 10, include_host_transfer: bool = False
):
    """Benchmarks the parallel lookup from the xtructure HashTable."""

    def lookup_op():
        return table.lookup_parallel(keys)

    # JIT compile and warm up
    jitted_lookup = jax.jit(lookup_op)
    result = jitted_lookup()
    jax.block_until_ready(result)

    # Time multiple trials
    times = []
    for _ in range(trials):
        start_time = time.perf_counter()
        result = jitted_lookup()
        jax.block_until_ready(result)
        if include_host_transfer:
            # Include device-to-host transfer cost outside JIT
            _ = jax.device_get(result)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    import numpy as np

    times = np.array(times)
    median_time = np.median(times)
    q75, q25 = np.percentile(times, [75, 25])
    iqr_time = q75 - q25

    return median_time, iqr_time


def benchmark_dict_insert(
    keys: BenchmarkValue,
    trials: int = 10,
    bulk_mode: bool = False,
    include_preprocessing: bool = False,
):
    """Benchmarks insertion into a standard Python dict."""
    # Optionally include preprocessing (conversion to bytes) in the timed region
    if not include_preprocessing:
        key_bytes = [arr.tobytes() for arr in jax.vmap(lambda x: x.bytes)(keys)]

    times = []
    for _ in range(trials):
        data_dict = {}  # Fresh dict per trial
        if bulk_mode:
            start_time = time.perf_counter()
            kb = (
                [arr.tobytes() for arr in jax.vmap(lambda x: x.bytes)(keys)]
                if include_preprocessing
                else key_bytes
            )
            data_dict.update({key: key for key in kb})
            end_time = time.perf_counter()
        else:
            start_time = time.perf_counter()
            kb = (
                [arr.tobytes() for arr in jax.vmap(lambda x: x.bytes)(keys)]
                if include_preprocessing
                else key_bytes
            )
            for key in kb:
                data_dict[key] = key  # Store key as value for payload parity
            end_time = time.perf_counter()
        times.append(end_time - start_time)

    import numpy as np

    times = np.array(times)
    median_time = np.median(times)
    q75, q25 = np.percentile(times, [75, 25])
    iqr_time = q75 - q25

    return median_time, iqr_time


def benchmark_dict_lookup(
    data_dict: Dict,
    keys: BenchmarkValue,
    trials: int = 10,
    include_preprocessing: bool = False,
):
    """Benchmarks lookup from a standard Python dict."""
    # Optionally include preprocessing (conversion to bytes) in the timed region
    if not include_preprocessing:
        key_bytes = [arr.tobytes() for arr in jax.vmap(lambda x: x.bytes)(keys)]

    def lookup_op():
        results = []
        kb = (
            [arr.tobytes() for arr in jax.vmap(lambda x: x.bytes)(keys)]
            if include_preprocessing
            else key_bytes
        )
        for key in kb:
            results.append(data_dict.get(key))
        # Return results to match xtructure lookup semantics
        return results

    return python_timer(lookup_op, trials)


def run_benchmarks(mode: str = "kernel", trials: int = 10, batch_sizes: Optional[List[int]] = None):
    """Runs the full suite of HashTable benchmarks and saves the results."""
    # Using smaller batch sizes to ensure completion within a reasonable time
    if batch_sizes is None:
        batch_sizes = [2**10, 2**12, 2**14]
    results: Dict[str, Any] = {"batch_sizes": batch_sizes, "xtructure": {}, "python": {}}
    load_factor_inverse = 1.5  # Keep load factor constant across batch sizes

    print("Running HashTable Benchmarks...")
    try:
        print(f"JAX backend: {jax.default_backend()}")
        print("JAX devices:", ", ".join([d.platform + ":" + d.device_kind for d in jax.devices()]))
    except Exception:
        pass
    for batch_size in batch_sizes:
        print(f"  Batch Size: {batch_size}")
        key = jax.random.PRNGKey(0)
        keys = BenchmarkValue.random(shape=(batch_size,), key=key)

        # Keep table size proportional to batch size for constant load factor
        table_size = int(batch_size * load_factor_inverse)

        # --- xtructure.HashTable Benchmark ---
        xtructure_table = HashTable.build(BenchmarkValue, 1, table_size)
        xtructure_insert_median, xtructure_insert_iqr = benchmark_hashtable_insert(
            xtructure_table, keys, trials=trials
        )

        # Re-build and insert for a fair lookup comparison
        table_with_data, _, _, _ = xtructure_table.parallel_insert(keys)
        jax.block_until_ready(table_with_data)
        xtructure_lookup_median, xtructure_lookup_iqr = benchmark_hashtable_lookup(
            table_with_data, keys, trials=trials, include_host_transfer=(mode == "e2e")
        )

        results["xtructure"].setdefault("insert_ops_per_sec", []).append(
            {
                "median": batch_size / xtructure_insert_median,
                "iqr": batch_size * xtructure_insert_iqr / (xtructure_insert_median**2),
            }
        )
        results["xtructure"].setdefault("lookup_ops_per_sec", []).append(
            {
                "median": batch_size / xtructure_lookup_median,
                "iqr": batch_size * xtructure_lookup_iqr / (xtructure_lookup_median**2),
            }
        )

        # --- Python dict Benchmark ---
        # Test both incremental and bulk modes, report the better one
        python_insert_incremental_median, python_insert_incremental_iqr = benchmark_dict_insert(
            keys, trials=trials, bulk_mode=False, include_preprocessing=(mode == "e2e")
        )
        python_insert_bulk_median, python_insert_bulk_iqr = benchmark_dict_insert(
            keys, trials=trials, bulk_mode=True, include_preprocessing=(mode == "e2e")
        )

        # Use the better (faster) Python baseline
        if python_insert_incremental_median < python_insert_bulk_median:
            python_insert_median, python_insert_iqr = (
                python_insert_incremental_median,
                python_insert_incremental_iqr,
            )
            # Build dict for lookup using faster method
            py_dict: Dict[bytes, bytes] = {}
            key_bytes = [arr.tobytes() for arr in jax.vmap(lambda x: x.bytes)(keys)]
            for key in key_bytes:
                py_dict[key] = key
        else:
            python_insert_median, python_insert_iqr = (
                python_insert_bulk_median,
                python_insert_bulk_iqr,
            )
            # Build dict for lookup using faster method
            key_bytes = [arr.tobytes() for arr in jax.vmap(lambda x: x.bytes)(keys)]
            py_dict = {key: key for key in key_bytes}

        python_lookup_median, python_lookup_iqr = benchmark_dict_lookup(
            py_dict, keys, trials=trials, include_preprocessing=(mode == "e2e")
        )

        results["python"].setdefault("insert_ops_per_sec", []).append(
            {
                "median": batch_size / python_insert_median,
                "iqr": batch_size * python_insert_iqr / (python_insert_median**2),
            }
        )
        results["python"].setdefault("lookup_ops_per_sec", []).append(
            {
                "median": batch_size / python_lookup_median,
                "iqr": batch_size * python_lookup_iqr / (python_lookup_median**2),
            }
        )

    # Validate and save results to the correct directory
    validate_results_schema(results)
    output_path = "xtructure_benchmarks/results/hashtable_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"HashTable benchmark results saved to {output_path}")
    print_results_table(results, "HashTable Performance Results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HashTable benchmarks")
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
