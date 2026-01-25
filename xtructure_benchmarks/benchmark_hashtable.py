import argparse
import json
import os
from typing import Any, Dict, List, Optional

import jax

from xtructure import HashTable
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


def benchmark_dict_lookup(
    data_dict: Dict,
    keys: BenchmarkValue,
    trials: int = 10,
    include_preprocessing: bool = False,
):
    """Benchmarks lookup from a standard Python dict using Python objects."""

    # Precompute Python objects for kernel mode
    precomputed_keys = to_python_values(keys)

    def lookup_op():
        # In e2e mode, we include the cost of converting JAX arrays to Python objects
        # mimicking the overhead of "receiving" data
        py_keys = to_python_values(keys) if include_preprocessing else precomputed_keys

        # Optimized bulk lookup using map (faster than list comprehension for built-in methods)
        return list(map(data_dict.get, py_keys))

    results_tuple = run_python_trials(lookup_op, trials)
    return throughput_stats(len(precomputed_keys), results_tuple)


def run_benchmarks(mode: str = "kernel", trials: int = 10, batch_sizes: Optional[List[int]] = None):
    """Runs the full suite of HashTable benchmarks and saves the results."""
    # Using smaller batch sizes to ensure completion within a reasonable time
    if batch_sizes is None:
        batch_sizes = [2**10, 2**12, 2**14]

    check_system_load()

    results: Dict[str, Any] = {
        "batch_sizes": batch_sizes,
        "xtructure": {},
        "python": {},
        "environment": get_system_info(),
    }
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
        keys_device = BenchmarkValue.random(shape=(batch_size,), key=key)
        keys_host = jax.device_get(keys_device)

        # Keep table size proportional to batch size for constant load factor
        table_size = int(batch_size * load_factor_inverse)

        # --- xtructure.HashTable Benchmark ---
        xtructure_table = HashTable.build(BenchmarkValue, 1, table_size)

        def insert_op(table, batch):
            # The key and value are the same in this benchmark
            return table.parallel_insert(batch)[0]

        def insert_args_supplier():
            return (
                xtructure_table,
                jax.device_put(keys_host) if mode == "e2e" else keys_device,
            )

        insert_durations = run_jax_trials(
            insert_op,
            trials=trials,
            args_supplier=insert_args_supplier,
        )
        xtructure_insert_stats = throughput_stats(batch_size, insert_durations)

        # Re-build and insert for a fair lookup comparison
        table_with_data, _, _, _ = xtructure_table.parallel_insert(keys_device)
        jax.block_until_ready(table_with_data)

        def lookup_op(table, batch):
            return table.lookup_parallel(batch)

        def lookup_args_supplier():
            return (
                table_with_data,
                jax.device_put(keys_host) if mode == "e2e" else keys_device,
            )

        lookup_durations = run_jax_trials(
            lookup_op,
            trials=trials,
            include_device_transfer=(mode == "e2e"),
            args_supplier=lookup_args_supplier,
        )
        xtructure_lookup_stats = throughput_stats(batch_size, lookup_durations)

        results["xtructure"].setdefault("insert_ops_per_sec", []).append(xtructure_insert_stats)
        results["xtructure"].setdefault("lookup_ops_per_sec", []).append(xtructure_lookup_stats)

        # --- Python dict Benchmark ---
        # Test both incremental and bulk modes, report the better one
        python_insert_candidates = []

        def _python_insert_trial(bulk_mode: bool):
            # Precompute Python objects
            precomputed = to_python_values(keys_device)

            def key_supplier():
                if mode == "e2e":
                    return to_python_values(keys_device)
                return precomputed

            def op():
                kb = key_supplier()
                if bulk_mode:
                    return {key: key for key in kb}
                data_dict = {}
                for key in kb:
                    data_dict[key] = key
                return data_dict

            results_tuple = run_python_trials(op, trials=trials)
            stats = throughput_stats(batch_size, results_tuple)
            return stats, op

        for candidate_bulk in (False, True):
            stats, op = _python_insert_trial(candidate_bulk)
            python_insert_candidates.append((stats, op, candidate_bulk))

        python_insert_candidates.sort(key=lambda t: t[0]["median"], reverse=True)
        best_python_insert_stats, best_insert_op, best_is_bulk = python_insert_candidates[0]
        py_dict = best_insert_op()

        python_lookup_stats = benchmark_dict_lookup(
            py_dict, keys_device, trials=trials, include_preprocessing=(mode == "e2e")
        )

        results["python"].setdefault("insert_ops_per_sec", []).append(best_python_insert_stats)
        results["python"].setdefault("lookup_ops_per_sec", []).append(python_lookup_stats)

    # Validate and save results to the correct directory
    validate_results_schema(results)
    output_path = "xtructure_benchmarks/results/hashtable_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
