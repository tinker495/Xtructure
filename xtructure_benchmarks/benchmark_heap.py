import argparse
import heapq
from typing import List, Optional

import jax
import jax.numpy as jnp

from xtructure import BGPQ
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


# Key generation function adapted from tests/heap_test.py
# Improved to reduce ties by using full hash range
@jax.jit
def key_gen(x: BenchmarkValue) -> float:
    uint32_hash = x.hash()
    # Map to [0,1) with much fewer ties
    key = uint32_hash / jnp.float32(2**32)
    return key.astype(jnp.float32)


vmapped_key_gen = jax.jit(jax.vmap(key_gen))


def benchmark_bgpq_insert(
    heap: BGPQ,
    keys: jnp.ndarray,
    values: BenchmarkValue,
    trials: int = 10,
    args_supplier=None,
):
    """Benchmarks the batched insertion into the xtructure BGPQ."""

    durations = run_jax_trials(
        lambda batch_keys, batch_values: heap.insert(batch_keys, batch_values),
        trials=trials,
        args_supplier=args_supplier or (lambda: (keys, values)),
    )
    return durations


def benchmark_bgpq_delete(
    heap: BGPQ, trials: int = 10, include_host_transfer: bool = False
):
    """Benchmarks the batched deletion from the xtructure BGPQ."""

    durations = run_jax_trials(
        lambda: BGPQ.delete_mins(heap),
        trials=trials,
        include_device_transfer=include_host_transfer,
    )
    return durations


def benchmark_heapq_insert(
    keys: jnp.ndarray,
    values: BenchmarkValue,
    trials: int = 10,
    bulk_mode: bool = False,
    include_preprocessing: bool = False,
):
    """Benchmarks insertion into Python's heapq."""
    # Precompute equivalent Python objects
    cached_values = to_python_values(values)
    cached_keys = jax.device_get(keys).tolist()

    def payload_supplier():
        if include_preprocessing:
            return (
                jax.device_get(keys).tolist(),
                to_python_values(values),
            )
        return cached_keys, cached_values

    def bulk_op():
        nk, vals = payload_supplier()
        # Python's heapq stores (priority, tie_breaker, value)
        items = [(nk[i], i, vals[i]) for i in range(len(nk))]
        heapq.heapify(items)
        return items

    def incremental_op():
        nk, vals = payload_supplier()
        data_heap = []  # Fresh heap per trial
        for i in range(len(nk)):
            heapq.heappush(data_heap, (nk[i], i, vals[i]))
        return data_heap

    op = bulk_op if bulk_mode else incremental_op
    results_tuple = run_python_trials(op, trials=trials)
    return throughput_stats(len(cached_keys), results_tuple), op


def benchmark_heapq_delete(prototype_heap: List, count: int, trials: int = 10):
    """Benchmarks deletion from Python's heapq."""

    def delete_op():
        data_heap = prototype_heap.copy()  # Fresh copy per trial
        results = []
        for _ in range(count):
            if data_heap:
                results.append(heapq.heappop(data_heap))
        return results

    results_tuple = run_python_trials(delete_op, trials=trials)
    return throughput_stats(count, results_tuple)


def run_benchmarks(
    mode: str = "kernel",
    trials: int = 10,
    batch_sizes: Optional[List[int]] = None,
    python_heap_insert_mode: str = "auto",
):
    """Runs the full suite of Heap benchmarks and saves the results."""
    # Using smaller batch sizes to ensure completion within a reasonable time
    # Using smaller batch sizes to ensure completion within a reasonable time
    batch_sizes = batch_sizes or [2**10, 2**12, 2**14]
    results = init_benchmark_results(batch_sizes)
    max_size = int(max(batch_sizes) * 1.5)

    print("Running Heap (BGPQ) Benchmarks...")
    try:
        print(f"JAX backend: {jax.default_backend()}")
        print(
            "JAX devices:",
            ", ".join([d.platform + ":" + d.device_kind for d in jax.devices()]),
        )
    except Exception:
        pass
    for batch_size in batch_sizes:
        print(f"  Batch Size: {batch_size}")
        key = jax.random.PRNGKey(batch_size)
        values_device = BenchmarkValue.random(shape=(batch_size,), key=key)
        keys_device = vmapped_key_gen(values_device)

        # --- xtructure.BGPQ Benchmark ---
        bgpq_heap = BGPQ.build(max_size, batch_size, BenchmarkValue, jnp.float32)

        # We need to make the keys/values batched to the BGPQ batch_size
        padded_keys, padded_values = BGPQ.make_batched(
            keys_device, values_device, batch_size
        )
        padded_keys_host = jax.device_get(padded_keys)
        padded_values_host = jax.device_get(padded_values)

        insert_args_supplier = (
            (
                lambda: (
                    jax.device_put(padded_keys_host),
                    jax.device_put(padded_values_host),
                )
            )
            if mode == "e2e"
            else (lambda: (padded_keys, padded_values))
        )
        insert_durations = benchmark_bgpq_insert(
            bgpq_heap,
            padded_keys,
            padded_values,
            trials=trials,
            args_supplier=insert_args_supplier,
        )
        xtructure_insert_stats = throughput_stats(batch_size, insert_durations)

        # Create a filled heap for deletion benchmark
        heap_with_data = bgpq_heap.insert(padded_keys, padded_values)
        jax.block_until_ready(heap_with_data)
        delete_durations = benchmark_bgpq_delete(
            heap_with_data, trials=trials, include_host_transfer=(mode == "e2e")
        )
        xtructure_delete_stats = throughput_stats(batch_size, delete_durations)

        results["xtructure"].setdefault("insert_ops_per_sec", []).append(
            xtructure_insert_stats
        )
        results["xtructure"].setdefault("delete_ops_per_sec", []).append(
            xtructure_delete_stats
        )

        # --- Python heapq Benchmark --- (fixed algorithm for fairness)
        insert_candidates = []
        modes_to_try = (
            [python_heap_insert_mode]
            if python_heap_insert_mode != "auto"
            else ["bulk", "incremental"]
        )
        for candidate in modes_to_try:
            use_bulk = candidate == "bulk"
            stats, op = benchmark_heapq_insert(
                keys_device,
                values_device,
                trials=trials,
                bulk_mode=use_bulk,
                include_preprocessing=(mode == "e2e"),
            )
            insert_candidates.append((stats, op))

        insert_candidates.sort(key=lambda t: t[0]["median"], reverse=True)
        best_insert_stats, build_heap_op = insert_candidates[0]
        py_heap = build_heap_op()

        python_delete_stats = benchmark_heapq_delete(py_heap, batch_size)

        results["python"].setdefault("insert_ops_per_sec", []).append(best_insert_stats)
        results["python"].setdefault("delete_ops_per_sec", []).append(
            python_delete_stats
        )

    # Validate and save results
    save_and_print_results(
        results,
        "xtructure_benchmarks/results/heap_results.json",
        "Heap (BGPQ) Performance Results",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heap (BGPQ) benchmarks")
    add_common_benchmark_args(parser)
    parser.add_argument(
        "--python-heap-insert-mode",
        choices=["auto", "bulk", "incremental"],
        default="auto",
        help="Choose Python heap insert algorithm for fairness",
    )
    args = parser.parse_args()

    run_benchmarks(
        mode=args.mode,
        trials=args.trials,
        batch_sizes=parse_batch_sizes(args.batch_sizes),
        python_heap_insert_mode=args.python_heap_insert_mode,
    )
