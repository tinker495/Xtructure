import argparse
from typing import List, Optional

import jax

from xtructure import Stack
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


def run_benchmarks(
    mode: str = "kernel", trials: int = 10, batch_sizes: Optional[List[int]] = None
):
    """Runs the full suite of Stack benchmarks and saves the results."""
    batch_sizes = batch_sizes or [2**10, 2**12, 2**14]
    results = init_benchmark_results(batch_sizes)
    max_size = int(max(batch_sizes) * 2)

    print("Running Stack Benchmarks...")
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
        values_host = jax.device_get(values_device)

        precomputed_items = to_python_values(values_device)

        def materialize_items(include_preprocessing: bool):
            return (
                to_python_values(values_device)
                if include_preprocessing
                else precomputed_items
            )

        # --- xtructure.Stack Benchmark ---
        xtructure_stack = Stack.build(max_size=max_size, value_class=BenchmarkValue)

        def push_args_supplier():
            if mode == "e2e":
                return (jax.device_put(values_host),)
            return (values_device,)

        push_durations = run_jax_trials(
            lambda batch: xtructure_stack.push(batch),
            trials=trials,
            args_supplier=push_args_supplier,
        )
        xtructure_push_stats = throughput_stats(batch_size, push_durations)

        # Create a filled stack for pop benchmark
        stack_with_data = xtructure_stack.push(values_device)
        jax.block_until_ready(stack_with_data)
        pop_durations = run_jax_trials(
            lambda: stack_with_data.pop(batch_size),
            trials=trials,
            include_device_transfer=(mode == "e2e"),
        )
        xtructure_pop_stats = throughput_stats(batch_size, pop_durations)

        results["xtructure"].setdefault("push_ops_per_sec", []).append(
            xtructure_push_stats
        )
        results["xtructure"].setdefault("pop_ops_per_sec", []).append(
            xtructure_pop_stats
        )

        # --- Python list as Stack Benchmark ---
        def python_push_op():
            items = materialize_items(mode == "e2e")
            lst = []
            lst.extend(items)
            return lst

        python_push_durations = run_python_trials(python_push_op, trials=trials)
        python_push_stats = throughput_stats(batch_size, python_push_durations)

        def python_pop_op():
            items = materialize_items(mode == "e2e")
            lst = list(items)
            popped = []
            for _ in range(batch_size):
                if lst:
                    popped.append(lst.pop())
            return popped

        python_pop_durations = run_python_trials(python_pop_op, trials=trials)
        python_pop_stats = throughput_stats(batch_size, python_pop_durations)

        results["python"].setdefault("push_ops_per_sec", []).append(python_push_stats)
        results["python"].setdefault("pop_ops_per_sec", []).append(python_pop_stats)

    save_and_print_results(
        results,
        "xtructure_benchmarks/results/stack_results.json",
        "Stack Performance Results",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stack benchmarks")
    add_common_benchmark_args(parser)
    args = parser.parse_args()

    run_benchmarks(
        mode=args.mode,
        trials=args.trials,
        batch_sizes=parse_batch_sizes(args.batch_sizes),
    )
