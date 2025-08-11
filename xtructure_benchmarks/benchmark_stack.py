import dataclasses
import json
import time
from typing import Any, Dict

import jax

from xtructure import Stack
from xtructure_benchmarks.common import BenchmarkValue, print_results_table


def benchmark_xtructure_stack_push(stack: Stack, values: BenchmarkValue, trials: int = 10):
    """Benchmarks the batched push operation for xtructure.Stack."""

    def push_op():
        return stack.push(values)

    # JIT compile and warm up
    jitted_push = jax.jit(push_op)
    new_stack = jitted_push()
    jax.block_until_ready(new_stack)

    # Time multiple trials
    times = []
    for _ in range(trials):
        start_time = time.perf_counter()
        new_stack = jitted_push()
        jax.block_until_ready(new_stack)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    import numpy as np

    times = np.array(times)
    median_time = np.median(times)
    q75, q25 = np.percentile(times, [75, 25])
    iqr_time = q75 - q25

    return median_time, iqr_time


def benchmark_xtructure_stack_pop(
    stack: Stack, count: int, trials: int = 10, include_host_transfer: bool = False
):
    """Benchmarks the batched pop operation for xtructure.Stack."""

    def pop_op():
        return stack.pop(count)

    # JIT compile and warm up
    jitted_pop = jax.jit(pop_op)
    new_stack, _ = jitted_pop()
    jax.block_until_ready(new_stack)

    # Time multiple trials
    times = []
    for _ in range(trials):
        start_time = time.perf_counter()
        new_stack, popped_values = jitted_pop()
        jax.block_until_ready(new_stack)
        if include_host_transfer:
            # Include device-to-host transfer cost outside JIT
            _ = jax.device_get(popped_values)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    import numpy as np

    times = np.array(times)
    median_time = np.median(times)
    q75, q25 = np.percentile(times, [75, 25])
    iqr_time = q75 - q25

    return median_time, iqr_time


def benchmark_list_push(values: BenchmarkValue, trials: int = 10):
    """Benchmarks the extend (batched push) operation for a Python list."""
    dict_of_arrays = dataclasses.asdict(values)
    items = [dict(zip(dict_of_arrays, t)) for t in zip(*dict_of_arrays.values())]

    times = []
    for _ in range(trials):
        lst = []  # Fresh list per trial
        start_time = time.perf_counter()
        lst.extend(items)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    import numpy as np

    times = np.array(times)
    median_time = np.median(times)
    q75, q25 = np.percentile(times, [75, 25])
    iqr_time = q75 - q25

    return median_time, iqr_time


def benchmark_list_pop(values: BenchmarkValue, count: int, trials: int = 10):
    """Benchmarks the pop operation for a Python list."""
    dict_of_arrays = dataclasses.asdict(values)
    items = [dict(zip(dict_of_arrays, t)) for t in zip(*dict_of_arrays.values())]

    times = []
    for _ in range(trials):
        lst = items[:]  # Fresh filled list per trial
        start_time = time.perf_counter()
        results = []
        for _ in range(count):
            if lst:
                results.append(lst.pop())
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    import numpy as np

    times = np.array(times)
    median_time = np.median(times)
    q75, q25 = np.percentile(times, [75, 25])
    iqr_time = q75 - q25

    return median_time, iqr_time


def run_benchmarks():
    """Runs the full suite of Stack benchmarks and saves the results."""
    batch_sizes = [2**10, 2**12, 2**14]
    results: Dict[str, Any] = {"batch_sizes": batch_sizes, "xtructure": {}, "python": {}}
    max_size = int(max(batch_sizes) * 2)

    print("Running Stack Benchmarks...")
    for batch_size in batch_sizes:
        print(f"  Batch Size: {batch_size}")
        key = jax.random.PRNGKey(batch_size)
        values = BenchmarkValue.random(shape=(batch_size,), key=key)

        # --- xtructure.Stack Benchmark ---
        xtructure_stack = Stack.build(max_size=max_size, value_class=BenchmarkValue)
        xtructure_push_median, xtructure_push_iqr = benchmark_xtructure_stack_push(
            xtructure_stack, values
        )

        # Create a filled stack for pop benchmark
        stack_with_data = xtructure_stack.push(values)
        jax.block_until_ready(stack_with_data)
        xtructure_pop_median, xtructure_pop_iqr = benchmark_xtructure_stack_pop(
            stack_with_data, batch_size
        )

        results["xtructure"].setdefault("push_ops_per_sec", []).append(
            {
                "median": batch_size / xtructure_push_median if xtructure_push_median > 0 else 0,
                "iqr": batch_size * xtructure_push_iqr / (xtructure_push_median**2)
                if xtructure_push_median > 0
                else 0,
            }
        )
        results["xtructure"].setdefault("pop_ops_per_sec", []).append(
            {
                "median": batch_size / xtructure_pop_median if xtructure_pop_median > 0 else 0,
                "iqr": batch_size * xtructure_pop_iqr / (xtructure_pop_median**2)
                if xtructure_pop_median > 0
                else 0,
            }
        )

        # --- Python list as Stack Benchmark ---
        python_push_median, python_push_iqr = benchmark_list_push(values)
        python_pop_median, python_pop_iqr = benchmark_list_pop(values, batch_size)

        results["python"].setdefault("push_ops_per_sec", []).append(
            {
                "median": batch_size / python_push_median if python_push_median > 0 else 0,
                "iqr": batch_size * python_push_iqr / (python_push_median**2)
                if python_push_median > 0
                else 0,
            }
        )
        results["python"].setdefault("pop_ops_per_sec", []).append(
            {
                "median": batch_size / python_pop_median if python_pop_median > 0 else 0,
                "iqr": batch_size * python_pop_iqr / (python_pop_median**2)
                if python_pop_median > 0
                else 0,
            }
        )

    # Save results
    output_path = "xtructure_benchmarks/results/stack_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Stack benchmark results saved to {output_path}")
    print_results_table(results, "Stack Performance Results")


if __name__ == "__main__":
    run_benchmarks()
