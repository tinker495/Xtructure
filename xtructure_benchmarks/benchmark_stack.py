import dataclasses
import json
import time
from typing import Any, Dict, List

import jax

from xtructure import Stack
from xtructure_benchmarks.common import (
    BenchmarkValue,
    print_results_table,
    python_timer,
)


def benchmark_xtructure_stack_push(stack: Stack, values: BenchmarkValue):
    """Benchmarks the batched push operation for xtructure.Stack."""

    def push_op():
        return stack.push(values)

    # JIT compile and warm up
    jitted_push = jax.jit(push_op)
    new_stack = jitted_push()
    jax.block_until_ready(new_stack)

    # Time the actual execution
    start_time = time.perf_counter()
    new_stack = jitted_push()
    jax.block_until_ready(new_stack)
    end_time = time.perf_counter()

    return end_time - start_time


def benchmark_xtructure_stack_pop(stack: Stack, count: int):
    """Benchmarks the batched pop operation for xtructure.Stack."""

    def pop_op():
        return stack.pop(count)

    # JIT compile and warm up
    jitted_pop = jax.jit(pop_op)
    new_stack, _ = jitted_pop()
    jax.block_until_ready(new_stack)

    # Time the actual execution
    start_time = time.perf_counter()
    new_stack, _ = jitted_pop()
    jax.block_until_ready(new_stack)
    end_time = time.perf_counter()

    return end_time - start_time


def benchmark_list_push(lst: List, values: BenchmarkValue):
    """Benchmarks the extend (batched push) operation for a Python list."""
    dict_of_arrays = dataclasses.asdict(values)
    items = [dict(zip(dict_of_arrays, t)) for t in zip(*dict_of_arrays.values())]

    def push_op():
        lst.extend(items)

    return python_timer(push_op)


def benchmark_list_pop(lst: List, count: int):
    """Benchmarks the pop operation for a Python list."""

    def pop_op():
        for _ in range(count):
            if lst:
                lst.pop()

    return python_timer(pop_op)


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
        xtructure_push_time = benchmark_xtructure_stack_push(xtructure_stack, values)

        # Create a filled stack for pop benchmark
        stack_with_data = xtructure_stack.push(values)
        jax.block_until_ready(stack_with_data)
        xtructure_pop_time = benchmark_xtructure_stack_pop(stack_with_data, batch_size)

        results["xtructure"].setdefault("push_ops_per_sec", []).append(
            batch_size / xtructure_push_time if xtructure_push_time > 0 else 0
        )
        results["xtructure"].setdefault("pop_ops_per_sec", []).append(
            batch_size / xtructure_pop_time if xtructure_pop_time > 0 else 0
        )

        # --- Python list as Stack Benchmark ---
        py_list: List[Dict] = []
        python_push_time = benchmark_list_push(py_list, values)

        # We already have the filled list for pop
        python_pop_time = benchmark_list_pop(py_list, batch_size)

        results["python"].setdefault("push_ops_per_sec", []).append(
            batch_size / python_push_time if python_push_time > 0 else 0
        )
        results["python"].setdefault("pop_ops_per_sec", []).append(
            batch_size / python_pop_time if python_pop_time > 0 else 0
        )

    # Save results
    output_path = "xtructure_benchmarks/results/stack_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Stack benchmark results saved to {output_path}")
    print_results_table(results, "Stack Performance Results")


if __name__ == "__main__":
    run_benchmarks()
