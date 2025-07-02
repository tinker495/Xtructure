#!/usr/bin/env python3
"""
Simplified benchmarks for Xtructure data structures vs standard Python equivalents.

This script benchmarks:
1. Stack (Xtructure) vs list (Python)
2. Queue (Xtructure) vs deque (Python) 
3. HashTable (Xtructure) vs dict (Python)

Note: BGPQ is skipped due to CPU limitations with Pallas operations.

Usage:
    python3 simple_benchmarks.py
"""

import time
import statistics
from collections import deque
from typing import List
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from xtructure import Stack, Queue, HashTable
from xtructure import xtructure_dataclass, FieldDescriptor


# Configure JAX for benchmarking
jax.config.update("jax_platform_name", "cpu")  # Use CPU for fair comparison
jax.config.update("jax_enable_x64", True)


@xtructure_dataclass
class BenchmarkValue:
    """Test data structure for benchmarks"""
    id: FieldDescriptor[jnp.uint32, (), 0]
    data: FieldDescriptor[jnp.float32, (4,), 0.0]  # Small array data
    flag: FieldDescriptor[jnp.bool_, (), False]


@dataclass
class BenchmarkResult:
    """Results from a single benchmark test"""
    operation: str
    data_structure: str
    size: int
    time_ms: float
    throughput_ops_per_sec: float


class SimpleBenchmarkSuite:
    """Simplified benchmarking suite"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.test_sizes = [100, 1000, 5000, 10000]
        self.num_iterations = 3
        
    def time_operation(self, func, *args, **kwargs) -> float:
        """Time a function call and return elapsed time in milliseconds"""
        times = []
        
        # Warmup
        for _ in range(1):
            try:
                func(*args, **kwargs)
            except:
                pass
            
        # Actual timing
        for _ in range(self.num_iterations):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            # For JAX operations, block until completion
            if hasattr(result, 'block_until_ready'):
                result.block_until_ready()
            elif isinstance(result, tuple) and len(result) > 0 and hasattr(result[0], 'block_until_ready'):
                for r in result:
                    if hasattr(r, 'block_until_ready'):
                        r.block_until_ready()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
            
        return statistics.mean(times)
    
    def benchmark_stack_operations(self):
        """Benchmark Stack vs Python list"""
        print("\n=== Stack vs Python List Benchmarks ===")
        
        for size in self.test_sizes:
            print(f"Testing size: {size}")
            
            # Generate test data
            test_values = BenchmarkValue(
                id=jnp.arange(size, dtype=jnp.uint32),
                data=jax.random.normal(jax.random.PRNGKey(42), (size, 4)).astype(jnp.float32),
                flag=jnp.ones(size, dtype=jnp.bool_)
            )
            
            # Xtructure Stack benchmarks
            stack = Stack.build(size, BenchmarkValue)
            
            # Push operations (batch)
            push_time = self.time_operation(lambda: stack.push(test_values))
            self.results.append(BenchmarkResult(
                "push_batch", "Xtructure_Stack", size, push_time, size/push_time*1000
            ))
            
            # Pop operations (batch)
            stack_full = stack.push(test_values)
            pop_time = self.time_operation(lambda: stack_full.pop(size))
            self.results.append(BenchmarkResult(
                "pop_batch", "Xtructure_Stack", size, pop_time, size/pop_time*1000
            ))
            
            # Python list benchmarks
            test_data = [(i, np.random.rand(4).astype(np.float32), True) for i in range(size)]
            
            # Push (append) operations
            py_list = []
            list_push_time = self.time_operation(
                lambda: [py_list.append(item) for item in test_data]
            )
            self.results.append(BenchmarkResult(
                "push_single", "Python_list", size, list_push_time, size/list_push_time*1000
            ))
            
            # Pop operations
            py_list_full = test_data.copy()
            list_pop_time = self.time_operation(
                lambda: [py_list_full.pop() for _ in range(len(py_list_full))]
            )
            self.results.append(BenchmarkResult(
                "pop_single", "Python_list", size, list_pop_time, size/list_pop_time*1000
            ))
    
    def benchmark_queue_operations(self):
        """Benchmark Queue vs Python deque"""
        print("\n=== Queue vs Python deque Benchmarks ===")
        
        for size in self.test_sizes:
            print(f"Testing size: {size}")
            
            # Generate test data
            test_values = BenchmarkValue(
                id=jnp.arange(size, dtype=jnp.uint32),
                data=jax.random.normal(jax.random.PRNGKey(42), (size, 4)).astype(jnp.float32),
                flag=jnp.ones(size, dtype=jnp.bool_)
            )
            
            # Xtructure Queue benchmarks
            queue = Queue.build(size, BenchmarkValue)
            
            # Enqueue operations (batch)
            enqueue_time = self.time_operation(lambda: queue.enqueue(test_values))
            self.results.append(BenchmarkResult(
                "enqueue_batch", "Xtructure_Queue", size, enqueue_time, size/enqueue_time*1000
            ))
            
            # Dequeue operations (batch)
            queue_full = queue.enqueue(test_values)
            dequeue_time = self.time_operation(lambda: queue_full.dequeue(size))
            self.results.append(BenchmarkResult(
                "dequeue_batch", "Xtructure_Queue", size, dequeue_time, size/dequeue_time*1000
            ))
            
            # Python deque benchmarks
            py_deque = deque()
            test_data = [(i, np.random.rand(4).astype(np.float32), True) for i in range(size)]
            
            # Enqueue (append) operations
            deque_enqueue_time = self.time_operation(
                lambda: [py_deque.append(item) for item in test_data]
            )
            self.results.append(BenchmarkResult(
                "enqueue_single", "Python_deque", size, deque_enqueue_time, size/deque_enqueue_time*1000
            ))
            
            # Dequeue operations
            py_deque_full = deque(test_data)
            deque_dequeue_time = self.time_operation(
                lambda: [py_deque_full.popleft() for _ in range(len(py_deque_full))]
            )
            self.results.append(BenchmarkResult(
                "dequeue_single", "Python_deque", size, deque_dequeue_time, size/deque_dequeue_time*1000
            ))
    
    def benchmark_hashtable_operations(self):
        """Benchmark HashTable vs Python dict"""
        print("\n=== HashTable vs Python dict Benchmarks ===")
        
        for size in self.test_sizes:
            print(f"Testing size: {size}")
            
            # Generate test data
            test_values = BenchmarkValue(
                id=jnp.arange(size, dtype=jnp.uint32),
                data=jax.random.normal(jax.random.PRNGKey(42), (size, 4)).astype(jnp.float32),
                flag=jnp.ones(size, dtype=jnp.bool_)
            )
            
            # Xtructure HashTable benchmarks
            hash_table = HashTable.build(BenchmarkValue, seed=42, capacity=size * 2)
            
            # Insert operations (parallel)
            insert_time = self.time_operation(
                lambda: hash_table.parallel_insert(test_values)
            )
            self.results.append(BenchmarkResult(
                "insert_parallel", "Xtructure_HashTable", size, insert_time, size/insert_time*1000
            ))
            
            # Lookup operations (parallel)
            hash_table_full, _, _, _ = hash_table.parallel_insert(test_values)
            lookup_indices = jnp.arange(0, size, max(1, size // 100))  # Sample 100 lookups
            lookup_values = BenchmarkValue(
                id=test_values.id[lookup_indices],
                data=test_values.data[lookup_indices],
                flag=test_values.flag[lookup_indices]
            )
            
            lookup_time = self.time_operation(
                lambda: hash_table_full.lookup_parallel(lookup_values)
            )
            self.results.append(BenchmarkResult(
                "lookup_parallel", "Xtructure_HashTable", len(lookup_indices), lookup_time, 
                len(lookup_indices)/lookup_time*1000
            ))
            
            # Python dict benchmarks
            py_dict = {}
            test_data = {i: (i, np.random.rand(4).astype(np.float32), True) for i in range(size)}
            
            # Insert operations
            dict_insert_time = self.time_operation(
                lambda: py_dict.update(test_data)
            )
            self.results.append(BenchmarkResult(
                "insert_batch", "Python_dict", size, dict_insert_time, size/dict_insert_time*1000
            ))
            
            # Lookup operations
            py_dict_full = test_data.copy()
            lookup_keys = list(range(0, size, max(1, size // 100)))
            
            dict_lookup_time = self.time_operation(
                lambda: [py_dict_full.get(key) for key in lookup_keys]
            )
            self.results.append(BenchmarkResult(
                "lookup_batch", "Python_dict", len(lookup_keys), dict_lookup_time, 
                len(lookup_keys)/dict_lookup_time*1000
            ))
    
    def run_all_benchmarks(self):
        """Run all benchmark suites"""
        print("Starting simplified Xtructure benchmarks...")
        print(f"JAX backend: {jax.default_backend()}")
        print(f"JAX devices: {jax.devices()}")
        
        # Run all benchmark suites
        self.benchmark_stack_operations()
        self.benchmark_queue_operations()
        self.benchmark_hashtable_operations()
        
        # Print results summary
        self.print_results_summary()
        
        # Save results to file
        self.save_results_to_file()
    
    def print_results_summary(self):
        """Print formatted benchmark results"""
        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*80)
        
        # Group results by data structure
        structure_groups = {}
        for result in self.results:
            key = (result.data_structure, result.operation)
            if key not in structure_groups:
                structure_groups[key] = []
            structure_groups[key].append(result)
        
        for (structure, operation), results in sorted(structure_groups.items()):
            print(f"\n{structure} - {operation}:")
            print(f"{'Size':<10} {'Time (ms)':<15} {'Throughput (ops/sec)':<20}")
            print("-" * 50)
            
            for result in sorted(results, key=lambda x: x.size):
                print(f"{result.size:<10} {result.time_ms:<15.2f} {result.throughput_ops_per_sec:<20.0f}")
        
        # Performance comparison summary
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON (Xtructure vs Python)")
        print("="*80)
        
        comparisons = self._calculate_performance_comparisons()
        for comparison in comparisons:
            print(comparison)
    
    def _calculate_performance_comparisons(self) -> List[str]:
        """Calculate performance comparisons between Xtructure and Python implementations"""
        comparisons = []
        
        # Group results for comparison
        xt_results = {(r.operation, r.size): r for r in self.results if "Xtructure" in r.data_structure}
        py_results = {(r.operation.replace("_batch", "_single").replace("_parallel", "_batch"), r.size): r 
                     for r in self.results if "Python" in r.data_structure}
        
        for (operation, size), xt_result in xt_results.items():
            # Find corresponding Python result
            py_key_candidates = [
                (operation.replace("_batch", "_single"), size),
                (operation.replace("_parallel", "_batch"), size),
                (operation, size)
            ]
            
            py_result = None
            for key in py_key_candidates:
                if key in py_results:
                    py_result = py_results[key]
                    break
            
            if py_result:
                speedup = py_result.throughput_ops_per_sec / xt_result.throughput_ops_per_sec
                if speedup > 1:
                    comparisons.append(f"{xt_result.data_structure} {operation} (size {size}): "
                                     f"{speedup:.2f}x SLOWER than {py_result.data_structure}")
                else:
                    comparisons.append(f"{xt_result.data_structure} {operation} (size {size}): "
                                     f"{1/speedup:.2f}x FASTER than {py_result.data_structure}")
        
        return comparisons
    
    def save_results_to_file(self):
        """Save benchmark results to CSV file"""
        import csv
        
        filename = f"xtructure_simple_benchmarks_{int(time.time())}.csv"
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['operation', 'data_structure', 'size', 'time_ms', 'throughput_ops_per_sec']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in self.results:
                writer.writerow({
                    'operation': result.operation,
                    'data_structure': result.data_structure,
                    'size': result.size,
                    'time_ms': result.time_ms,
                    'throughput_ops_per_sec': result.throughput_ops_per_sec
                })
        
        print(f"\nResults saved to: {filename}")


def main():
    """Main function to run benchmarks"""
    benchmark_suite = SimpleBenchmarkSuite()
    benchmark_suite.run_all_benchmarks()


if __name__ == "__main__":
    main()