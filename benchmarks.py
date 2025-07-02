#!/usr/bin/env python3
"""
Comprehensive benchmarks for Xtructure data structures vs standard Python equivalents.

This script benchmarks:
1. Stack (Xtructure) vs list (Python)
2. Queue (Xtructure) vs deque (Python) 
3. BGPQ (Xtructure) vs heapq (Python)
4. HashTable (Xtructure) vs dict (Python)

Usage:
    python benchmarks.py
"""

import time
import heapq
import statistics
from collections import deque
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from xtructure import Stack, Queue, BGPQ, HashTable
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
    memory_mb: float = 0.0


class BenchmarkSuite:
    """Main benchmarking suite"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.test_sizes = [100, 1000, 5000, 10000, 50000]
        self.num_iterations = 5
        
    def time_operation(self, func, *args, **kwargs) -> float:
        """Time a function call and return elapsed time in milliseconds"""
        times = []
        
        # Warmup
        for _ in range(2):
            func(*args, **kwargs)
            
        # Actual timing
        for _ in range(self.num_iterations):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            # For JAX operations, block until completion
            if hasattr(result, 'block_until_ready'):
                result.block_until_ready()
            elif isinstance(result, tuple) and hasattr(result[0], 'block_until_ready'):
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
            print(f"\nTesting size: {size}")
            
            # Generate test data
            test_values = BenchmarkValue(
                id=jnp.arange(size, dtype=jnp.uint32),
                data=jax.random.normal(jax.random.PRNGKey(42), (size, 4)).astype(jnp.float32),
                flag=jnp.ones(size, dtype=jnp.bool_)
            )
            
            # Xtructure Stack benchmarks
            stack = Stack.build(size, BenchmarkValue)
            
            # Push operations
            push_time = self.time_operation(lambda: stack.push(test_values))
            self.results.append(BenchmarkResult(
                "push_batch", "Xtructure_Stack", size, push_time, size/push_time*1000
            ))
            
            # Individual push operations
            stack_single = Stack.build(size, BenchmarkValue)
            single_push_time = self.time_operation(
                lambda: self._push_single_items(stack_single, test_values, size)
            )
            self.results.append(BenchmarkResult(
                "push_single", "Xtructure_Stack", size, single_push_time, size/single_push_time*1000
            ))
            
            # Pop operations (after pushing)
            stack_full = stack.push(test_values)
            pop_time = self.time_operation(lambda: stack_full.pop(size))
            self.results.append(BenchmarkResult(
                "pop_batch", "Xtructure_Stack", size, pop_time, size/pop_time*1000
            ))
            
            # Python list benchmarks
            py_list = []
            
            # Push (append) operations
            test_data = [(i, np.random.rand(4).astype(np.float32), True) for i in range(size)]
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
    
    def _push_single_items(self, stack, test_values, size):
        """Helper to push items one by one"""
        current_stack = stack
        for i in range(size):
            item = BenchmarkValue(
                id=test_values.id[i:i+1],
                data=test_values.data[i:i+1],
                flag=test_values.flag[i:i+1]
            )
            current_stack = current_stack.push(item)
        return current_stack
    
    def benchmark_queue_operations(self):
        """Benchmark Queue vs Python deque"""
        print("\n=== Queue vs Python deque Benchmarks ===")
        
        for size in self.test_sizes:
            print(f"\nTesting size: {size}")
            
            # Generate test data
            test_values = BenchmarkValue(
                id=jnp.arange(size, dtype=jnp.uint32),
                data=jax.random.normal(jax.random.PRNGKey(42), (size, 4)).astype(jnp.float32),
                flag=jnp.ones(size, dtype=jnp.bool_)
            )
            
            # Xtructure Queue benchmarks
            queue = Queue.build(size, BenchmarkValue)
            
            # Enqueue operations
            enqueue_time = self.time_operation(lambda: queue.enqueue(test_values))
            self.results.append(BenchmarkResult(
                "enqueue_batch", "Xtructure_Queue", size, enqueue_time, size/enqueue_time*1000
            ))
            
            # Dequeue operations (after enqueuing)
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
    
    def benchmark_priority_queue_operations(self):
        """Benchmark BGPQ vs Python heapq"""
        print("\n=== BGPQ vs Python heapq Benchmarks ===")
        
        batch_size = 64  # BGPQ batch size
        
        for size in self.test_sizes:
            print(f"\nTesting size: {size}")
            
            # Generate test data
            keys = jax.random.uniform(jax.random.PRNGKey(42), (size,)).astype(jnp.float32)
            test_values = BenchmarkValue(
                id=jnp.arange(size, dtype=jnp.uint32),
                data=jax.random.normal(jax.random.PRNGKey(43), (size, 4)).astype(jnp.float32),
                flag=jnp.ones(size, dtype=jnp.bool_)
            )
            
            # Xtructure BGPQ benchmarks
            bgpq = BGPQ.build(size + batch_size, batch_size, BenchmarkValue)
            
            # Insert operations (batched)
            num_batches = (size + batch_size - 1) // batch_size
            insert_times = []
            
            current_bgpq = bgpq
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, size)
                batch_keys = keys[start_idx:end_idx]
                batch_values = BenchmarkValue(
                    id=test_values.id[start_idx:end_idx],
                    data=test_values.data[start_idx:end_idx],
                    flag=test_values.flag[start_idx:end_idx]
                )
                
                # Pad to batch size if needed
                if len(batch_keys) < batch_size:
                    pad_size = batch_size - len(batch_keys)
                    batch_keys = jnp.concatenate([
                        batch_keys, 
                        jnp.full(pad_size, jnp.inf, dtype=jnp.float32)
                    ])
                    batch_values = batch_values.padding_as_batch((batch_size,))
                
                insert_time = self.time_operation(
                    lambda bq=current_bgpq, k=batch_keys, v=batch_values: BGPQ.insert(bq, k, v)
                )
                current_bgpq = BGPQ.insert(current_bgpq, batch_keys, batch_values)
                insert_times.append(insert_time)
            
            avg_insert_time = statistics.mean(insert_times)
            self.results.append(BenchmarkResult(
                "insert_batch", "Xtructure_BGPQ", size, avg_insert_time, batch_size/avg_insert_time*1000
            ))
            
            # Delete min operations
            delete_times = []
            for _ in range(min(10, num_batches)):  # Test up to 10 deletions
                delete_time = self.time_operation(
                    lambda bq=current_bgpq: BGPQ.delete_mins(bq)
                )
                current_bgpq, _, _ = BGPQ.delete_mins(current_bgpq)
                delete_times.append(delete_time)
            
            if delete_times:
                avg_delete_time = statistics.mean(delete_times)
                self.results.append(BenchmarkResult(
                    "delete_min_batch", "Xtructure_BGPQ", batch_size, avg_delete_time, batch_size/avg_delete_time*1000
                ))
            
            # Python heapq benchmarks
            py_heap = []
            test_data = [(float(keys[i]), (i, np.random.rand(4).astype(np.float32), True)) for i in range(size)]
            
            # Insert operations
            heap_insert_time = self.time_operation(
                lambda: [heapq.heappush(py_heap, item) for item in test_data]
            )
            self.results.append(BenchmarkResult(
                "insert_single", "Python_heapq", size, heap_insert_time, size/heap_insert_time*1000
            ))
            
            # Delete min operations
            py_heap_full = test_data.copy()
            heapq.heapify(py_heap_full)
            
            delete_count = min(size, 10 * batch_size)  # Compare similar number of deletions
            heap_delete_time = self.time_operation(
                lambda: [heapq.heappop(py_heap_full) for _ in range(min(delete_count, len(py_heap_full)))]
            )
            self.results.append(BenchmarkResult(
                "delete_min_single", "Python_heapq", delete_count, heap_delete_time, delete_count/heap_delete_time*1000
            ))
    
    def benchmark_hashtable_operations(self):
        """Benchmark HashTable vs Python dict"""
        print("\n=== HashTable vs Python dict Benchmarks ===")
        
        for size in self.test_sizes:
            print(f"\nTesting size: {size}")
            
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
            
            # Single insert operations
            hash_table_single = HashTable.build(BenchmarkValue, seed=42, capacity=size * 2)
            single_insert_times = []
            
            current_table = hash_table_single
            for i in range(min(size, 1000)):  # Test up to 1000 single inserts
                item = BenchmarkValue(
                    id=jnp.array([test_values.id[i]]),
                    data=test_values.data[i:i+1],
                    flag=jnp.array([test_values.flag[i]])
                )
                single_time = self.time_operation(
                    lambda t=current_table, it=item: t.insert(it)
                )
                current_table, _, _ = current_table.insert(item)
                single_insert_times.append(single_time)
            
            if single_insert_times:
                avg_single_insert = statistics.mean(single_insert_times)
                self.results.append(BenchmarkResult(
                    "insert_single", "Xtructure_HashTable", min(size, 1000), avg_single_insert, 1/avg_single_insert*1000
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
        print("Starting comprehensive Xtructure benchmarks...")
        print(f"JAX backend: {jax.default_backend()}")
        print(f"JAX devices: {jax.devices()}")
        
        # Run all benchmark suites
        self.benchmark_stack_operations()
        self.benchmark_queue_operations()
        self.benchmark_priority_queue_operations()
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
        
        filename = f"xtructure_benchmarks_{int(time.time())}.csv"
        
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
    benchmark_suite = BenchmarkSuite()
    benchmark_suite.run_all_benchmarks()


if __name__ == "__main__":
    main()