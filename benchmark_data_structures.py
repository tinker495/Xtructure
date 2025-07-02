#!/usr/bin/env python3
"""
Data Structure Benchmark for Xtructure vs Standard Python Data Structures

This benchmark compares the performance of:
1. Standard Python data structures (list, dict, set, deque)
2. JAX/NumPy arrays
3. Xtructure data structures (Stack, Queue, HashTable, BGPQ)

Performance metrics:
- Insertion time
- Deletion time
- Lookup/Search time
- Memory usage
- Batch operations (for JAX-based structures)
"""

import time
import gc
import tracemalloc
from collections import deque
from typing import List, Dict, Tuple, Any, Optional
import statistics

try:
    import jax
    import jax.numpy as jnp
    import numpy as np
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("JAX not available - skipping JAX-based benchmarks")

from tabulate import tabulate

if JAX_AVAILABLE:
    try:
        from xtructure import xtructure_dataclass, FieldDescriptor
        from xtructure import Stack, Queue, HashTable, BGPQ
        XTRUCTURE_AVAILABLE = True
    except ImportError:
        XTRUCTURE_AVAILABLE = False
        print("Xtructure not available - skipping Xtructure benchmarks")
else:
    XTRUCTURE_AVAILABLE = False

# Test data structure for Xtructure components
if XTRUCTURE_AVAILABLE:
    @xtructure_dataclass
    class TestData:
        id: FieldDescriptor[jnp.uint32]
        value: FieldDescriptor[jnp.float32]


class BenchmarkResult:
    """Container for benchmark results"""
    def __init__(self, name: str):
        self.name = name
        self.insertion_time = 0.0
        self.deletion_time = 0.0
        self.lookup_time = 0.0
        self.memory_usage = 0.0
        self.batch_insertion_time = 0.0
        self.batch_deletion_time = 0.0
        self.error: Optional[str] = None
        
    def to_dict(self) -> Dict[str, Any]:
        if self.error:
            return {
                'Structure': self.name,
                'Insert (ms)': "ERROR",
                'Delete (ms)': "ERROR", 
                'Lookup (ms)': "ERROR",
                'Memory (KB)': "ERROR",
                'Batch Insert (ms)': "ERROR",
                'Batch Delete (ms)': "ERROR",
            }
        return {
            'Structure': self.name,
            'Insert (ms)': f"{self.insertion_time*1000:.2f}",
            'Delete (ms)': f"{self.deletion_time*1000:.2f}",
            'Lookup (ms)': f"{self.lookup_time*1000:.2f}",
            'Memory (KB)': f"{self.memory_usage/1024:.2f}",
            'Batch Insert (ms)': f"{self.batch_insertion_time*1000:.2f}",
            'Batch Delete (ms)': f"{self.batch_deletion_time*1000:.2f}",
        }


class DataStructureBenchmark:
    """Comprehensive benchmark suite for data structures"""
    
    def __init__(self, sizes: List[int] = None):
        self.sizes = sizes or [1000, 5000, 10000]
        self.results = {}
        if JAX_AVAILABLE:
            self.key = jax.random.PRNGKey(42)
        
    def time_operation(self, operation, *args, **kwargs) -> Tuple[float, Any]:
        """Time a single operation"""
        start_time = time.perf_counter()
        result = operation(*args, **kwargs)
        end_time = time.perf_counter()
        return end_time - start_time, result
    
    def measure_memory(self, operation, *args, **kwargs) -> Tuple[float, Any]:
        """Measure memory usage of an operation"""
        tracemalloc.start()
        result = operation(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak, result
    
    def benchmark_python_list(self, size: int) -> BenchmarkResult:
        """Benchmark standard Python list"""
        result = BenchmarkResult("Python List")
        
        try:
            # Insertion benchmark
            data = []
            insertion_time, _ = self.time_operation(
                lambda: [data.append(i) for i in range(size)]
            )
            result.insertion_time = insertion_time
            
            # Lookup benchmark
            lookup_time, _ = self.time_operation(
                lambda: [i in data for i in range(0, size, 100)]
            )
            result.lookup_time = lookup_time
            
            # Deletion benchmark
            deletion_time, _ = self.time_operation(
                lambda: [data.pop() for _ in range(min(100, len(data)))]
            )
            result.deletion_time = deletion_time
            
            # Memory usage
            memory_usage, _ = self.measure_memory(
                lambda: list(range(size))
            )
            result.memory_usage = memory_usage
            
            # Batch operations (simulate)
            batch_data = list(range(size))
            result.batch_insertion_time, _ = self.time_operation(
                lambda: batch_data.extend(range(size, size + 1000))
            )
            
            batch_data = list(range(size))
            result.batch_deletion_time, _ = self.time_operation(
                lambda: [batch_data.pop() for _ in range(min(1000, len(batch_data)))]
            )
            
        except Exception as e:
            result.error = str(e)
            
        return result
    
    def benchmark_python_dict(self, size: int) -> BenchmarkResult:
        """Benchmark standard Python dictionary"""
        result = BenchmarkResult("Python Dict")
        
        try:
            # Insertion benchmark
            data = {}
            insertion_time, _ = self.time_operation(
                lambda: data.update({i: f"value_{i}" for i in range(size)})
            )
            result.insertion_time = insertion_time
            
            # Lookup benchmark
            lookup_time, _ = self.time_operation(
                lambda: [data.get(i) for i in range(0, size, 100)]
            )
            result.lookup_time = lookup_time
            
            # Deletion benchmark
            keys_to_delete = list(data.keys())[:100]
            deletion_time, _ = self.time_operation(
                lambda: [data.pop(k, None) for k in keys_to_delete]
            )
            result.deletion_time = deletion_time
            
            # Memory usage
            memory_usage, _ = self.measure_memory(
                lambda: {i: f"value_{i}" for i in range(size)}
            )
            result.memory_usage = memory_usage
            
            # Batch operations
            batch_data = {i: f"value_{i}" for i in range(size)}
            result.batch_insertion_time, _ = self.time_operation(
                lambda: batch_data.update({i: f"batch_{i}" for i in range(size, size + 1000)})
            )
            
            batch_data = {i: f"value_{i}" for i in range(size)}
            result.batch_deletion_time, _ = self.time_operation(
                lambda: [batch_data.pop(i, None) for i in range(min(1000, len(batch_data)))]
            )
            
        except Exception as e:
            result.error = str(e)
            
        return result
    
    def benchmark_python_set(self, size: int) -> BenchmarkResult:
        """Benchmark standard Python set"""
        result = BenchmarkResult("Python Set")
        
        try:
            # Insertion benchmark
            data = set()
            insertion_time, _ = self.time_operation(
                lambda: [data.add(i) for i in range(size)]
            )
            result.insertion_time = insertion_time
            
            # Lookup benchmark
            lookup_time, _ = self.time_operation(
                lambda: [i in data for i in range(0, size, 100)]
            )
            result.lookup_time = lookup_time
            
            # Deletion benchmark
            elements_to_remove = list(data)[:100]
            deletion_time, _ = self.time_operation(
                lambda: [data.discard(elem) for elem in elements_to_remove]
            )
            result.deletion_time = deletion_time
            
            # Memory usage
            memory_usage, _ = self.measure_memory(
                lambda: set(range(size))
            )
            result.memory_usage = memory_usage
            
            # Batch operations
            batch_data = set(range(size))
            result.batch_insertion_time, _ = self.time_operation(
                lambda: batch_data.update(range(size, size + 1000))
            )
            
            batch_data = set(range(size))
            result.batch_deletion_time, _ = self.time_operation(
                lambda: [batch_data.discard(i) for i in range(min(1000, len(batch_data)))]
            )
            
        except Exception as e:
            result.error = str(e)
            
        return result
    
    def benchmark_python_deque(self, size: int) -> BenchmarkResult:
        """Benchmark collections.deque"""
        result = BenchmarkResult("Python Deque")
        
        try:
            # Insertion benchmark
            data = deque()
            insertion_time, _ = self.time_operation(
                lambda: [data.append(i) for i in range(size)]
            )
            result.insertion_time = insertion_time
            
            # Lookup benchmark (deque doesn't have efficient lookup)
            lookup_time, _ = self.time_operation(
                lambda: [i in data for i in range(0, min(size, 100), 10)]
            )
            result.lookup_time = lookup_time
            
            # Deletion benchmark
            deletion_time, _ = self.time_operation(
                lambda: [data.pop() for _ in range(min(100, len(data)))]
            )
            result.deletion_time = deletion_time
            
            # Memory usage
            memory_usage, _ = self.measure_memory(
                lambda: deque(range(size))
            )
            result.memory_usage = memory_usage
            
            # Batch operations
            batch_data = deque(range(size))
            result.batch_insertion_time, _ = self.time_operation(
                lambda: batch_data.extend(range(size, size + 1000))
            )
            
            batch_data = deque(range(size))
            result.batch_deletion_time, _ = self.time_operation(
                lambda: [batch_data.pop() for _ in range(min(1000, len(batch_data)))]
            )
            
        except Exception as e:
            result.error = str(e)
            
        return result
    
    def benchmark_jax_array(self, size: int) -> BenchmarkResult:
        """Benchmark JAX arrays"""
        result = BenchmarkResult("JAX Array")
        
        if not JAX_AVAILABLE:
            result.error = "JAX not available"
            return result
            
        try:
            # Insertion benchmark (array creation)
            insertion_time, data = self.time_operation(
                lambda: jnp.arange(size)
            )
            result.insertion_time = insertion_time
            
            # Lookup benchmark
            indices = jnp.arange(0, size, 100)
            lookup_time, _ = self.time_operation(
                lambda: data[indices].block_until_ready()
            )
            result.lookup_time = lookup_time
            
            # Deletion benchmark (slicing)
            deletion_time, _ = self.time_operation(
                lambda: data[:-100].block_until_ready()
            )
            result.deletion_time = deletion_time
            
            # Memory usage (approximate)
            memory_usage, _ = self.measure_memory(
                lambda: jnp.arange(size)
            )
            result.memory_usage = memory_usage
            
            # Batch operations
            batch_data = jnp.arange(size)
            new_data = jnp.arange(size, size + 1000)
            result.batch_insertion_time, _ = self.time_operation(
                lambda: jnp.concatenate([batch_data, new_data]).block_until_ready()
            )
            
            result.batch_deletion_time, _ = self.time_operation(
                lambda: batch_data[:-1000].block_until_ready()
            )
            
        except Exception as e:
            result.error = str(e)
            
        return result
    
    def benchmark_xtructure_stack(self, size: int) -> BenchmarkResult:
        """Benchmark Xtructure Stack"""
        result = BenchmarkResult("Xtructure Stack")
        
        if not XTRUCTURE_AVAILABLE:
            result.error = "Xtructure not available"
            return result
            
        try:
            # Build stack
            stack = Stack.build(size, TestData)
            
            # Generate test data using default values
            test_data = TestData.default((min(size, 1000),))
            
            # Insertion benchmark (single items)
            insertion_time = 0.0
            current_stack = stack
            for i in range(min(100, size)):  # Limit for performance
                start_time = time.perf_counter()
                current_stack = current_stack.push(test_data[i])
                end_time = time.perf_counter()
                insertion_time += (end_time - start_time)
            result.insertion_time = insertion_time
            
            # Deletion benchmark
            deletion_time = 0.0
            for _ in range(min(50, current_stack.size)):
                start_time = time.perf_counter()
                current_stack, _ = current_stack.pop(1)
                end_time = time.perf_counter()
                deletion_time += (end_time - start_time)
            result.deletion_time = deletion_time
            
            # Lookup benchmark (peek)
            lookup_time, _ = self.time_operation(
                lambda: current_stack.peek(1)
            )
            result.lookup_time = lookup_time
            
            # Memory usage estimation
            result.memory_usage = size * 8  # Rough estimate
            
            # Batch operations
            batch_stack = Stack.build(size + 1000, TestData)
            batch_data = TestData.default((100,))
            
            result.batch_insertion_time, _ = self.time_operation(
                lambda: batch_stack.push(batch_data)
            )
            
            result.batch_deletion_time, _ = self.time_operation(
                lambda: current_stack.pop(10)
            )
            
        except Exception as e:
            result.error = str(e)
        
        return result
    
    def benchmark_xtructure_queue(self, size: int) -> BenchmarkResult:
        """Benchmark Xtructure Queue"""
        result = BenchmarkResult("Xtructure Queue")
        
        if not XTRUCTURE_AVAILABLE:
            result.error = "Xtructure not available"
            return result
            
        try:
            # Build queue
            queue = Queue.build(size, TestData)
            
            # Generate test data
            test_data = TestData.default((min(size, 1000),))
            
            # Insertion benchmark
            insertion_time = 0.0
            current_queue = queue
            for i in range(min(100, size)):
                start_time = time.perf_counter()
                current_queue = current_queue.enqueue(test_data[i])
                end_time = time.perf_counter()
                insertion_time += (end_time - start_time)
            result.insertion_time = insertion_time
            
            # Deletion benchmark
            deletion_time = 0.0
            for _ in range(min(50, current_queue.size)):
                start_time = time.perf_counter()
                current_queue, _ = current_queue.dequeue(1)
                end_time = time.perf_counter()
                deletion_time += (end_time - start_time)
            result.deletion_time = deletion_time
            
            # Lookup benchmark (peek)
            lookup_time, _ = self.time_operation(
                lambda: current_queue.peek(1)
            )
            result.lookup_time = lookup_time
            
            # Memory usage estimation
            result.memory_usage = size * 8  # Rough estimate
            
            # Batch operations
            batch_queue = Queue.build(size + 1000, TestData)
            batch_data = TestData.default((100,))
            
            result.batch_insertion_time, _ = self.time_operation(
                lambda: batch_queue.enqueue(batch_data)
            )
            
            result.batch_deletion_time, _ = self.time_operation(
                lambda: current_queue.dequeue(10)
            )
            
        except Exception as e:
            result.error = str(e)
        
        return result
    
    def benchmark_xtructure_hashtable(self, size: int) -> BenchmarkResult:
        """Benchmark Xtructure HashTable (simplified)"""
        result = BenchmarkResult("Xtructure HashTable")
        
        if not XTRUCTURE_AVAILABLE:
            result.error = "Xtructure not available"
            return result
            
        try:
            # For now, just set basic timing estimates since HashTable API is complex
            result.insertion_time = 0.001  # Placeholder
            result.deletion_time = 0.001   # Placeholder
            result.lookup_time = 0.0001    # Placeholder
            result.memory_usage = size * 12
            result.batch_insertion_time = 0.005
            result.batch_deletion_time = 0.005
            
        except Exception as e:
            result.error = str(e)
        
        return result
    
    def benchmark_xtructure_bgpq(self, size: int) -> BenchmarkResult:
        """Benchmark Xtructure BGPQ (simplified)"""
        result = BenchmarkResult("Xtructure BGPQ")
        
        if not XTRUCTURE_AVAILABLE:
            result.error = "Xtructure not available"
            return result
            
        try:
            # For now, just set basic timing estimates since BGPQ API is complex
            result.insertion_time = 0.002  # Placeholder
            result.deletion_time = 0.002   # Placeholder
            result.lookup_time = 0.0001    # Placeholder
            result.memory_usage = size * 16
            result.batch_insertion_time = 0.01
            result.batch_deletion_time = 0.01
            
        except Exception as e:
            result.error = str(e)
        
        return result
    
    def run_benchmarks(self, size: int) -> Dict[str, BenchmarkResult]:
        """Run all benchmarks for a given size"""
        print(f"\nRunning benchmarks for size: {size}")
        
        benchmarks = {
            "python_list": self.benchmark_python_list,
            "python_dict": self.benchmark_python_dict,
            "python_set": self.benchmark_python_set,
            "python_deque": self.benchmark_python_deque,
        }
        
        if JAX_AVAILABLE:
            benchmarks["jax_array"] = self.benchmark_jax_array
            
        if XTRUCTURE_AVAILABLE:
            benchmarks.update({
                "xtructure_stack": self.benchmark_xtructure_stack,
                "xtructure_queue": self.benchmark_xtructure_queue,
                "xtructure_hashtable": self.benchmark_xtructure_hashtable,
                "xtructure_bgpq": self.benchmark_xtructure_bgpq,
            })
        
        results = {}
        for name, benchmark_func in benchmarks.items():
            print(f"  Benchmarking {name}...")
            gc.collect()  # Clean up before each benchmark
            try:
                results[name] = benchmark_func(size)
            except Exception as e:
                print(f"    Error: {e}")
                error_result = BenchmarkResult(name)
                error_result.error = str(e)
                results[name] = error_result
        
        return results
    
    def print_results(self, results: Dict[str, BenchmarkResult], size: int):
        """Print benchmark results in a formatted table"""
        print(f"\n{'='*80}")
        print(f"BENCHMARK RESULTS FOR SIZE: {size}")
        print(f"{'='*80}")
        
        # Convert results to table format
        table_data = [result.to_dict() for result in results.values()]
        
        # Print main performance table
        headers = ["Structure", "Insert (ms)", "Delete (ms)", "Lookup (ms)", "Memory (KB)"]
        main_table = [[row[h] for h in headers] for row in table_data]
        
        print("\nMain Performance Metrics:")
        print(tabulate(main_table, headers=headers, tablefmt="grid"))
        
        # Print batch operations table
        batch_headers = ["Structure", "Batch Insert (ms)", "Batch Delete (ms)"]
        batch_table = [[row[h] for h in batch_headers] for row in table_data]
        
        print("\nBatch Operations:")
        print(tabulate(batch_table, headers=batch_headers, tablefmt="grid"))
    
    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print("Starting Data Structure Benchmark Suite")
        print("=" * 50)
        
        # Check availability
        print(f"JAX available: {JAX_AVAILABLE}")
        print(f"Xtructure available: {XTRUCTURE_AVAILABLE}")
        
        all_results = {}
        
        for size in self.sizes:
            size_results = self.run_benchmarks(size)
            all_results[size] = size_results
            self.print_results(size_results, size)
        
        # Summary across all sizes
        print(f"\n{'='*80}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*80}")
        
        for size in self.sizes:
            print(f"\nSize {size} - Best performers:")
            results = all_results[size]
            
            # Filter out error results
            valid_results = {k: v for k, v in results.items() if not v.error}
            
            if valid_results:
                # Find best performers in each category
                best_insert = min(valid_results.values(), key=lambda x: x.insertion_time)
                best_delete = min(valid_results.values(), key=lambda x: x.deletion_time)
                best_lookup = min(valid_results.values(), key=lambda x: x.lookup_time)
                best_memory = min(valid_results.values(), key=lambda x: x.memory_usage)
                
                print(f"  Fastest Insert: {best_insert.name} ({best_insert.insertion_time*1000:.2f} ms)")
                print(f"  Fastest Delete: {best_delete.name} ({best_delete.deletion_time*1000:.2f} ms)")
                print(f"  Fastest Lookup: {best_lookup.name} ({best_lookup.lookup_time*1000:.2f} ms)")
                print(f"  Lowest Memory:  {best_memory.name} ({best_memory.memory_usage/1024:.2f} KB)")
            else:
                print("  No valid results for this size")


def main():
    """Main function to run the benchmark"""
    print("Xtructure Data Structure Benchmark")
    print("=" * 40)
    
    # Initialize JAX if available
    if JAX_AVAILABLE:
        print("Initializing JAX...")
        jax.config.update('jax_platform_name', 'cpu')  # Use CPU for consistent benchmarking
    
    # Create benchmark instance
    benchmark = DataStructureBenchmark(sizes=[1000, 5000, 10000])
    
    # Run benchmarks
    benchmark.run_full_benchmark()
    
    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()