#!/usr/bin/env python3
"""
Simple Data Structure Benchmark (No External Dependencies)

This benchmark compares the performance of standard Python data structures:
- list
- dict  
- set
- deque
- tuple
"""

import time
import gc
import tracemalloc
from collections import deque
from typing import List, Dict, Tuple, Any, Optional


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


class SimpleDataStructureBenchmark:
    """Simple benchmark suite for Python data structures"""
    
    def __init__(self, sizes: Optional[List[int]] = None):
        self.sizes = sizes or [1000, 5000, 10000]
        
    def time_operation(self, operation) -> Tuple[float, Any]:
        """Time a single operation"""
        start_time = time.perf_counter()
        result = operation()
        end_time = time.perf_counter()
        return end_time - start_time, result
    
    def measure_memory(self, operation) -> Tuple[float, Any]:
        """Measure memory usage of an operation"""
        tracemalloc.start()
        result = operation()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak, result
    
    def benchmark_list(self, size: int) -> BenchmarkResult:
        """Benchmark Python list"""
        result = BenchmarkResult("Python List")
        
        try:
            # Insertion benchmark
            def insert_test():
                data = []
                for i in range(size):
                    data.append(i)
                return data
            
            result.insertion_time, data = self.time_operation(insert_test)
            
            # Lookup benchmark
            def lookup_test():
                return [i in data for i in range(0, size, 100)]
            
            result.lookup_time, _ = self.time_operation(lookup_test)
            
            # Deletion benchmark
            def delete_test():
                for _ in range(min(100, len(data))):
                    data.pop()
                return data
            
            result.deletion_time, _ = self.time_operation(delete_test)
            
            # Memory usage
            result.memory_usage, _ = self.measure_memory(
                lambda: list(range(size))
            )
            
            # Batch operations
            def batch_insert_test():
                batch_data = list(range(size))
                batch_data.extend(range(size, size + 1000))
                return batch_data
            
            result.batch_insertion_time, _ = self.time_operation(batch_insert_test)
            
            def batch_delete_test():
                batch_data = list(range(size))
                for _ in range(min(1000, len(batch_data))):
                    batch_data.pop()
                return batch_data
            
            result.batch_deletion_time, _ = self.time_operation(batch_delete_test)
            
        except Exception as e:
            result.error = str(e)
            
        return result
    
    def benchmark_dict(self, size: int) -> BenchmarkResult:
        """Benchmark Python dictionary"""
        result = BenchmarkResult("Python Dict")
        
        try:
            # Insertion benchmark
            def insert_test():
                data = {}
                for i in range(size):
                    data[i] = f"value_{i}"
                return data
            
            result.insertion_time, data = self.time_operation(insert_test)
            
            # Lookup benchmark
            def lookup_test():
                return [data.get(i) for i in range(0, size, 100)]
            
            result.lookup_time, _ = self.time_operation(lookup_test)
            
            # Deletion benchmark
            def delete_test():
                keys_to_delete = list(data.keys())[:100]
                for k in keys_to_delete:
                    data.pop(k, None)
                return data
            
            result.deletion_time, _ = self.time_operation(delete_test)
            
            # Memory usage
            result.memory_usage, _ = self.measure_memory(
                lambda: {i: f"value_{i}" for i in range(size)}
            )
            
            # Batch operations
            def batch_insert_test():
                batch_data = {i: f"value_{i}" for i in range(size)}
                batch_data.update({i: f"batch_{i}" for i in range(size, size + 1000)})
                return batch_data
            
            result.batch_insertion_time, _ = self.time_operation(batch_insert_test)
            
            def batch_delete_test():
                batch_data = {i: f"value_{i}" for i in range(size)}
                for i in range(min(1000, len(batch_data))):
                    batch_data.pop(i, None)
                return batch_data
            
            result.batch_deletion_time, _ = self.time_operation(batch_delete_test)
            
        except Exception as e:
            result.error = str(e)
            
        return result
    
    def benchmark_set(self, size: int) -> BenchmarkResult:
        """Benchmark Python set"""
        result = BenchmarkResult("Python Set")
        
        try:
            # Insertion benchmark
            def insert_test():
                data = set()
                for i in range(size):
                    data.add(i)
                return data
            
            result.insertion_time, data = self.time_operation(insert_test)
            
            # Lookup benchmark
            def lookup_test():
                return [i in data for i in range(0, size, 100)]
            
            result.lookup_time, _ = self.time_operation(lookup_test)
            
            # Deletion benchmark
            def delete_test():
                elements_to_remove = list(data)[:100]
                for elem in elements_to_remove:
                    data.discard(elem)
                return data
            
            result.deletion_time, _ = self.time_operation(delete_test)
            
            # Memory usage
            result.memory_usage, _ = self.measure_memory(
                lambda: set(range(size))
            )
            
            # Batch operations
            def batch_insert_test():
                batch_data = set(range(size))
                batch_data.update(range(size, size + 1000))
                return batch_data
            
            result.batch_insertion_time, _ = self.time_operation(batch_insert_test)
            
            def batch_delete_test():
                batch_data = set(range(size))
                for i in range(min(1000, len(batch_data))):
                    batch_data.discard(i)
                return batch_data
            
            result.batch_deletion_time, _ = self.time_operation(batch_delete_test)
            
        except Exception as e:
            result.error = str(e)
            
        return result
    
    def benchmark_deque(self, size: int) -> BenchmarkResult:
        """Benchmark collections.deque"""
        result = BenchmarkResult("Python Deque")
        
        try:
            # Insertion benchmark
            def insert_test():
                data = deque()
                for i in range(size):
                    data.append(i)
                return data
            
            result.insertion_time, data = self.time_operation(insert_test)
            
            # Lookup benchmark (deque doesn't have efficient lookup)
            def lookup_test():
                return [i in data for i in range(0, min(size, 100), 10)]
            
            result.lookup_time, _ = self.time_operation(lookup_test)
            
            # Deletion benchmark
            def delete_test():
                for _ in range(min(100, len(data))):
                    data.pop()
                return data
            
            result.deletion_time, _ = self.time_operation(delete_test)
            
            # Memory usage
            result.memory_usage, _ = self.measure_memory(
                lambda: deque(range(size))
            )
            
            # Batch operations
            def batch_insert_test():
                batch_data = deque(range(size))
                batch_data.extend(range(size, size + 1000))
                return batch_data
            
            result.batch_insertion_time, _ = self.time_operation(batch_insert_test)
            
            def batch_delete_test():
                batch_data = deque(range(size))
                for _ in range(min(1000, len(batch_data))):
                    batch_data.pop()
                return batch_data
            
            result.batch_deletion_time, _ = self.time_operation(batch_delete_test)
            
        except Exception as e:
            result.error = str(e)
            
        return result
    
    def benchmark_tuple(self, size: int) -> BenchmarkResult:
        """Benchmark Python tuple (immutable)"""
        result = BenchmarkResult("Python Tuple")
        
        try:
            # Insertion benchmark (creation)
            def insert_test():
                return tuple(range(size))
            
            result.insertion_time, data = self.time_operation(insert_test)
            
            # Lookup benchmark
            def lookup_test():
                return [i in data for i in range(0, size, 100)]
            
            result.lookup_time, _ = self.time_operation(lookup_test)
            
            # Deletion benchmark (recreation without elements)
            def delete_test():
                return data[:-100]
            
            result.deletion_time, _ = self.time_operation(delete_test)
            
            # Memory usage
            result.memory_usage, _ = self.measure_memory(
                lambda: tuple(range(size))
            )
            
            # Batch operations
            def batch_insert_test():
                return data + tuple(range(size, size + 1000))
            
            result.batch_insertion_time, _ = self.time_operation(batch_insert_test)
            
            def batch_delete_test():
                return data[:-1000]
            
            result.batch_deletion_time, _ = self.time_operation(batch_delete_test)
            
        except Exception as e:
            result.error = str(e)
            
        return result
    
    def run_benchmarks(self, size: int) -> Dict[str, BenchmarkResult]:
        """Run all benchmarks for a given size"""
        print(f"\nRunning benchmarks for size: {size}")
        
        benchmarks = {
            "list": self.benchmark_list,
            "dict": self.benchmark_dict,
            "set": self.benchmark_set,
            "deque": self.benchmark_deque,
            "tuple": self.benchmark_tuple,
        }
        
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
        """Print benchmark results in a simple table format"""
        print(f"\n{'='*80}")
        print(f"BENCHMARK RESULTS FOR SIZE: {size}")
        print(f"{'='*80}")
        
        # Simple table formatting
        headers = ["Structure", "Insert(ms)", "Delete(ms)", "Lookup(ms)", "Memory(KB)", "BatchIns(ms)", "BatchDel(ms)"]
        col_widths = [12, 10, 10, 10, 11, 11, 11]
        
        # Print headers
        header_line = ""
        for i, header in enumerate(headers):
            header_line += f"{header:<{col_widths[i]}} "
        print(header_line)
        print("-" * len(header_line))
        
        # Print data
        for result in results.values():
            if result.error:
                row = [result.name, "ERROR", "ERROR", "ERROR", "ERROR", "ERROR", "ERROR"]
            else:
                row = [
                    result.name,
                    f"{result.insertion_time*1000:.2f}",
                    f"{result.deletion_time*1000:.2f}",
                    f"{result.lookup_time*1000:.2f}",
                    f"{result.memory_usage/1024:.2f}",
                    f"{result.batch_insertion_time*1000:.2f}",
                    f"{result.batch_deletion_time*1000:.2f}",
                ]
            
            row_line = ""
            for i, cell in enumerate(row):
                row_line += f"{cell:<{col_widths[i]}} "
            print(row_line)
    
    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print("Simple Data Structure Benchmark Suite")
        print("=" * 50)
        
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
    print("Python Data Structure Performance Benchmark")
    print("=" * 45)
    
    # Create benchmark instance
    benchmark = SimpleDataStructureBenchmark(sizes=[1000, 5000, 10000])
    
    # Run benchmarks
    benchmark.run_full_benchmark()
    
    print("\nBenchmark completed!")
    print("\nInterpretation:")
    print("- Insert: Time to add elements one by one")
    print("- Delete: Time to remove elements one by one")
    print("- Lookup: Time to search for elements")
    print("- Memory: Peak memory usage during creation")
    print("- BatchIns: Time for batch insertion operations")
    print("- BatchDel: Time for batch deletion operations")


if __name__ == "__main__":
    main()