"""
BGPQ benchmarks: Xtructure BGPQ vs Python heapq

Note: BGPQ requires GPU backend for optimal performance.
CPU execution may fail due to Pallas limitations.
"""

import heapq
import statistics
from typing import List, Optional

import jax
import jax.numpy as jnp

from .common.base_benchmark import BaseBenchmark
from .common.test_data import (BenchmarkValue, create_priority_queue_data, 
                               create_python_heap_data)

try:
    from xtructure import BGPQ
    BGPQ_AVAILABLE = True
except ImportError:
    BGPQ_AVAILABLE = False
    print("Warning: BGPQ not available")


class BGPQBenchmark(BaseBenchmark):
    """Benchmark BGPQ operations against Python heapq"""
    
    def __init__(self, test_sizes: Optional[List[int]] = None, num_iterations: int = 3,
                 batch_size: int = 64):
        super().__init__("BGPQ vs Python heapq", test_sizes, num_iterations)
        self.batch_size = batch_size
        
        if not BGPQ_AVAILABLE:
            print("Warning: BGPQ not available, skipping BGPQ benchmarks")
    
    def run_xtructure_benchmarks(self) -> None:
        """Run Xtructure BGPQ benchmarks"""
        
        if not BGPQ_AVAILABLE:
            print("Skipping BGPQ benchmarks - not available")
            return
        
        # Check if we're on GPU backend
        if jax.default_backend() == "cpu":
            print("Warning: BGPQ may not work properly on CPU backend")
            print("Consider using GPU backend for BGPQ benchmarks")
        
        total_tests = len(self.test_sizes) * 2  # 2 operations per size
        current_test = 0
        
        for size in self.test_sizes:
            current_test += 1
            self.print_progress("BGPQ Insert (batch)", size, current_test, total_tests)
            
            try:
                # Generate test data
                keys, test_values = create_priority_queue_data(size)
                
                # Create BGPQ with sufficient capacity
                bgpq = BGPQ.build(size + self.batch_size, self.batch_size, BenchmarkValue)
                
                # Test insert operations (batched)
                num_batches = (size + self.batch_size - 1) // self.batch_size
                insert_times = []
                
                current_bgpq = bgpq
                for i in range(num_batches):
                    start_idx = i * self.batch_size
                    end_idx = min((i + 1) * self.batch_size, size)
                    
                    batch_keys = keys[start_idx:end_idx]
                    batch_values = BenchmarkValue(
                        id=test_values.id[start_idx:end_idx],
                        data=test_values.data[start_idx:end_idx],
                        flag=test_values.flag[start_idx:end_idx]
                    )
                    
                    # Pad to batch size if needed
                    if len(batch_keys) < self.batch_size:
                        pad_size = self.batch_size - len(batch_keys)
                        batch_keys = jnp.concatenate([
                            batch_keys,
                            jnp.full(pad_size, jnp.inf, dtype=jnp.float32)
                        ])
                        batch_values = batch_values.padding_as_batch((self.batch_size,))
                    
                    insert_time = self.time_operation(
                        lambda bq=current_bgpq, k=batch_keys, v=batch_values: BGPQ.insert(bq, k, v)
                    )
                    current_bgpq = BGPQ.insert(current_bgpq, batch_keys, batch_values)
                    insert_times.append(insert_time)
                
                avg_insert_time = statistics.mean(insert_times)
                self.add_result("insert_batch", "Xtructure_BGPQ", size, avg_insert_time)
                
                current_test += 1
                self.print_progress("BGPQ Delete Min (batch)", size, current_test, total_tests)
                
                # Test delete min operations
                delete_times = []
                for _ in range(min(10, num_batches)):  # Test up to 10 deletions
                    delete_time = self.time_operation(
                        lambda bq=current_bgpq: BGPQ.delete_mins(bq)
                    )
                    current_bgpq, _, _ = BGPQ.delete_mins(current_bgpq)
                    delete_times.append(delete_time)
                
                if delete_times:
                    avg_delete_time = statistics.mean(delete_times)
                    self.add_result("delete_min_batch", "Xtructure_BGPQ", 
                                  self.batch_size, avg_delete_time)
                
            except Exception as e:
                print(f"Error in BGPQ benchmark for size {size}: {e}")
                if "Only interpret mode is supported on CPU backend" in str(e):
                    print("BGPQ requires GPU backend. Skipping remaining BGPQ tests.")
                    break
                # Continue with other sizes
                continue
    
    def run_python_benchmarks(self) -> None:
        """Run Python heapq benchmarks"""
        
        total_tests = len(self.test_sizes) * 2  # 2 operations per size
        current_test = 0
        
        for size in self.test_sizes:
            current_test += 1
            self.print_progress("Python heapq Insert (single)", size, current_test, total_tests)
            
            # Generate test data
            test_data = create_python_heap_data(size)
            
            # Test insert operations
            py_heap = []
            insert_time = self.time_operation(
                lambda: [heapq.heappush(py_heap, item) for item in test_data]
            )
            self.add_result("insert_single", "Python_heapq", size, insert_time)
            
            current_test += 1
            self.print_progress("Python heapq Delete Min (single)", size, current_test, total_tests)
            
            # Test delete min operations
            py_heap_full = test_data.copy()
            heapq.heapify(py_heap_full)
            
            # Compare similar number of deletions to BGPQ
            delete_count = min(size, 10 * self.batch_size)
            delete_time = self.time_operation(
                lambda: [heapq.heappop(py_heap_full) 
                        for _ in range(min(delete_count, len(py_heap_full)))]
            )
            self.add_result("delete_min_single", "Python_heapq", delete_count, delete_time)


def run_bgpq_benchmark(test_sizes: Optional[List[int]] = None, num_iterations: int = 3,
                      batch_size: int = 64) -> List:
    """
    Run BGPQ benchmarks and return results.
    
    Args:
        test_sizes: List of test sizes to benchmark
        num_iterations: Number of iterations per test
        batch_size: BGPQ batch size
        
    Returns:
        List of benchmark results
    """
    benchmark = BGPQBenchmark(test_sizes, num_iterations, batch_size)
    return benchmark.run_all_benchmarks()


def check_bgpq_compatibility() -> bool:
    """
    Check if BGPQ can run on current backend.
    
    Returns:
        True if BGPQ is compatible, False otherwise
    """
    if not BGPQ_AVAILABLE:
        return False
    
    backend = jax.default_backend()
    if backend == "cpu":
        print("Warning: BGPQ may not work on CPU backend due to Pallas limitations")
        return False
    
    return True


if __name__ == "__main__":
    # Check compatibility first
    if check_bgpq_compatibility():
        print("BGPQ appears compatible with current backend")
    else:
        print("BGPQ may not work with current backend")
    
    # Run BGPQ benchmarks independently
    results = run_bgpq_benchmark()
    
    benchmark = BGPQBenchmark()
    benchmark.results = results
    benchmark.print_results_summary()
    
    print("\nPerformance Comparisons:")
    comparisons = benchmark.get_performance_comparisons()
    for comparison in comparisons:
        print(f"  {comparison}")