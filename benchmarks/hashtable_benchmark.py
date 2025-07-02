"""
HashTable benchmarks: Xtructure HashTable vs Python dict
"""

from typing import List, Optional

import jax.numpy as jnp
from xtructure import HashTable

from .common.base_benchmark import BaseBenchmark
from .common.test_data import BenchmarkValue, create_test_data, create_python_dict_data


class HashTableBenchmark(BaseBenchmark):
    """Benchmark HashTable operations against Python dict"""
    
    def __init__(self, test_sizes: Optional[List[int]] = None, num_iterations: int = 3):
        super().__init__("HashTable vs Python dict", test_sizes, num_iterations)
    
    def run_xtructure_benchmarks(self) -> None:
        """Run Xtructure HashTable benchmarks"""
        
        total_tests = len(self.test_sizes) * 2  # 2 operations per size
        current_test = 0
        
        for size in self.test_sizes:
            current_test += 1
            self.print_progress("HashTable Insert (parallel)", size, current_test, total_tests)
            
            # Generate test data
            test_values = create_test_data(size)
            
            # Create HashTable with sufficient capacity
            hash_table = HashTable.build(BenchmarkValue, seed=42, capacity=size * 2)
            
            # Test insert operations (parallel)
            insert_time = self.time_operation(
                lambda: hash_table.parallel_insert(test_values)
            )
            self.add_result("insert_parallel", "Xtructure_HashTable", size, insert_time)
            
            current_test += 1
            self.print_progress("HashTable Lookup (parallel)", size, current_test, total_tests)
            
            # Test lookup operations (parallel)
            hash_table_full, _, _, _ = hash_table.parallel_insert(test_values)
            
            # Sample some values for lookup (to avoid memory issues with large datasets)
            lookup_indices = jnp.arange(0, size, max(1, size // 100))  # Sample ~100 lookups
            lookup_values = BenchmarkValue(
                id=test_values.id[lookup_indices],
                data=test_values.data[lookup_indices],
                flag=test_values.flag[lookup_indices]
            )
            
            lookup_time = self.time_operation(
                lambda: hash_table_full.lookup_parallel(lookup_values)
            )
            self.add_result("lookup_parallel", "Xtructure_HashTable", 
                          len(lookup_indices), lookup_time)
    
    def run_python_benchmarks(self) -> None:
        """Run Python dict benchmarks"""
        
        total_tests = len(self.test_sizes) * 2  # 2 operations per size
        current_test = 0
        
        for size in self.test_sizes:
            current_test += 1
            self.print_progress("Python dict Insert (batch)", size, current_test, total_tests)
            
            # Generate test data
            test_data = create_python_dict_data(size)
            
            # Test insert operations
            py_dict = {}
            insert_time = self.time_operation(
                lambda: py_dict.update(test_data)
            )
            self.add_result("insert_batch", "Python_dict", size, insert_time)
            
            current_test += 1
            self.print_progress("Python dict Lookup (batch)", size, current_test, total_tests)
            
            # Test lookup operations
            py_dict_full = test_data.copy()
            
            # Sample some keys for lookup (to match Xtructure test)
            lookup_keys = list(range(0, size, max(1, size // 100)))
            
            lookup_time = self.time_operation(
                lambda: [py_dict_full.get(key) for key in lookup_keys]
            )
            self.add_result("lookup_batch", "Python_dict", len(lookup_keys), lookup_time)


def run_hashtable_benchmark(test_sizes: Optional[List[int]] = None, num_iterations: int = 3) -> List:
    """
    Run hashtable benchmarks and return results.
    
    Args:
        test_sizes: List of test sizes to benchmark
        num_iterations: Number of iterations per test
        
    Returns:
        List of benchmark results
    """
    benchmark = HashTableBenchmark(test_sizes, num_iterations)
    return benchmark.run_all_benchmarks()


if __name__ == "__main__":
    # Run hashtable benchmarks independently
    results = run_hashtable_benchmark()
    
    benchmark = HashTableBenchmark()
    benchmark.results = results
    benchmark.print_results_summary()
    
    print("\nPerformance Comparisons:")
    comparisons = benchmark.get_performance_comparisons()
    for comparison in comparisons:
        print(f"  {comparison}")