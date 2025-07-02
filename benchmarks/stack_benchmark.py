"""
Stack benchmarks: Xtructure Stack vs Python list
"""

from typing import List, Optional

from xtructure import Stack

from .common.base_benchmark import BaseBenchmark
from .common.test_data import BenchmarkValue, create_test_data, create_python_test_data


class StackBenchmark(BaseBenchmark):
    """Benchmark Stack operations against Python list"""
    
    def __init__(self, test_sizes: Optional[List[int]] = None, num_iterations: int = 3):
        super().__init__("Stack vs Python List", test_sizes, num_iterations)
    
    def run_xtructure_benchmarks(self) -> None:
        """Run Xtructure Stack benchmarks"""
        
        total_tests = len(self.test_sizes) * 2  # 2 operations per size
        current_test = 0
        
        for size in self.test_sizes:
            current_test += 1
            self.print_progress("Stack Push (batch)", size, current_test, total_tests)
            
            # Generate test data
            test_values = create_test_data(size)
            
            # Create Stack
            stack = Stack.build(size, BenchmarkValue)
            
            # Test push operations (batch)
            push_time = self.time_operation(lambda: stack.push(test_values))
            self.add_result("push_batch", "Xtructure_Stack", size, push_time)
            
            current_test += 1
            self.print_progress("Stack Pop (batch)", size, current_test, total_tests)
            
            # Test pop operations (batch)
            stack_full = stack.push(test_values)
            pop_time = self.time_operation(lambda: stack_full.pop(size))
            self.add_result("pop_batch", "Xtructure_Stack", size, pop_time)
    
    def run_python_benchmarks(self) -> None:
        """Run Python list benchmarks"""
        
        total_tests = len(self.test_sizes) * 2  # 2 operations per size  
        current_test = 0
        
        for size in self.test_sizes:
            current_test += 1
            self.print_progress("Python List Push (single)", size, current_test, total_tests)
            
            # Generate test data
            test_data = create_python_test_data(size)
            
            # Test push (append) operations
            py_list = []
            push_time = self.time_operation(
                lambda: [py_list.append(item) for item in test_data]
            )
            self.add_result("push_single", "Python_list", size, push_time)
            
            current_test += 1
            self.print_progress("Python List Pop (single)", size, current_test, total_tests)
            
            # Test pop operations
            py_list_full = test_data.copy()
            pop_time = self.time_operation(
                lambda: [py_list_full.pop() for _ in range(len(py_list_full))]
            )
            self.add_result("pop_single", "Python_list", size, pop_time)


def run_stack_benchmark(test_sizes: Optional[List[int]] = None, num_iterations: int = 3) -> List:
    """
    Run stack benchmarks and return results.
    
    Args:
        test_sizes: List of test sizes to benchmark
        num_iterations: Number of iterations per test
        
    Returns:
        List of benchmark results
    """
    benchmark = StackBenchmark(test_sizes, num_iterations)
    return benchmark.run_all_benchmarks()


if __name__ == "__main__":
    # Run stack benchmarks independently
    results = run_stack_benchmark()
    
    benchmark = StackBenchmark()
    benchmark.results = results
    benchmark.print_results_summary()
    
    print("\nPerformance Comparisons:")
    comparisons = benchmark.get_performance_comparisons()
    for comparison in comparisons:
        print(f"  {comparison}")