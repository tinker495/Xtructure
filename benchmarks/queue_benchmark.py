"""
Queue benchmarks: Xtructure Queue vs Python deque
"""

from collections import deque
from typing import List, Optional

from xtructure import Queue

from .common.base_benchmark import BaseBenchmark
from .common.test_data import BenchmarkValue, create_test_data, create_python_test_data


class QueueBenchmark(BaseBenchmark):
    """Benchmark Queue operations against Python deque"""
    
    def __init__(self, test_sizes: Optional[List[int]] = None, num_iterations: int = 3):
        super().__init__("Queue vs Python deque", test_sizes, num_iterations)
    
    def run_xtructure_benchmarks(self) -> None:
        """Run Xtructure Queue benchmarks"""
        
        total_tests = len(self.test_sizes) * 2  # 2 operations per size
        current_test = 0
        
        for size in self.test_sizes:
            current_test += 1
            self.print_progress("Queue Enqueue (batch)", size, current_test, total_tests)
            
            # Generate test data
            test_values = create_test_data(size)
            
            # Create Queue
            queue = Queue.build(size, BenchmarkValue)
            
            # Test enqueue operations (batch)
            enqueue_time = self.time_operation(lambda: queue.enqueue(test_values))
            self.add_result("enqueue_batch", "Xtructure_Queue", size, enqueue_time)
            
            current_test += 1
            self.print_progress("Queue Dequeue (batch)", size, current_test, total_tests)
            
            # Test dequeue operations (batch)
            queue_full = queue.enqueue(test_values)
            dequeue_time = self.time_operation(lambda: queue_full.dequeue(size))
            self.add_result("dequeue_batch", "Xtructure_Queue", size, dequeue_time)
    
    def run_python_benchmarks(self) -> None:
        """Run Python deque benchmarks"""
        
        total_tests = len(self.test_sizes) * 2  # 2 operations per size
        current_test = 0
        
        for size in self.test_sizes:
            current_test += 1
            self.print_progress("Python deque Enqueue (single)", size, current_test, total_tests)
            
            # Generate test data
            test_data = create_python_test_data(size)
            
            # Test enqueue (append) operations
            py_deque = deque()
            enqueue_time = self.time_operation(
                lambda: [py_deque.append(item) for item in test_data]
            )
            self.add_result("enqueue_single", "Python_deque", size, enqueue_time)
            
            current_test += 1
            self.print_progress("Python deque Dequeue (single)", size, current_test, total_tests)
            
            # Test dequeue (popleft) operations
            py_deque_full = deque(test_data)
            dequeue_time = self.time_operation(
                lambda: [py_deque_full.popleft() for _ in range(len(py_deque_full))]
            )
            self.add_result("dequeue_single", "Python_deque", size, dequeue_time)


def run_queue_benchmark(test_sizes: Optional[List[int]] = None, num_iterations: int = 3) -> List:
    """
    Run queue benchmarks and return results.
    
    Args:
        test_sizes: List of test sizes to benchmark
        num_iterations: Number of iterations per test
        
    Returns:
        List of benchmark results
    """
    benchmark = QueueBenchmark(test_sizes, num_iterations)
    return benchmark.run_all_benchmarks()


if __name__ == "__main__":
    # Run queue benchmarks independently
    results = run_queue_benchmark()
    
    benchmark = QueueBenchmark()
    benchmark.results = results
    benchmark.print_results_summary()
    
    print("\nPerformance Comparisons:")
    comparisons = benchmark.get_performance_comparisons()
    for comparison in comparisons:
        print(f"  {comparison}")