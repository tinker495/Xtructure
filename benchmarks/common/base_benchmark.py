"""
Base benchmark class and utilities for Xtructure benchmarks.
"""

import time
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Callable, Any, Optional

import jax


@dataclass
class BenchmarkResult:
    """Results from a single benchmark test"""
    operation: str
    data_structure: str
    size: int
    time_ms: float
    throughput_ops_per_sec: float
    memory_mb: Optional[float] = None
    notes: Optional[str] = None


class BaseBenchmark(ABC):
    """
    Base class for data structure benchmarks.
    
    Provides common timing and result management functionality.
    """
    
    def __init__(self, name: str, test_sizes: Optional[List[int]] = None, num_iterations: int = 3):
        """
        Initialize benchmark.
        
        Args:
            name: Name of the benchmark
            test_sizes: List of sizes to test
            num_iterations: Number of iterations per test
        """
        self.name = name
        self.test_sizes = test_sizes or [100, 1000, 5000, 10000]
        self.num_iterations = num_iterations
        self.results: List[BenchmarkResult] = []
    
    def time_operation(self, func: Callable, *args, **kwargs) -> float:
        """
        Time a function call and return elapsed time in milliseconds.
        Uses jax.block_until_ready() for accurate JAX operation timing.
        
        Args:
            func: Function to time
            *args: Arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Average execution time in milliseconds
        """
        times = []
        
        # Warmup runs with proper synchronization
        for _ in range(min(2, self.num_iterations)):
            try:
                result = func(*args, **kwargs)
                # Use jax.block_until_ready for proper synchronization during warmup
                # This ensures JIT compilation completes before actual timing
                result = jax.block_until_ready(result)
            except Exception:
                pass  # Ignore warmup errors
        
        # Actual timing runs
        for _ in range(self.num_iterations):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            
            # Use jax.block_until_ready for proper synchronization
            # This handles both single arrays and pytrees correctly
            result = jax.block_until_ready(result)
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        return statistics.mean(times)
    
    def add_result(self, operation: str, data_structure: str, size: int, 
                  time_ms: float, notes: Optional[str] = None) -> None:
        """
        Add a benchmark result.
        
        Args:
            operation: Name of the operation
            data_structure: Name of the data structure
            size: Size of the test data
            time_ms: Execution time in milliseconds
            notes: Optional notes about the test
        """
        throughput = size / time_ms * 1000 if time_ms > 0 else 0
        
        result = BenchmarkResult(
            operation=operation,
            data_structure=data_structure,
            size=size,
            time_ms=time_ms,
            throughput_ops_per_sec=throughput,
            notes=notes
        )
        
        self.results.append(result)
    
    def clear_results(self) -> None:
        """Clear all benchmark results"""
        self.results.clear()
    
    def get_results(self) -> List[BenchmarkResult]:
        """Get all benchmark results"""
        return self.results.copy()
    
    def print_progress(self, operation: str, size: int, current: int, total: int) -> None:
        """Print progress information"""
        progress = (current / total) * 100
        print(f"  [{progress:5.1f}%] {operation} (size: {size:,})")
    
    @abstractmethod
    def run_xtructure_benchmarks(self) -> None:
        """Run benchmarks for Xtructure data structures"""
        pass
    
    @abstractmethod
    def run_python_benchmarks(self) -> None:
        """Run benchmarks for Python data structures"""
        pass
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """
        Run all benchmarks and return results.
        
        Returns:
            List of benchmark results
        """
        print(f"\n=== {self.name} Benchmarks ===")
        
        self.clear_results()
        
        try:
            print("Running Xtructure benchmarks...")
            self.run_xtructure_benchmarks()
            
            print("Running Python benchmarks...")
            self.run_python_benchmarks()
            
            print(f"Completed {len(self.results)} benchmark tests")
            
        except Exception as e:
            print(f"Error during benchmark: {e}")
            raise
        
        return self.get_results()
    
    def print_results_summary(self) -> None:
        """Print a summary of benchmark results"""
        if not self.results:
            print("No results to display")
            return
        
        print(f"\n{self.name} Results Summary:")
        print("-" * 60)
        
        # Group by data structure and operation
        grouped = {}
        for result in self.results:
            key = (result.data_structure, result.operation)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)
        
        for (data_structure, operation), results in sorted(grouped.items()):
            print(f"\n{data_structure} - {operation}:")
            print(f"{'Size':<10} {'Time (ms)':<12} {'Throughput (ops/sec)':<20}")
            print("-" * 50)
            
            for result in sorted(results, key=lambda x: x.size):
                print(f"{result.size:<10} {result.time_ms:<12.2f} {result.throughput_ops_per_sec:<20.0f}")
    
    def get_performance_comparisons(self) -> List[str]:
        """
        Generate performance comparison strings.
        
        Returns:
            List of comparison strings
        """
        comparisons = []
        
        # Group results for comparison
        xt_results = {}
        py_results = {}
        
        for result in self.results:
            key = (result.operation, result.size)
            if "Xtructure" in result.data_structure:
                xt_results[key] = result
            elif "Python" in result.data_structure:
                py_results[key] = result
        
        # Compare results
        for (operation, size), xt_result in xt_results.items():
            # Find corresponding Python result
            py_key_candidates = [
                (operation.replace("_batch", "_single").replace("_parallel", "_batch"), size),
                (operation.replace("_parallel", "_batch"), size),
                (operation, size)
            ]
            
            py_result = None
            for key in py_key_candidates:
                if key in py_results:
                    py_result = py_results[key]
                    break
            
            if py_result and py_result.throughput_ops_per_sec > 0 and xt_result.throughput_ops_per_sec > 0:
                speedup = py_result.throughput_ops_per_sec / xt_result.throughput_ops_per_sec
                
                if speedup > 1:
                    comparisons.append(
                        f"{xt_result.data_structure} {operation} (size {size:,}): "
                        f"{speedup:.2f}x SLOWER than {py_result.data_structure}"
                    )
                else:
                    comparisons.append(
                        f"{xt_result.data_structure} {operation} (size {size:,}): "
                        f"{1/speedup:.2f}x FASTER than {py_result.data_structure}"
                    )
        
        return comparisons


def safe_benchmark_operation(func: Callable, *args, **kwargs) -> Any:
    """
    Safely execute a benchmark operation with error handling.
    
    Args:
        func: Function to execute
        *args: Arguments for function
        **kwargs: Keyword arguments for function
        
    Returns:
        Function result or None if error occurred
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"Warning: Benchmark operation failed: {e}")
        return None


def configure_jax_for_benchmarking(backend: str = "cpu", enable_x64: bool = True) -> None:
    """
    Configure JAX settings optimal for benchmarking.
    
    Args:
        backend: JAX backend to use ("cpu", "gpu", etc.)
        enable_x64: Whether to enable 64-bit precision
    """
    import jax
    
    try:
        jax.config.update("jax_platform_name", backend)
        if enable_x64:
            jax.config.update("jax_enable_x64", enable_x64)
        
        print(f"JAX configured for benchmarking:")
        print(f"  Backend: {jax.default_backend()}")
        print(f"  Devices: {jax.devices()}")
        print(f"  X64 enabled: {enable_x64}")
        
    except Exception as e:
        print(f"Warning: Failed to configure JAX: {e}")


def format_number(num: float) -> str:
    """Format large numbers with appropriate units"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.2f}"