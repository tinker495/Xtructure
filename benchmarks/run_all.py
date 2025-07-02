"""
Main benchmark runner for all Xtructure data structures.

This module runs all available benchmarks and generates comprehensive reports
including hardware information and performance comparisons.
"""

import time
import csv
from typing import List, Optional, Dict, Any

import jax

from .common.base_benchmark import BenchmarkResult, configure_jax_for_benchmarking
from .common.hardware_info import get_hardware_info, print_hardware_info
from .stack_benchmark import run_stack_benchmark
from .queue_benchmark import run_queue_benchmark
from .hashtable_benchmark import run_hashtable_benchmark
from .bgpq_benchmark import run_bgpq_benchmark, check_bgpq_compatibility


class BenchmarkRunner:
    """Main benchmark runner that coordinates all benchmark modules"""
    
    def __init__(self, test_sizes: Optional[List[int]] = None, num_iterations: int = 3,
                 backend: str = "cpu", enable_x64: bool = True):
        """
        Initialize benchmark runner.
        
        Args:
            test_sizes: List of test sizes to benchmark
            num_iterations: Number of iterations per test
            backend: JAX backend to use ("cpu", "gpu", etc.)
            enable_x64: Whether to enable 64-bit precision
        """
        self.test_sizes = test_sizes or [100, 1000, 5000, 10000]
        self.num_iterations = num_iterations
        self.backend = backend
        self.enable_x64 = enable_x64
        self.all_results: List[BenchmarkResult] = []
        self.hardware_info = None
        self.start_time = None
        self.end_time = None
    
    def setup_environment(self) -> None:
        """Configure JAX and collect hardware information"""
        print("Setting up benchmark environment...")
        
        # Configure JAX
        configure_jax_for_benchmarking(self.backend, self.enable_x64)
        
        # Collect hardware information
        self.hardware_info = get_hardware_info()
        
        print("Environment setup complete.\n")
    
    def run_stack_benchmarks(self) -> List[BenchmarkResult]:
        """Run stack benchmarks"""
        print("="*60)
        print("RUNNING STACK BENCHMARKS")
        print("="*60)
        
        try:
            results = run_stack_benchmark(self.test_sizes, self.num_iterations)
            print(f"Stack benchmarks completed: {len(results)} results")
            return results
        except Exception as e:
            print(f"Error in stack benchmarks: {e}")
            return []
    
    def run_queue_benchmarks(self) -> List[BenchmarkResult]:
        """Run queue benchmarks"""
        print("="*60)
        print("RUNNING QUEUE BENCHMARKS")
        print("="*60)
        
        try:
            results = run_queue_benchmark(self.test_sizes, self.num_iterations)
            print(f"Queue benchmarks completed: {len(results)} results")
            return results
        except Exception as e:
            print(f"Error in queue benchmarks: {e}")
            return []
    
    def run_hashtable_benchmarks(self) -> List[BenchmarkResult]:
        """Run hashtable benchmarks"""
        print("="*60)
        print("RUNNING HASHTABLE BENCHMARKS")
        print("="*60)
        
        try:
            results = run_hashtable_benchmark(self.test_sizes, self.num_iterations)
            print(f"HashTable benchmarks completed: {len(results)} results")
            return results
        except Exception as e:
            print(f"Error in hashtable benchmarks: {e}")
            return []
    
    def run_bgpq_benchmarks(self) -> List[BenchmarkResult]:
        """Run BGPQ benchmarks (if compatible)"""
        print("="*60)
        print("RUNNING BGPQ BENCHMARKS")
        print("="*60)
        
        if not check_bgpq_compatibility():
            print("BGPQ benchmarks skipped due to compatibility issues")
            return []
        
        try:
            results = run_bgpq_benchmark(self.test_sizes, self.num_iterations)
            print(f"BGPQ benchmarks completed: {len(results)} results")
            return results
        except Exception as e:
            print(f"Error in BGPQ benchmarks: {e}")
            return []
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """
        Run all available benchmarks.
        
        Returns:
            List of all benchmark results
        """
        self.start_time = time.time()
        print("Starting comprehensive Xtructure benchmarks...")
        
        # Setup environment
        self.setup_environment()
        
        # Print hardware information
        if self.hardware_info:
            print_hardware_info(self.hardware_info)
        
        # Run all benchmark modules
        all_results = []
        
        # Stack benchmarks
        stack_results = self.run_stack_benchmarks()
        all_results.extend(stack_results)
        
        # Queue benchmarks  
        queue_results = self.run_queue_benchmarks()
        all_results.extend(queue_results)
        
        # HashTable benchmarks
        hashtable_results = self.run_hashtable_benchmarks()
        all_results.extend(hashtable_results)
        
        # BGPQ benchmarks (if available)
        bgpq_results = self.run_bgpq_benchmarks()
        all_results.extend(bgpq_results)
        
        self.all_results = all_results
        self.end_time = time.time()
        
        print("\n" + "="*80)
        print("ALL BENCHMARKS COMPLETED")
        print("="*80)
        print(f"Total tests run: {len(all_results)}")
        print(f"Total time: {self.end_time - self.start_time:.1f} seconds")
        
        return all_results
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a comprehensive summary report"""
        if not self.all_results:
            return {}
        
        # Group results by data structure
        structure_groups = {}
        for result in self.all_results:
            key = (result.data_structure, result.operation)
            if key not in structure_groups:
                structure_groups[key] = []
            structure_groups[key].append(result)
        
        # Calculate performance comparisons
        comparisons = self._calculate_performance_comparisons()
        
        # Create summary
        summary = {
            'hardware_info': self.hardware_info,
            'test_config': {
                'test_sizes': self.test_sizes,
                'num_iterations': self.num_iterations,
                'backend': self.backend,
                'enable_x64': self.enable_x64
            },
            'execution_info': {
                'start_time': self.start_time,
                'end_time': self.end_time,
                'duration_seconds': self.end_time - self.start_time if self.end_time and self.start_time else 0,
                'total_tests': len(self.all_results)
            },
            'results_by_structure': structure_groups,
            'performance_comparisons': comparisons,
            'raw_results': self.all_results
        }
        
        return summary
    
    def _calculate_performance_comparisons(self) -> List[str]:
        """Calculate performance comparisons between Xtructure and Python"""
        comparisons = []
        
        # Group results for comparison
        xt_results = {}
        py_results = {}
        
        for result in self.all_results:
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
    
    def print_detailed_report(self) -> None:
        """Print a detailed benchmark report"""
        print("\n" + "="*80)
        print("DETAILED BENCHMARK REPORT")
        print("="*80)
        
        if not self.all_results:
            print("No results to display")
            return
        
        # Group and display results by data structure
        structure_groups = {}
        for result in self.all_results:
            key = (result.data_structure, result.operation)
            if key not in structure_groups:
                structure_groups[key] = []
            structure_groups[key].append(result)
        
        for (data_structure, operation), results in sorted(structure_groups.items()):
            print(f"\n{data_structure} - {operation}:")
            print(f"{'Size':<10} {'Time (ms)':<15} {'Throughput (ops/sec)':<20}")
            print("-" * 50)
            
            for result in sorted(results, key=lambda x: x.size):
                print(f"{result.size:<10} {result.time_ms:<15.2f} {result.throughput_ops_per_sec:<20.0f}")
        
        # Performance comparisons
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISONS (Xtructure vs Python)")
        print("="*80)
        
        comparisons = self._calculate_performance_comparisons()
        for comparison in comparisons:
            print(comparison)
    
    def save_results_to_csv(self, filename: Optional[str] = None) -> str:
        """Save benchmark results to CSV file"""
        if not filename:
            timestamp = int(time.time())
            filename = f"xtructure_benchmark_results_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['operation', 'data_structure', 'size', 'time_ms', 
                         'throughput_ops_per_sec', 'notes']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in self.all_results:
                writer.writerow({
                    'operation': result.operation,
                    'data_structure': result.data_structure,
                    'size': result.size,
                    'time_ms': result.time_ms,
                    'throughput_ops_per_sec': result.throughput_ops_per_sec,
                    'notes': result.notes or ''
                })
        
        print(f"Results saved to: {filename}")
        return filename


def run_all_benchmarks(test_sizes: Optional[List[int]] = None, num_iterations: int = 3,
                      backend: str = "cpu", enable_x64: bool = True) -> BenchmarkRunner:
    """
    Run all Xtructure benchmarks and return a runner with results.
    
    Args:
        test_sizes: List of test sizes to benchmark
        num_iterations: Number of iterations per test
        backend: JAX backend to use
        enable_x64: Whether to enable 64-bit precision
        
    Returns:
        BenchmarkRunner instance with all results
    """
    runner = BenchmarkRunner(test_sizes, num_iterations, backend, enable_x64)
    runner.run_all_benchmarks()
    return runner


def main():
    """Main entry point for benchmark execution"""
    # Run all benchmarks
    runner = run_all_benchmarks()
    
    # Print detailed report
    runner.print_detailed_report()
    
    # Save results to CSV
    runner.save_results_to_csv()


if __name__ == "__main__":
    main()