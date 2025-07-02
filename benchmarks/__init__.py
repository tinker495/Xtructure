"""
Xtructure Benchmarks Package

This package contains comprehensive benchmarks for Xtructure data structures
compared against standard Python implementations.
"""

from .run_all import run_all_benchmarks, main, BenchmarkRunner

__version__ = "1.0.0"
__author__ = "Xtructure Benchmarks"

__all__ = [
    "run_all_benchmarks",
    "main",
    "BenchmarkRunner"
]