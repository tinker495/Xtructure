"""
Common test data structures and generation utilities for benchmarks.
"""

import jax
import jax.numpy as jnp
import numpy as np
from xtructure import xtructure_dataclass, FieldDescriptor


@xtructure_dataclass
class BenchmarkValue:
    """Test data structure used across all benchmarks"""
    id: FieldDescriptor[jnp.uint32, (), 0]
    data: FieldDescriptor[jnp.float32, (4,), 0.0]  # 4-element array data
    flag: FieldDescriptor[jnp.bool_, (), False]


def create_test_data(size: int, seed: int = 42) -> BenchmarkValue:
    """
    Create test data for benchmarks.
    
    Args:
        size: Number of elements to generate
        seed: Random seed for reproducible results
        
    Returns:
        BenchmarkValue instance with specified size
    """
    key = jax.random.PRNGKey(seed)
    
    return BenchmarkValue(
        id=jnp.arange(size, dtype=jnp.uint32),
        data=jax.random.normal(key, (size, 4)).astype(jnp.float32),
        flag=jnp.ones(size, dtype=jnp.bool_)
    )


def create_python_test_data(size: int, seed: int = 42) -> list:
    """
    Create equivalent Python test data for comparison.
    
    Args:
        size: Number of elements to generate
        seed: Random seed for reproducible results
        
    Returns:
        List of tuples containing (id, data, flag)
    """
    np.random.seed(seed)
    
    return [
        (i, np.random.rand(4).astype(np.float32), True)
        for i in range(size)
    ]


def create_python_dict_data(size: int, seed: int = 42) -> dict:
    """
    Create Python dict test data for hash table comparison.
    
    Args:
        size: Number of elements to generate
        seed: Random seed for reproducible results
        
    Returns:
        Dictionary mapping integers to tuples
    """
    np.random.seed(seed)
    
    return {
        i: (i, np.random.rand(4).astype(np.float32), True)
        for i in range(size)
    }


def create_priority_queue_data(size: int, seed: int = 42) -> tuple:
    """
    Create test data for priority queue benchmarks.
    
    Args:
        size: Number of elements to generate
        seed: Random seed for reproducible results
        
    Returns:
        Tuple of (keys, values) for priority queue operations
    """
    key = jax.random.PRNGKey(seed)
    keys = jax.random.uniform(key, (size,)).astype(jnp.float32)
    values = create_test_data(size, seed)
    
    return keys, values


def create_python_heap_data(size: int, seed: int = 42) -> list:
    """
    Create Python heapq test data for priority queue comparison.
    
    Args:
        size: Number of elements to generate
        seed: Random seed for reproducible results
        
    Returns:
        List of (priority, data) tuples suitable for heapq
    """
    np.random.seed(seed)
    
    return [
        (np.random.rand(), (i, np.random.rand(4).astype(np.float32), True))
        for i in range(size)
    ]