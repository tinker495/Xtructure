# Xtructure

A Python package providing JAX-optimized data structures, including a batched priority queue and a cuckoo hash table.

## Features

- Batched GPU Priority Queue (`BGPQ`)
- Cuckoo Hash Table (`HashTable`)
- Optimized for JAX

## Installation

```bash
# Coming soon - after uploading to PyPI
# pip install Xtructure
```

## Usage

```python
import jax
import jax.numpy as jnp
from Xtructure import BGPQ, HeapValue, HashTable # Assuming __init__.py is set up

# Example usage (conceptual)
# See individual modules for detailed API and examples

# BGPQ Example
# @bgpq_value_dataclass
# class MyValue(HeapValue):
#     data: jax.Array
# 
#     @staticmethod
#     def default(shape=()):
#         return MyValue(data=jnp.full(shape + (1,), -1, dtype=jnp.int32))
#
# queue = BGPQ.build(total_size=100, batch_size=10, value_class=MyValue)
# keys_to_insert = jnp.array([5.0, 2.0, 8.0]) # ... and more
# values_to_insert = MyValue(data=jnp.array([[10], [20], [30]])) # ... and more
# # queue = queue.insert(keys_to_insert, values_to_insert)

# HashTable Example
# class MyPuzzleState:
#     # Define your state dataclass here, compatible with Puzzle.State if using original hash_func_builder context
#     # For standalone, ensure it has a .default() static method
#     board: jax.Array
#
#     @staticmethod
#     def default(shape=()): # Dummy shape for example
#         if isinstance(shape, int):
#             shape = (shape,)
#         return MyPuzzleState(board=jnp.zeros(shape + (4,4), dtype=jnp.int32))
#
# hash_fn = hash_func_builder(MyPuzzleState) # If Puzzle.State is defined and MyPuzzleState adheres to it
# table = HashTable.build(statecls=MyPuzzleState, seed=42, capacity=1000)
# # state_to_insert = MyPuzzleState(board=jnp.arange(16).reshape(4,4))
# # table, inserted = table.insert(hash_fn, state_to_insert)
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details (you'll need to create this file).
