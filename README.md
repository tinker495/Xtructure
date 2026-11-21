<div align="center">
  <img src="images/Xtructure.svg" alt="Xtructure Logo" width="600">
</div>

# Xtructure

A Python package providing JAX-optimized data structures, including a batched priority queue and a bucketed double hash table.

## Features

- Stack (`Stack`): A LIFO (Last-In, First-Out) data structure.
- Queue (`Queue`): A FIFO (First-In, First-Out) data structure.
- Batched GPU Priority Queue (`BGPQ`): A batched priority queue optimized for GPU operations.
- Bucketed Double Hash Table (`HashTable`): A hash table optimized for GPU operations using double hashing and bucketed storage.
- Xtructure NumPy (`xtructure.numpy` / `xnp`): JAX-compatible operations for dataclass manipulation including concatenation, stacking, padding, conditional selection, deduplication, and element selection.
- Optimized for JAX with Structure-of-Arrays (SoA) storage and Array-of-Structures (AoS) semantics.

## Installation

```bash
uv add "xtructure @ git+https://github.com/tinker495/xtructure.git"
```

For local development, run `uv sync --group dev`, then use `uv run pytest`.

## Documentation

Detailed documentation on how to use Xtructure is available at [https://tinker495.github.io/Xtructure/](https://tinker495.github.io/Xtructure/).

*   **[Core Concepts](https://tinker495.github.io/Xtructure/core_concepts.html)**: Learn how to define custom data structures by inheriting from `Xtructurable` and declaring fields with `FieldDescriptor`.
*   **[Stack Usage](https://tinker495.github.io/Xtructure/stack.html)**: Guide to using the Stack data structure.
*   **[Queue Usage](https://tinker495.github.io/Xtructure/queue.html)**: Guide to using the Queue data structure.
*   **[BGPQ Usage](https://tinker495.github.io/Xtructure/bgpq.html)**: Guide to using the Batched GPU Priority Queue.
*   **[HashTable Usage](https://tinker495.github.io/Xtructure/hashtable.html)**: Guide to using the bucketed double hash table.
*   **[Xtructure NumPy Operations](https://tinker495.github.io/Xtructure/xnp.html)**: Guide to using `xtructure.numpy` (`xnp`) operations for dataclass manipulation.
*   **[Structure Layout Flexibility](https://tinker495.github.io/Xtructure/layout_flexibility.html)**: How SoA storage and AoS semantics coexist in Xtructure.

## Quick Examples

Define structures by inheriting from `Xtructurable`, with options in the class header.

An instance's `.shape` is a plain tuple of shared batch axes. Read field shape metadata with `.layout.field_shape_for("field")`.

Data construction and operations validate field dtypes, intrinsic shapes, shared batch axes, and custom validators. Use `replace` for validated changes. JAX PyTree reconstruction preserves structure and defers validation until a data API is used; see [runtime validation](docs/core_concepts.md#runtime-validation) for the boundary.

```python
import jax
import jax.numpy as jnp

from xtructure import BGPQ, FieldDescriptor, HashTable, Xtructurable
from xtructure import numpy as xnp


class MyDataValue(Xtructurable):
    a: FieldDescriptor.scalar(dtype=jnp.uint8)
    b: FieldDescriptor.tensor(dtype=jnp.uint32, shape=(1, 2))


key = jax.random.PRNGKey(0)
items = MyDataValue.random((64,), key=key)

table = HashTable.build(MyDataValue, seed=1, capacity=256)
table, inserted_mask, _, _ = table.parallel_insert(items)
_, founds = table.lookup_parallel(items[:5])

pq = BGPQ.build(total_size=256, batch_size=64, value_class=MyDataValue)
keys = jax.random.uniform(key, (64,)).astype(jnp.float16)
pq = pq.insert(keys, items)
pq, min_keys, _ = pq.delete_mins()

data = MyDataValue.default((3,)).replace(
    a=jnp.array([1, 2, 1], dtype=jnp.uint8),
    b=jnp.array([[[1, 2]], [[3, 4]], [[1, 2]]], dtype=jnp.uint32),
)
mask = xnp.unique_mask(data)
taken = xnp.take(data, jnp.array([0, 2]), axis=0)
merged = xnp.concat([data, taken])
```

## Working Example

For a fully functional example using `Xtructure`, check out the [JAxtar](https://github.com/tinker495/JAxtar) repository. `JAxtar` demonstrates how to use `Xtructure` to build a JAX-native, parallelizable A* and Q* solver for neural heuristic search research, showcasing the library in a real, high-performance computing workflow.

## Citation

If you use this code in your research, please cite:

```
@software{kyuseokjung2025xtructure,
    title={xtructure: JAX-optimized Data Structures},
    author={Kyuseok Jung},
    url = {https://github.com/tinker495/Xtructure},
    year={2025},
}
```
