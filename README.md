<div align="center">
  <img src="images/Xtructure.svg" alt="Xtructure Logo" width="600">
</div>

# Xtructure

A Python package providing JAX-optimized data structures, including a batched priority queue and a cuckoo hash table.

## Features

- Stack (`Stack`): A LIFO (Last-In, First-Out) data structure.
- Queue (`Queue`): A FIFO (First-In, First-Out) data structure.
- Batched GPU Priority Queue (`BGPQ`): A batched priority queue optimized for GPU operations.
- Cuckoo Hash Table (`HashTable`): A cuckoo hash table optimized for GPU operations.
- Xtructure NumPy (`xtructure_numpy`): JAX-compatible operations for dataclass manipulation including concatenation, stacking, padding, conditional selection, deduplication, and element selection.
- Optimized for JAX.

## Installation

```bash
pip install xtructure
pip install git+https://github.com/tinker495/xtructure.git # recommended
```

Currently under active development, with frequent updates and potential bug fixes. For the most up-to-date version, it is recommended to install directly from the Git repository.

## Documentation

Detailed documentation on how to use Xtructure is available in the `doc/` directory:

*   **[Core Concepts](./doc/core_concepts.md)**: Learn how to define custom data structures using `@xtructure_dataclass` and `FieldDescriptor`.
*   **[Stack Usage](./doc/stack.md)**: Guide to using the Stack data structure.
*   **[Queue Usage](./doc/queue.md)**: Guide to using the Queue data structure.
*   **[BGPQ Usage](./doc/bgpq.md)**: Guide to using the Batched GPU Priority Queue.
*   **[HashTable Usage](./doc/hashtable.md)**: Guide to using the Cuckoo hash table.
*   **[Xtructure NumPy Operations](./doc/xnp.md)**: Guide to using `xtructure_numpy` (`xnp`) operations for dataclass manipulation.

Quick examples can still be found below for a brief overview.

## Quick Examples

```python
import jax
import jax.numpy as jnp

from xtructure import xtructure_dataclass, FieldDescriptor
from xtructure import HashTable, BGPQ
from xtructure import numpy as xnp  # Recommended import method


# Define a custom data structure using xtructure_data
@xtructure_dataclass
class MyDataValue:
    a: FieldDescriptor[jnp.uint8]
    b: FieldDescriptor[jnp.uint32, (1, 2)]


# --- HashTable Example ---
print("--- HashTable Example ---")

# Build a HashTable for a custom data structure
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
hash_table: HashTable = HashTable.build(MyDataValue, 1, capacity=1000)

# Insert random data
items_to_insert = MyDataValue.random((100,), key=subkey)
hash_table, inserted_mask, _, _ = hash_table.parallel_insert(items_to_insert)
print(f"HashTable: Inserted {jnp.sum(inserted_mask)} items. Current size: {hash_table.size}")

# Lookup an item
item_to_find = items_to_insert[0]
_, found = hash_table.lookup(item_to_find)
print(f"HashTable: Item found? {found}")

# Parallel lookup for multiple items
items_to_lookup = items_to_insert[:5]
idxs, founds = hash_table.lookup_parallel(items_to_lookup)
print(f"HashTable: Found {jnp.sum(founds)} out of {len(items_to_lookup)} items in parallel lookup.")


# --- Batched GPU Priority Queue (BGPQ) Example ---
print("\n--- BGPQ Example ---")

# Build a BGPQ with a specific batch size
key = jax.random.PRNGKey(1)
pq_batch_size = 64
priority_queue = BGPQ.build(
    2000,
    pq_batch_size,
    MyDataValue,
)
print(f"BGPQ: Built with max_size={priority_queue.max_size}, batch_size={priority_queue.batch_size}")

# Prepare a batch of keys and values to insert
key, subkey1, subkey2 = jax.random.split(key, 3)
keys_to_insert = jax.random.uniform(subkey1, (pq_batch_size,)).astype(jnp.bfloat16)
values_to_insert = MyDataValue.random((pq_batch_size,), key=subkey2)

# Insert data
priority_queue = BGPQ.insert(priority_queue, keys_to_insert, values_to_insert)
print(f"BGPQ: Inserted a batch. Current size: {priority_queue.size}")

# Delete a batch of minimums
priority_queue, min_keys, _ = BGPQ.delete_mins(priority_queue)
valid_mask = jnp.isfinite(min_keys)
print(f"BGPQ: Deleted {jnp.sum(valid_mask)} items. Size after deletion: {priority_queue.size}")


# --- Xtructure NumPy Operations Example ---
print("\n--- Xtructure NumPy Operations Example ---")

# Create some test data
data1 = MyDataValue.default((3,))
data1 = data1.replace(
    a=jnp.array([1, 2, 3], dtype=jnp.uint8), b=jnp.array([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]], dtype=jnp.uint32)
)

data2 = MyDataValue.default((2,))
data2 = data2.replace(
    a=jnp.array([4, 5], dtype=jnp.uint8), b=jnp.array([[[7.0, 8.0]], [[9.0, 10.0]]], dtype=jnp.uint32)
)

# Concatenate dataclasses
concatenated = xnp.concat([data1, data2])
print(f"XNP: Concatenated shape: {concatenated.shape.batch}")

# Stack dataclasses (requires same batch shape)
data3 = MyDataValue.default((3,))
data3 = data3.replace(
    a=jnp.array([6, 7, 8], dtype=jnp.uint8),
    b=jnp.array([[[11.0, 12.0]], [[13.0, 14.0]], [[15.0, 16.0]]], dtype=jnp.uint32),
)
stacked = xnp.stack([data1, data3])
print(f"XNP: Stacked shape: {stacked.shape.batch}")

# Conditional operations
condition = jnp.array([True, False, True])
filtered = xnp.where(condition, data1, -1)
print(f"XNP: Conditional filtering: {filtered.a}")

# Unique filtering
mask = xnp.unique_mask(data1)
print(f"XNP: Unique mask: {mask}")

# Take specific elements
taken = xnp.take(data1, jnp.array([0, 2]))
print(f"XNP: Taken elements: {taken.a}")

# Update values conditionally
indices = jnp.array([0, 1])
condition = jnp.array([True, False])
new_values = MyDataValue.default((2,))
new_values = new_values.replace(
    a=jnp.array([99, 100], dtype=jnp.uint8), b=jnp.array([[[99.0, 99.0]], [[100.0, 100.0]]], dtype=jnp.uint32)
)
updated = xnp.update_on_condition(data1, indices, condition, new_values)
print(f"XNP: Updated elements: {updated.a}")
```

## Working Example

For a fully functional example using `Xtructure`, check out the [JAxtar](https://github.com/tinker495/JAxtar) repository. `JAxtar` demonstrates how to use `Xtructure` to build a JAX-native, parallelizable A* and Q* solver for neural heuristic search research, showcasing the library in a real, high-performance computing workflow.

## Benchmark Results

Measured on NVIDIA GeForce RTX 5090.

Raw JSON links are in the last column; plots show ops/sec by batch size.

| Structure | Op A (plot) | Op B (plot) | Results |
| --- | --- | --- | --- |
| Stack | ![Push](./xtructure_benchmarks/benchmark_data/stack_push_performance.png) | ![Pop](./xtructure_benchmarks/benchmark_data/stack_pop_performance.png) | [`stack_results.json`](./xtructure_benchmarks/benchmark_data/stack_results.json) |
| Queue | ![Enqueue](./xtructure_benchmarks/benchmark_data/queue_enqueue_performance.png) | ![Dequeue](./xtructure_benchmarks/benchmark_data/queue_dequeue_performance.png) | [`queue_results.json`](./xtructure_benchmarks/benchmark_data/queue_results.json) |
| BGPQ (Heap) | ![Insert](./xtructure_benchmarks/benchmark_data/heap_insert_performance.png) | ![Delete](./xtructure_benchmarks/benchmark_data/heap_delete_performance.png) | [`heap_results.json`](./xtructure_benchmarks/benchmark_data/heap_results.json) |
| Hash Table | ![Insert](./xtructure_benchmarks/benchmark_data/hashtable_insert_performance.png) | ![Lookup](./xtructure_benchmarks/benchmark_data/hashtable_lookup_performance.png) | [`hashtable_results.json`](./xtructure_benchmarks/benchmark_data/hashtable_results.json) |

### Detailed Results (median ops/sec ± IQR; speedup = xtructure/python)

Values are shown in the order 1,024 / 4,096 / 16,384.

| Structure | Operation | xtructure (median ± IQR) | python (median ± IQR) | Speedup (×) |
| --- | --- | --- | --- | --- |
| Stack | Push | 14,535,435 ± 2,635,103<br/>80,672,798 ± 24,314,367<br/>269,429,330 ± 37,552,312 | 42,558 ± 1,908<br/>51,527 ± 14,995<br/>45,999 ± 1,314 | 341.64 / 1,566.09 / 5,859.23 |
| Stack | Pop | 5,266,311 ± 786,689<br/>13,375,678 ± 2,135,048<br/>30,326,310 ± 1,865,609 | 251,700 ± 30,268<br/>253,131 ± 11,311<br/>246,296 ± 10,523 | 20.93 / 52.85 / 123.17 |
| Queue | Enqueue | 12,870,713 ± 2,238,375<br/>57,148,845 ± 25,254,233<br/>225,195,655 ± 76,034,047 | 49,916 ± 1,391<br/>50,113 ± 13,593<br/>47,148 ± 3,058 | 257.83 / 1,140.49 / 4,777.20 |
| Queue | Dequeue | 5,022,229 ± 483,230<br/>12,913,416 ± 1,668,073<br/>29,107,706 ± 2,382,835 | 259,493 ± 28,879<br/>244,835 ± 4,627<br/>241,120 ± 5,366 | 19.36 / 52.77 / 120.74 |
| BGPQ (Heap) | Insert | 6,057,863 ± 943,945<br/>22,304,210 ± 8,135,547<br/>85,577,360 ± 9,516,407 | 30,375,845 ± 362,698<br/>27,901,911 ± 2,065,789<br/>29,658,943 ± 1,095,604 | 0.20 / 0.80 / 2.89 |
| BGPQ (Heap) | Delete | 3,721,160 ± 284,567<br/>11,557,757 ± 2,903,894<br/>26,708,899 ± 1,896,707 | 5,083,955 ± 99,315<br/>3,914,545 ± 96,803<br/>3,164,385 ± 289,304 | 0.73 / 2.95 / 8.44 |
| Hash Table | Insert | 289,286 ± 1,768<br/>1,151,791 ± 18,977<br/>3,907,092 ± 60,753 | 37,297 ± 1,333<br/>24,822 ± 3,206<br/>37,056 ± 13,532 | 7.76 / 46.43 / 105.49 |
| Hash Table | Lookup | 317,278 ± 2,884<br/>1,373,139 ± 54,821<br/>4,671,074 ± 160,999 | 39,219 ± 1,534<br/>39,613 ± 443<br/>36,586 ± 2,419 | 8.09 / 34.67 / 127.67 |

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
