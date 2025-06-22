# Xtructure

A Python package providing JAX-optimized data structures, including a batched priority queue and a cuckoo hash table.

## Features

- Stack (`Stack`): A LIFO (Last-In, First-Out) data structure.
- Queue (`Queue`): A FIFO (First-In, First-Out) data structure.
- Batched GPU Priority Queue (`BGPQ`): A batched priority queue optimized for GPU operations.
- Cuckoo Hash Table (`HashTable`): A cuckoo hash table optimized for GPU operations.
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

Quick examples can still be found below for a brief overview.

## Quick Examples

```python
import jax
import jax.numpy as jnp

from xtructure import xtructure_dataclass, FieldDescriptor
from xtructure import HashTable, BGPQ


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
