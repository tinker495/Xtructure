# Xtructure

A Python package providing JAX-optimized data structures, including a batched priority queue and a cuckoo hash table.

## Features

- Batched GPU Priority Queue (`BGPQ`)
- Cuckoo Hash Table (`HashTable`)
- Optimized for JAX

## Installation

```bash
pip install xtructure
```

## Documentation

Detailed documentation on how to use Xtructure is available in the `doc/` directory:

*   **[Core Concepts](./doc/core_concepts.md)**: Learn how to define custom data structures using `@xtructure_dataclass` and `FieldDescriptor`.
*   **[HashTable Usage](./doc/hashtable.md)**: Guide to using the Cuckoo hash table.
*   **[BGPQ Usage](./doc/bgpq.md)**: Guide to using the Batched GPU Priority Queue.

Quick examples can still be found below for a brief overview.

## Quick Examples

```python
import jax
import jax.numpy as jnp

from xtructure import HashTable, BGPQ, xtructure_dataclass, FieldDescriptor


# Define a custom data structure using xtructure_data
@xtructure_dataclass
class MyDataValue:
    a: FieldDescriptor[jnp.uint8]
    b: FieldDescriptor[jnp.uint32, (1, 2)]


# --- HashTable Example ---
print("--- HashTable Example ---")

# 1. Build the HashTable
# HashTable.build(pytree_def_type_class, inital_hash_seed, capacity)
table_capacity = 1000
hash_table = HashTable.build(MyDataValue, 1, table_capacity)  # Pass the class for build

# 3. Prepare data to insert
num_items_to_insert = 100
sample_data = MyDataValue.random((num_items_to_insert,), key=jax.random.PRNGKey(0))

# 4. Insert data
# HashTable.parallel_insert(table, samples, filled_mask (optional))
hash_table, inserted_mask, unique_mask, idxs, table_idxs = HashTable.parallel_insert(
    hash_table, sample_data, jnp.ones(num_items_to_insert, dtype=jnp.bool_)
)
print(f"HashTable: Inserted {jnp.sum(inserted_mask)} items. Unique items inserted: {jnp.sum(unique_mask)}")
print(f"HashTable size: {hash_table.size}")

# 5. Lookup data
# HashTable.lookup(table, item_to_lookup)
item_to_check = sample_data[0]
idx, table_idx, found = HashTable.lookup(hash_table, item_to_check)

if found:
    retrieved_item = hash_table.table[idx, table_idx]
    print(f"HashTable: Item found at index {idx}, table_index {table_idx}.")
else:
    print("HashTable: Item not found.")

# --- Batched GPU Priority Queue (BGPQ) Example ---
print("\n--- BGPQ Example ---")


# Define another custom data structure for the BGPQ values (can be the same or different)
@xtructure_dataclass
class MyHeapValue:
    id: FieldDescriptor[jnp.int32]
    data: FieldDescriptor[jnp.float32, (2,)]


# 1. Build a BGPQ
# BGPQ.build(max_size, batch_size, pytree_def_type_for_values_class)
pq_max_size = 2000
pq_batch_size = 64  # Items to insert/delete per operation
priority_queue = BGPQ.build(pq_max_size, pq_batch_size, MyHeapValue)  # Pass the class for build
print(f"BGPQ: Built with max_size={priority_queue.max_size}, batch_size={priority_queue.batch_size}")

# 2. Prepare keys and values to insert
num_items_to_insert_pq = 150
prng_key = jax.random.PRNGKey(10)
keys_for_pq = jax.random.uniform(prng_key, (num_items_to_insert_pq,)).astype(jnp.bfloat16)
prng_key, subkey = jax.random.split(prng_key)
values_for_pq = MyHeapValue.random((num_items_to_insert_pq,), key=subkey)

# 3. Insert data into BGPQ in batches
print(f"BGPQ: Starting to insert {num_items_to_insert_pq} items.")
for i in range(0, num_items_to_insert_pq, pq_batch_size):
    start_idx = i
    end_idx = min(i + pq_batch_size, num_items_to_insert_pq)

    current_keys_chunk = keys_for_pq[start_idx:end_idx]
    current_values_chunk = jax.tree_util.tree_map(lambda arr: arr[start_idx:end_idx], values_for_pq)

    keys_to_insert, values_to_insert = BGPQ.make_batched(current_keys_chunk, current_values_chunk, pq_batch_size)

    priority_queue = BGPQ.insert(priority_queue, keys_to_insert, values_to_insert)

print(f"BGPQ: Inserted items. Current size: {priority_queue.size}")

# 4. Delete minimums
if priority_queue.size > 0:
    priority_queue, min_keys, min_values = BGPQ.delete_mins(priority_queue)
    valid_mask = jnp.isfinite(min_keys)
    actual_min_keys = min_keys[valid_mask]
    actual_min_values = jax.tree_util.tree_map(lambda x: x[valid_mask], min_values)

    print(f"BGPQ: Deleted {jnp.sum(valid_mask)} items.")
    if jnp.sum(valid_mask) > 0:
        print(f"BGPQ: Smallest key deleted: {actual_min_keys[0]}")
    print(f"BGPQ: Size after deletion: {priority_queue.size}")
else:
    print("BGPQ: Heap is empty, cannot delete.")
```

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
