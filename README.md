# Xtructure

A Python package providing JAX-optimized data structures, including a batched priority queue and a cuckoo hash table.

## Features

- Batched GPU Priority Queue (`BGPQ`)
- Cuckoo Hash Table (`HashTable`)
- Optimized for JAX

## Installation

```bash
pip install Xtructure
```

## Usage

```python
import jax
import jax.numpy as jnp

from Xtructure import HashTable, hash_func_builder, BGPQ, KEY_DTYPE, xtructure_dataclass, FieldDescriptor

# Define a custom data structure using xtructure_data
@xtructure_dataclass
class MyDataValue:
    a: FieldDescriptor(jnp.uint8) # type: ignore
    b: FieldDescriptor(jnp.uint32, (1, 2)) # type: ignore

# --- HashTable Example ---
print("--- HashTable Example ---")

# 1. Build a hash function for your custom data structure
my_hash_func = hash_func_builder(MyDataValue)

# 2. Build the HashTable
# HashTable.build(pytree_def_type, num_hash_funcs, capacity)
table_capacity = 1000
hash_table = HashTable.build(MyDataValue, 1, table_capacity)

# 3. Prepare data to insert
num_items_to_insert = 100
sample_data = MyDataValue.random((num_items_to_insert,))

# 4. (Optional) Batch data if inserting more items than internal batch size
# This is a helper, often parallel_insert handles batching internally if data isn't pre-batched.
# batch_size_for_insert = 50 # Example internal batch size
# batched_sample_data, filled_mask = HashTable.make_batched(MyDataValue, sample_data, batch_size_for_insert)

# 5. Insert data
# HashTable.parallel_insert(table, hash_func, samples, filled_mask (optional))
hash_table, inserted_mask, unique_mask, idxs, table_idxs = HashTable.parallel_insert(
    hash_table, my_hash_func, sample_data, jnp.ones(num_items_to_insert, dtype=jnp.bool_)
)
print(f"HashTable: Inserted {jnp.sum(inserted_mask)} items. Unique items inserted: {jnp.sum(unique_mask)}")
print(f"HashTable size: {hash_table.size}")

# 6. Lookup data
# HashTable.lookup(table, hash_func, item_to_lookup)
item_to_check = sample_data[0]
idx, table_idx, found = HashTable.lookup(hash_table, my_hash_func, item_to_check)

if found:
    retrieved_item = hash_table.table[idx, table_idx]
    print(f"HashTable: Item found at index {idx}, table_index {table_idx}.")
    # You can compare retrieved_item with item_to_check
else:
    print("HashTable: Item not found.")

# --- Batched GPU Priority Queue (BGPQ) Example ---
print("\n--- BGPQ Example ---")

# Define another custom data structure for the BGPQ values (can be the same or different)
@xtructure_dataclass
class MyHeapValue:
    id: FieldDescriptor(jnp.int32) # type: ignore
    data: FieldDescriptor(jnp.float32, (2,)) # type: ignore

# 1. Build a BGPQ
# BGPQ.build(max_size, batch_size, pytree_def_type_for_values)
pq_max_size = 2000
pq_batch_size = 64 # Items to insert/delete per operation
priority_queue = BGPQ.build(pq_max_size, pq_batch_size, MyHeapValue)
print(f"BGPQ: Built with max_size={pq_max_size}, batch_size={pq_batch_size}")

# 2. Prepare keys and values to insert
num_items_to_insert_pq = 150
keys_for_pq = jax.random.uniform(jax.random.PRNGKey(10), (num_items_to_insert_pq,)).astype(KEY_DTYPE)
values_for_pq = MyHeapValue.random((num_items_to_insert_pq,), key=jax.random.PRNGKey(11))

# 3. Insert data into BGPQ in batches
# BGPQ.insert expects keys and values to be shaped to pq_batch_size.
# We iterate through our data in chunks.
# BGPQ.make_batched is used here to ensure each chunk fed to insert()
# is correctly padded if it's the last, smaller chunk.
print(f"BGPQ: Starting to insert {num_items_to_insert_pq} items.")
for i in range(0, num_items_to_insert_pq, pq_batch_size):
    start_idx = i
    end_idx = min(i + pq_batch_size, num_items_to_insert_pq)

    current_keys_chunk = keys_for_pq[start_idx:end_idx]
    # For PyTrees, slice each leaf array
    current_values_chunk = jax.tree_util.tree_map(lambda arr: arr[start_idx:end_idx], values_for_pq)

    # Pad the chunk if it's smaller than pq_batch_size (typically the last chunk)
    # If chunk is already pq_batch_size, make_batched should handle it appropriately (e.g. no-op or truncate if larger, though our chunks are <= pq_batch_size)
    keys_to_insert, values_to_insert = BGPQ.make_batched(current_keys_chunk, current_values_chunk, pq_batch_size)
    
    priority_queue = BGPQ.insert(priority_queue, keys_to_insert, values_to_insert)

print(f"BGPQ: Inserted items. Current size: {priority_queue.size}")

# 4. Delete minimums
# BGPQ.delete_mins(heap) - deletes batch_size items
if priority_queue.size > 0:
    priority_queue, min_keys, min_values = BGPQ.delete_mins(priority_queue)
    # min_keys and min_values will have shape (pq_batch_size, ...)
    # You might want to filter by jnp.isfinite(min_keys) if the heap had < batch_size items
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

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details (you'll need to create this file).
