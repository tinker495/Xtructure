# `BGPQ` (Batched GPU Priority Queue) Usage

A priority queue optimized for batched operations on GPUs. It maintains items sorted by a key.

```python
import jax
import jax.numpy as jnp
from xtructure import BGPQ, xtructure_dataclass, FieldDescriptor


# Define a data structure for BGPQ values (as an example from core_concepts.md)
@xtructure_dataclass
class MyHeapItem:
    task_id: FieldDescriptor.scalar(dtype=jnp.int32)
    payload: FieldDescriptor.tensor(dtype=jnp.float64, shape=(2, 2))


# 1. Build a BGPQ
#    BGPQ.build(total_size, batch_size, value_class)
pq_total_size = 2000  # Max number of items
pq_batch_size = 64  # Items to insert/delete per operation
priority_queue = BGPQ.build(pq_total_size, pq_batch_size, MyHeapItem)
# Note: MyHeapItem (the class itself) is passed.

print(f"BGPQ: Built with max_size={priority_queue.max_size}, batch_size={priority_queue.batch_size}")

# 2. Prepare keys and values to insert
num_items_to_insert_pq = 150
prng_key = jax.random.PRNGKey(10)
keys_for_pq = jax.random.uniform(prng_key, (num_items_to_insert_pq,)).astype(jnp.float16)
prng_key, subkey = jax.random.split(prng_key)
values_for_pq = MyHeapItem.random(shape=(num_items_to_insert_pq,), key=subkey)

# 3. Insert data into BGPQ in batches
#    BGPQ.insert expects keys and values to be shaped to pq_batch_size.
#    Loop through data in chunks and use BGPQ.make_batched for padding.
print(f"BGPQ: Starting to insert {num_items_to_insert_pq} items.")
for i in range(0, num_items_to_insert_pq, pq_batch_size):
    start_idx = i
    end_idx = min(i + pq_batch_size, num_items_to_insert_pq)

    current_keys_chunk = keys_for_pq[start_idx:end_idx]
    # For PyTrees (like our MyHeapItem), slice each leaf array
    current_values_chunk = jax.tree_util.tree_map(lambda arr: arr[start_idx:end_idx], values_for_pq)

    # Pad the chunk if it's smaller than pq_batch_size
    keys_to_insert, values_to_insert = BGPQ.make_batched(current_keys_chunk, current_values_chunk, pq_batch_size)

    priority_queue = BGPQ.insert(priority_queue, keys_to_insert, values_to_insert)

print(f"BGPQ: Inserted items. Current size: {priority_queue.size}")

# 4. Delete minimums (deletes a batch of batch_size items)
#    BGPQ.delete_mins(heap)
if priority_queue.size > 0:
    priority_queue, min_keys, min_values = BGPQ.delete_mins(priority_queue)
    # min_keys and min_values will have shape (pq_batch_size, ...)

    # Filter out padded items (keys will be jnp.inf for padding)
    valid_mask = jnp.isfinite(min_keys)
    actual_min_keys = min_keys[valid_mask]
    actual_min_values = jax.tree_util.tree_map(lambda x: x[valid_mask], min_values)

    print(f"BGPQ: Deleted {jnp.sum(valid_mask)} items.")
    if jnp.sum(valid_mask) > 0:
        print(f"BGPQ: Smallest key deleted: {actual_min_keys[0]}")
        # print(f"BGPQ: Corresponding value: {actual_min_values[0]}") # If you want to see the value
    print(f"BGPQ: Size after deletion: {priority_queue.size}")
else:
    print("BGPQ: Heap is empty, cannot delete.")
```

## Key `BGPQ` Details

*   **Batched Operations**: All operations (insert, delete_mins) are designed to work on batches of data of size `batch_size`.
*   **`BGPQ.build(total_size, batch_size, value_class, key_dtype=jnp.float16)`**:
    *   `total_size`: Desired maximum capacity. The actual `max_size` of the queue might be slightly larger to be an exact multiple of `batch_size` (calculated as `ceil(total_size / batch_size) * batch_size`).
    *   `batch_size`: The fixed size for all batch operations.
    *   `value_class`: The *class* of your custom `@xtructure_dataclass` used for storing values in the queue. This class must have a `.default()` method.
    *   `key_dtype`: Dtype for keys; defaults to `jnp.float16`.
*   **`BGPQ.make_batched(keys, values, batch_size)`**: (Static method)
    *   A crucial helper to prepare data for `BGPQ.insert()`. It takes a chunk of keys and corresponding values and pads them to the required `batch_size`.
    *   Keys are padded with `jnp.inf`.
    *   Values are padded using `value_class.default()` for the padding portion.
    *   Returns `batched_keys, batched_values`.
*   **`BGPQ.insert(heap, block_key, block_val)`**:
    *   Inserts a batch of keys and values. Inputs (`block_key`, `block_val`) *must* be pre-batched, typically using `BGPQ.make_batched()`.
    *   The function automatically counts the number of finite keys in `block_key` to determine how many items are being added.
*   **`BGPQ.delete_mins(heap)`**:
    *   Returns the modified queue, a batch of `batch_size` smallest keys, and their corresponding values.
    *   **Important**: If the queue contains fewer than `batch_size` items, the returned `min_keys` and `min_values` arrays will be padded (keys with `jnp.inf`, values with their defaults). You **must** use a filter like `valid_mask = jnp.isfinite(min_keys)` to identify and use only the actual (non-padded) items returned.
*   **Internal Structure**: The BGPQ maintains a min-heap structure. This heap is composed of multiple sorted blocks, each of size `batch_size`, allowing for efficient batched heap operations.
