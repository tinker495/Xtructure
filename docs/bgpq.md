# `BGPQ` (Batched GPU Priority Queue) Usage

A min-priority queue optimized for fixed-batch operations on GPUs.

Define stored values by inheriting from `Xtructurable`.

```python
import jax
import jax.numpy as jnp
from xtructure import BGPQ, FieldDescriptor, Xtructurable


class MyHeapItem(Xtructurable):
    task_id: FieldDescriptor.scalar(dtype=jnp.int32)
    payload: FieldDescriptor.tensor(dtype=jnp.float32, shape=(2, 2))


pq_total_size = 2000
pq_batch_size = 64
priority_queue = BGPQ.build(pq_total_size, pq_batch_size, MyHeapItem)

num_items = 150
prng_key = jax.random.PRNGKey(10)
keys_for_pq = jax.random.uniform(prng_key, (num_items,)).astype(jnp.float16)
prng_key, subkey = jax.random.split(prng_key)
values_for_pq = MyHeapItem.random(shape=(num_items,), key=subkey)

# Insert in fixed batch_size chunks using make_batched to pad the final partial chunk
for start_idx in range(0, num_items, pq_batch_size):
    end_idx = min(start_idx + pq_batch_size, num_items)
    keys_to_insert, values_to_insert = BGPQ.make_batched(
        keys_for_pq[start_idx:end_idx],
        values_for_pq[start_idx:end_idx],
        pq_batch_size,
    )
    priority_queue = priority_queue.insert(keys_to_insert, values_to_insert)

# Delete a batch of minimums and filter out jnp.inf padding sentinels
if priority_queue.size > 0:
    priority_queue, min_keys, min_values = priority_queue.delete_mins()
    valid_mask = jnp.isfinite(min_keys)
    actual_min_keys = min_keys[valid_mask]
    actual_min_values = min_values[valid_mask]
    print(f"Deleted {jnp.sum(valid_mask)} items; smallest key: {actual_min_keys[0]}")
```

## Key `BGPQ` Contracts

*   **Static config fields**: `BGPQ` is a `@base_dataclass` with `static_fields=("max_size", "branch_size", "batch_size")`, so JIT treats these sizes as compile-time constants. Keep them as hashable Python `int` values, not JAX arrays.
*   **Capacity rounding**: `BGPQ.build(total_size, batch_size, value_class, key_dtype=jnp.float16)` rounds `max_size` up to `ceil(total_size / batch_size) * batch_size` so every heap node holds a complete `batch_size` block.
*   **Sentinel padding contract**:
    *   `BGPQ.make_batched(keys, values, batch_size)` and `heap.make_batched_like(keys, values)` pad short chunks to `batch_size` using `jnp.inf` keys and `value_class.default()` values; `insert` counts finite keys to track live items.
    *   `delete_mins()` always returns `(batch_size,)` keys and values; when fewer than `batch_size` items remain, unused slots contain `jnp.inf` keys and `value_class.default()` values. Callers **must** filter outputs with `jnp.isfinite(min_keys)`.
*   **Sorted-block min-heap structure**: `BGPQ` maintains a min-heap of sorted blocks of length `batch_size` (plus a `batch_size - 1` staging buffer), enabling parallel merge-split heapification without host-side per-element control flow.
