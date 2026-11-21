# `HashTable` Usage

A bucketed double hash table optimized for JAX.

Define stored values by inheriting from `Xtructurable`.

```python
import jax
import jax.numpy as jnp
from xtructure import FieldDescriptor, HashTable, Xtructurable


class MyDataValue(Xtructurable):
    id: FieldDescriptor.scalar(dtype=jnp.uint32)
    position: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))
    flags: FieldDescriptor.tensor(dtype=jnp.bool_, shape=(4,))


# 1. Build the HashTable
table_capacity = 1000
hash_table = HashTable.build(MyDataValue, seed=123, capacity=table_capacity)

# 2. Prepare random batch data
num_items = 100
sample_data = MyDataValue.random(shape=(num_items,), key=jax.random.PRNGKey(0))

# 3. Parallel insert
filled = jnp.ones(num_items, dtype=jnp.bool_)
hash_table, inserted_mask, unique_mask, idxs = hash_table.parallel_insert(sample_data, filled)
print(f"Inserted {jnp.sum(inserted_mask)} new items ({jnp.sum(unique_mask)} unique in batch).")

# 4. Single-item lookup
idx, found = hash_table.lookup(sample_data[0])
if found:
    retrieved_item = hash_table[idx]
    print(f"Found item at flat index {idx.index}: id={retrieved_item.id}")

# 5. Parallel lookup with probe reuse before inserting another batch
candidates = MyDataValue.random(shape=(16,), key=jax.random.PRNGKey(1))
lookup_idxs, founds, probe = hash_table.lookup_parallel_with_probe(candidates)
hash_table, newly_inserted, _, _ = hash_table.parallel_insert(candidates, probe=probe)

# 6. Single-item insertion
single_item = MyDataValue.default().replace(
    id=jnp.uint32(999),
    position=jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
    flags=jnp.array([True, False, True, False], dtype=jnp.bool_),
)
hash_table, was_inserted, single_idx = hash_table.insert(single_item)
```

## Key `HashTable` Contracts

*   **Bucketed double hashing & over-provisioning**: `HashTable.build(dataclass, seed, capacity, bucket_size=8, hash_size_multiplier=2, max_probes=None)` packs `bucket_size` slots per bucket (default `8`) and rounds the internal bucket count up to a power of two based on `hash_size_multiplier * capacity / bucket_size`. Over-provisioning keeps open-addressed probe lengths short under GPU parallel insertion without resizing, and power-of-two sizing allows bitmask modulus reduction.
*   **Static config fields**: `HashTable` is a `@base_dataclass` with `static_fields=("seed", "capacity", "_capacity", "bucket_size", "max_probes")` so JIT compiles bucket shapes and probe bounds directly. Keep these fields as hashable Python `int` values, not JAX arrays.
*   **`inserted_mask` vs `unique_mask`**: In `parallel_insert(inputs, filled=None, unique_key=None, probe=None, *, assume_unique=False)`:
    *   `unique_mask` marks the single chosen representative of each distinct key among `filled` inputs in the batch.
    *   `inserted_mask` marks rows that were both `unique_mask` representatives and not already present in the table (actually written by this call).
    *   Pass `assume_unique=True` only when every row selected by `filled` is guaranteed unique to skip intra-batch deduplication.
*   **Single-hash-pass probe reuse**: `lookup_parallel_with_probe(inputs, filled=True)` returns `(idxs, founds, probe)` where `probe` (`HashTableProbe`) holds the uint32 hashing products computed during lookup. Passing `probe=probe` to `parallel_insert` for the exact same batch and table reuses that hash pass for bit-identical state without hashing twice; shape or dtype mismatches raise immediately rather than silently recomputing.
*   **Element retrieval**: Both `lookup` and `lookup_parallel` return a `HashIdx`; index directly with `hash_table[idx]`.
