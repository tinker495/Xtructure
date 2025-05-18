# `HashTable` Usage

A Cuckoo hash table optimized for JAX.

```python
import jax
import jax.numpy as jnp
from xtructure import HashTable, xtructure_dataclass, FieldDescriptor


# Define a data structure (as an example from core_concepts.md)
@xtructure_dataclass
class MyDataValue:
    id: FieldDescriptor[jnp.uint32]
    position: FieldDescriptor[jnp.float32, (3,)]
    flags: FieldDescriptor[jnp.bool_, (4,)]


# 1. Build the HashTable
#    HashTable.build(pytree_def_type, initial_hash_seed, capacity)
table_capacity = 1000
hash_table = HashTable.build(MyDataValue, 123, table_capacity)
# Note: MyDataValue (the class itself) is passed, not an instance, for build.

# 3. Prepare data to insert
#    Let's create some random data.
num_items_to_insert = 100
key = jax.random.PRNGKey(0)
sample_data = MyDataValue.random(shape=(num_items_to_insert,), key=key)

# 4. Insert data
#    HashTable.parallel_insert(table, samples, filled_mask)
#    'filled_mask' indicates which items in 'sample_data' are valid.
filled_mask = jnp.ones(num_items_to_insert, dtype=jnp.bool_)
hash_table, inserted_mask, unique_mask, idxs, table_idxs = HashTable.parallel_insert(
    hash_table, sample_data, filled_mask
)

print(f"HashTable: Inserted {jnp.sum(inserted_mask)} items.")
print(f"HashTable: Unique items inserted: {jnp.sum(unique_mask)}")  # Number of items that were not already present
print(f"HashTable size: {hash_table.size}")

# inserted_mask: boolean array, true if the item at the corresponding input index was successfully inserted.
# unique_mask: boolean array, true if the inserted item was unique (not a duplicate).
# idxs: primary indices in the hash table where items were stored.
# table_idxs: cuckoo table indices (0 to CUCKOO_TABLE_N-1) used for each stored item.

# 5. Lookup data
#    HashTable.lookup(table, item_to_lookup)
item_to_check = sample_data[0]  # Let's check the first item we inserted
idx, table_idx, found = HashTable.lookup(hash_table, item_to_check)

if found:
    retrieved_item = hash_table.table[idx, table_idx]  # Accessing the item from the internal table
    print(f"HashTable: Item found at primary index {idx}, cuckoo_index {table_idx}.")
    # You can then compare retrieved_item with item_to_check
else:
    print("HashTable: Item not found.")

# (Optional) Batching data for insertion if your data isn't already batched appropriately:
# batch_size_for_insert = 50 # Example internal batch size if HashTable has one
# batched_sample_data, filled_mask_for_batched = HashTable.make_batched(
#     MyDataValue, # The class, not an instance
#     sample_data,
#     batch_size_for_insert
# )
# Then use batched_sample_data and filled_mask_for_batched in parallel_insert.
# Note: The `parallel_insert` in the README example did not require pre-batching with `HashTable.make_batched`.
# The provided `hash.py` also seems to handle arbitrary input sizes for `parallel_insert` with a `filled` mask.
# `HashTable.make_batched` is available if manual batch control is needed.
```

## Key `HashTable` Details

*   **Cuckoo Hashing**: Uses `CUCKOO_TABLE_N` (an internal constant, typically small e.g. 2-4) hash functions/slots per primary index to resolve collisions. This means an item can be stored in one of `N` locations.
*   **`HashTable.build(dataclass, seed, capacity)`**:
    *   `dataclass`: The *class* of your custom data structure (e.g., `MyDataValue`). An instance of this class (e.g., `MyDataValue.default()`) is used internally to define the table structure.
    *   `seed`: Integer seed for hashing.
    *   `capacity`: Desired user capacity. The internal capacity (`_capacity`) will be larger to accommodate Cuckoo hashing (specifically, `int(HASH_SIZE_MULTIPLIER * capacity / CUCKOO_TABLE_N)`).
*   **`HashTable.parallel_insert(table, hash_func, inputs, filled_mask)`**:
    *   `inputs`: A PyTree (or batch of PyTrees) of items to insert.
    *   `filled_mask`: A boolean JAX array indicating which entries in `inputs` are valid.
    *   Returns the updated table, `inserted_mask` (boolean array for successful insertions for each input), `unique_mask` (boolean array, true if the item was new and not a duplicate), `idxs` (main table indices where items were stored), and `table_idxs` (Cuckoo slot indices used).
*   **`HashTable.lookup(table, hash_func, item_to_lookup)`**:
    *   Returns `idx` (main table index), `table_idx` (Cuckoo slot index), and `found` (boolean).
    *   If `found` is true, the item can be retrieved from `table.table[idx, table_idx]`.
*   **`HashTable.make_batched(dataclass, inputs, batch_size)`**: (Static method)
    *   A helper to reshape and pad input data into fixed-size batches if needed for specific batch processing workflows, though `parallel_insert` itself can handle variable-sized inputs with a `filled_mask`.
    *   `dataclass`: The class of your custom data structure.
    *   Returns `batched_pytree, filled_mask_for_batched`.
