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
#    HashTable.build(dataclass, seed, capacity)
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
hash_table, inserted_mask, unique_mask, idxs = HashTable.parallel_insert(hash_table, sample_data, filled_mask)

print(f"HashTable: Inserted {jnp.sum(inserted_mask)} items.")
print(f"HashTable: Unique items inserted: {jnp.sum(unique_mask)}")  # Number of items that were not already present
print(f"HashTable size: {hash_table.size}")

# inserted_mask: boolean array, true if the item at the corresponding input index was successfully inserted.
# unique_mask: boolean array, true if the inserted item was unique (not a duplicate).
# idxs: HashIdx object containing indices in the hash table where items were stored.

# 5. Lookup data
#    HashTable.lookup(table, item_to_lookup)
item_to_check = sample_data[0]  # Let's check the first item we inserted
idx, found = HashTable.lookup(hash_table, item_to_check)

if found:
    retrieved_item = hash_table.table[idx.index]  # Accessing the item from the internal table
    print(f"HashTable: Item found at index {idx.index}.")
    # You can then compare retrieved_item with item_to_check
else:
    print("HashTable: Item not found.")

# 6. Parallel lookup (for multiple items)
#    HashTable.lookup_parallel(table, items_to_lookup)
items_to_lookup = sample_data[:5]  # Look up first 5 items
idxs, founds = HashTable.lookup_parallel(hash_table, items_to_lookup)
print(f"HashTable: Found {jnp.sum(founds)} out of {len(items_to_lookup)} items in parallel lookup.")

# 7. Single item insertion
#    HashTable.insert(table, item_to_insert)
single_item = MyDataValue.default()
single_item = single_item.replace(
    id=jnp.array(999), position=jnp.array([1.0, 2.0, 3.0]), flags=jnp.array([True, False, True, False])
)
hash_table, was_inserted, idx = HashTable.insert(hash_table, single_item)
print(f"HashTable: Single item inserted? {was_inserted}")
```

## Key `HashTable` Details

*   **Cuckoo Hashing**: Uses `CUCKOO_TABLE_N` (an internal constant, typically small e.g. 2-4) hash functions/slots per primary index to resolve collisions. This means an item can be stored in one of `N` locations.
*   **`HashTable.build(dataclass, seed, capacity)`**:
    *   `dataclass`: The *class* of your custom data structure (e.g., `MyDataValue`). An instance of this class (e.g., `MyDataValue.default()`) is used internally to define the table structure.
    *   `seed`: Integer seed for hashing.
    *   `capacity`: Desired user capacity. The internal capacity (`_capacity`) will be larger to accommodate Cuckoo hashing (specifically, `int(HASH_SIZE_MULTIPLIER * capacity / CUCKOO_TABLE_N)`).
*   **`HashTable.parallel_insert(table, inputs, filled_mask=None)`**:
    *   `inputs`: A PyTree (or batch of PyTrees) of items to insert.
    *   `filled_mask`: A boolean JAX array indicating which entries in `inputs` are valid. If `None`, all inputs are considered valid.
    *   Returns a tuple of:
        *   `updated_table`: The updated HashTable instance
        *   `inserted_mask`: Boolean array for successful insertions for each input
        *   `unique_mask`: Boolean array, true if the item was new and not a duplicate
        *   `idxs`: A `HashIdx` object containing `.index` for where items were stored
*   **`HashTable.lookup(table, item_to_lookup)`**:
    *   Returns `idx` (a `HashIdx` object with `.index`) and `found` (boolean).
    *   If `found` is true, the item can be retrieved from `table.table[idx.index]`.
*   **`HashTable.lookup_parallel(table, items_to_lookup)`**:
    *   Performs parallel lookup for multiple items.
    *   `items_to_lookup`: A batch of items to look up.
    *   Returns `idxs` (a `HashIdx` object with `.index` for each item) and `founds` (boolean array indicating which items were found).
*   **`HashTable.insert(table, item_to_insert)`**:
    *   Inserts a single item into the hash table.
    *   `item_to_insert`: The item to insert.
    *   Returns a tuple of:
        *   `updated_table`: The updated HashTable instance
        *   `was_inserted`: Boolean indicating if the item was actually inserted (not already present)
        *   `idx`: A `HashIdx` object with `.index` for where the item was stored
