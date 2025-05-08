# Xtructure

A Python package providing JAX-optimized data structures, including a batched priority queue and a cuckoo hash table.

## Features

- Batched GPU Priority Queue (`BGPQ`)
- Cuckoo Hash Table (`HashTable`)
- Optimized for JAX

## Installation

```bash
# Coming soon - after uploading to PyPI
# pip install Xtructure
```

## Usage

```python
import jax
import jax.numpy as jnp
from Xtructure import BGPQ, HashTable, xtructure_data, hash_func_builder # Assuming __init__.py is set up
import chex # For type hinting

# Example usage (conceptual)
# See individual modules for detailed API and examples

# BGPQ Example
@xtructure_data
class MyHeapValue:
    a: chex.Array
    b: chex.Array

    @classmethod
    def default(cls, shape=()) -> "MyHeapValue":
        return cls(
            a=jnp.full(shape, jnp.inf, dtype=jnp.uint8),
            b=jnp.full(shape + (1, 2), jnp.inf, dtype=jnp.uint32),
        )

    @classmethod
    def random(cls, shape=(), key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key, 2)
        return cls(
            a=jax.random.randint(key1, shape, 0, 10, dtype=jnp.uint8),
            b=jax.random.randint(key2, shape + (1, 2), 0, 10, dtype=jnp.uint32),
        )

# A simple key generation function for demonstration
# In practice, this would be more sophisticated, like in heap_test.py
def simple_key_gen(value: MyHeapValue):
    # Combine sums of elements, normalized - this is a placeholder
    key_val = jnp.sum(value.a.astype(jnp.float32)) + jnp.sum(value.b.astype(jnp.float32))
    return key_val / (value.a.size + value.b.size)

heap = BGPQ.build(total_size=1000, batch_size=10, value_type=MyHeapValue)

# Generate some random values and keys
key = jax.random.PRNGKey(0)
random_values = MyHeapValue.random(shape=(20,), key=key) # 2 batches of 10
keys_to_insert = jax.vmap(simple_key_gen)(random_values)

# Insert into BGPQ
# Note: BGPQ.insert expects batched keys and values.
# If your data isn't batched, or if the last batch might be partial,
# use BGPQ.make_batched to prepare it.
batched_keys, batched_values = BGPQ.make_batched(keys_to_insert, random_values, batch_size=10)
heap = BGPQ.insert(heap, batched_keys, batched_values)

# Delete minimums
heap, min_keys, min_values = BGPQ.delete_mins(heap)


# HashTable Example
@xtructure_data
class MyHashableValue:
    data: chex.Array

    @classmethod
    def default(cls, shape=()) -> "MyHashableValue":
        if isinstance(shape, int):
            shape = (shape,)
        return MyHashableValue(data=jnp.zeros(shape + (4,4), dtype=jnp.int32))

    @classmethod
    def random(cls, shape=(), key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        return MyHashableValue(data=jax.random.randint(key, shape + (4,4), 0, 100, dtype=jnp.int32))

custom_hash_func = hash_func_builder(MyHashableValue)
table = HashTable.build(value_type=MyHashableValue, seed=1, capacity=1000)

# Create a sample to insert
sample_value = MyHashableValue.random((10,)) # 10 items

# HashTable.parallel_insert expects batched samples.
# Use HashTable.make_batched if your data is not already in the correct batch format.
# For simplicity, let's assume batch_size matches our sample count for this example.
batch_size = 10
batched_sample, filled_mask = HashTable.make_batched(MyHashableValue, sample_value, batch_size)

# Insert into HashTable
table, inserted_mask, unique_mask, idxs, table_idxs = HashTable.parallel_insert(
    table, custom_hash_func, batched_sample, filled_mask
)

# Lookup values
# HashTable.lookup also expects unbatched samples for the query
idx, table_idx, found_mask = jax.vmap(
    lambda t, s: HashTable.lookup(t, custom_hash_func, s)
)(table, sample_value)

print(f"Found in table: {jnp.sum(found_mask)}/{len(sample_value)}")
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details (you'll need to create this file).
