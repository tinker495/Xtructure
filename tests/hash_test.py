import jax
import jax.numpy as jnp
import pytest

from Xtructure.hash import HashTable, hash_func_builder
from Xtructure.util import set_tree
from Xtructure.data import xtructure_data

import chex
@xtructure_data
class XtructureValue:
    a: chex.Array
    b: chex.Array

    @classmethod
    def default(cls, shape=()) -> "XtructureValue":
        a = jnp.full(shape, -1, dtype=jnp.uint8)
        b = jnp.full(shape + (1, 2), -1, dtype=jnp.uint32)
        return cls(a=a, b=b)
    
    @classmethod
    def random(cls, shape=(), key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key, 2)
        a = jax.random.randint(key1, shape, 0, 10, dtype=jnp.uint8)
        b = jax.random.randint(key2, shape + (1, 2), 0, 10, dtype=jnp.uint32)
        return cls(a=a, b=b)

@jax.jit
def is_equal(a, b):
    tree_equal = jax.tree_util.tree_map(lambda x, y: jnp.all(x == y), a, b)
    return jax.tree_util.tree_reduce(jnp.logical_and, tree_equal)

@pytest.fixture
def hash_func():
    return hash_func_builder(XtructureValue)


def test_hash_table_lookup(hash_func):
    count = 1000
    sample = XtructureValue.random((count,))
    table = HashTable.build(XtructureValue, 1, int(1e4))

    lookup = jax.jit(lambda table, sample: HashTable.lookup(table, hash_func, sample))
    idx, table_idx, found = jax.vmap(lookup, in_axes=(None, 0))(table, sample)

    assert idx.shape == (count,)
    assert table_idx.shape == (count,)
    assert found.shape == (count,)
    assert not jnp.any(found)  # Initially all should be not found


def test_hash_table_insert(hash_func):
    count = 1000
    batch = 4000
    table = HashTable.build(XtructureValue, 1, int(1e4))

    sample = XtructureValue.random((count,))

    lookup = jax.jit(lambda table, sample: HashTable.lookup(table, hash_func, sample))
    parallel_insert = jax.jit(
        lambda table, sample, filled: HashTable.parallel_insert(table, hash_func, sample, filled)
    )

    # Check initial state
    _, _, old_found = jax.vmap(lookup, in_axes=(None, 0))(table, sample)
    assert not jnp.any(old_found)

    # Insert states
    batched_sample, filled = HashTable.make_batched(XtructureValue, sample, batch)
    table, inserted, _, _, _ = parallel_insert(table, batched_sample, filled)

    # Verify insertion
    _, _, found = jax.vmap(lookup, in_axes=(None, 0))(table, sample)
    assert jnp.all(found)  # All states should be found after insertion
    assert jnp.mean(inserted) > 0  # Some states should have been inserted


def test_same_state_insert_at_batch(hash_func):
    batch = 5000
    table = HashTable.build(XtructureValue, 1, int(1e5))
    parallel_insert = jax.jit(
        lambda table, sample, filled: HashTable.parallel_insert(table, hash_func, sample, filled)
    )
    lookup = jax.jit(lambda table, sample: HashTable.lookup(table, hash_func, sample))

    num = 10
    counts = 0
    all_samples = []
    for i in range(num):
        key = jax.random.PRNGKey(i)
        samples = XtructureValue.random((batch,))
        cloned_sample_num = jax.random.randint(key, (), 1, batch // 2)
        cloned_sample_idx = jax.random.randint(key, (cloned_sample_num,), 0, batch)
        cloned_sample_idx = jnp.sort(cloned_sample_idx)
        new_clone_idx = jax.random.randint(key, (cloned_sample_num,), 0, batch)

        # Create deliberate duplicates within the batch
        samples = set_tree(samples, samples[cloned_sample_idx], new_clone_idx)
        h, bytesed = jax.vmap(hash_func, in_axes=(0, None))(samples, 0)
        unique_count = jnp.unique(bytesed, axis=0).shape[0]
        # after this, some states are duplicated
        all_samples.append(samples)

        batched_sample, filled = HashTable.make_batched(XtructureValue, samples, batch)
        table, updatable, unique, idxs, table_idxs = parallel_insert(table, batched_sample, filled)
        counts += jnp.sum(updatable)

        # Verify uniqueness tracking
        unique_idxs = jnp.unique(jnp.stack([idxs, table_idxs], axis=1), axis=0)
        assert (
            unique_idxs.shape[0] == unique_count
        ), f"unique_idxs.shape: {unique_idxs.shape}, unique_count: {unique_count}"
        assert unique_idxs.shape[0] == jnp.sum(unique), "Unique index mismatch"
        assert jnp.all(
            jnp.unique(unique_idxs, axis=0) == unique_idxs
        ), "Duplicate indices in unique set"

        # Verify inserted states exist in table
        _, _, found = jax.vmap(lookup, in_axes=(None, 0))(table, samples)
        assert jnp.all(found), (
            "Inserted states not found in table\n",
            f"unique_count: {unique_count}\n",
            f"unique_idxs.shape: {unique_idxs.shape}, unique: {jnp.sum(unique)}\n",
            f"found: {jnp.sum(found)}\n",
            f"not_found_idxs: {jnp.where(~found)[0]}\n",
            f"cloned_sample_idx: {cloned_sample_idx}\n",
        )

    # Final validation
    assert table.size == counts, f"Size mismatch: {table.size} vs {counts}"

    # Verify cross-batch duplicates
    for samples in all_samples:
        idx, table_idx, found = jax.vmap(lookup, in_axes=(None, 0))(table, samples)
        assert jnp.all(found), "Cross-batch state missing"
        contents = table.table[idx, table_idx]
        assert jnp.all(
            jax.vmap(is_equal)(contents, samples)
        ), "Inserted states not found in table"


def test_large_hash_table(hash_func):
    count = int(1e7)
    batch = int(1e4)
    table = HashTable.build(XtructureValue, 1, count)

    sample = XtructureValue.random((count,))
    hash, bytes = jax.vmap(hash_func, in_axes=(0, None))(sample, 0)
    unique_bytes = jnp.unique(bytes, axis=0, return_index=True)[1]
    unique_bytes_len = unique_bytes.shape[0]
    unique_hash = jnp.unique(hash, axis=0, return_index=True)[1]
    unique_hash_len = unique_hash.shape[0]
    print(f"unique_bytes_len: {unique_bytes_len}, unique_hash_len: {unique_hash_len}")

    parallel_insert = jax.jit(
        lambda table, sample, filled: HashTable.parallel_insert(table, hash_func, sample, filled)
    )
    lookup = jax.jit(lambda table, sample: HashTable.lookup(table, hash_func, sample))

    # Insert in batches
    inserted_count = 0
    for i in range(0, count, batch):
        batch_sample = sample[i : i + batch]
        table, inserted, _, _, _ = parallel_insert(
            table, batch_sample, jnp.ones(len(batch_sample), dtype=jnp.bool_)
        )
        inserted_count += jnp.sum(inserted)

    assert (
        inserted_count == unique_bytes_len
    ), f"inserted_count: {inserted_count}, unique_bytes_len: {unique_bytes_len}, unique_hash_len: {unique_hash_len}"

    # Verify all states can be found
    _, _, found = jax.vmap(lookup, in_axes=(None, 0))(table, sample)
    assert jnp.mean(found) == 1.0  # All states should be found