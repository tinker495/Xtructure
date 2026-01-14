import jax
import jax.numpy as jnp

from tests.testdata import HashValueAB, OddBytesValue47
from xtructure import HashTable


def test_hash_table_lookup():
    count = 1000
    sample = HashValueAB.random((count,))
    table: HashTable = HashTable.build(HashValueAB, 1, int(1e4))

    idx, found = table.lookup_parallel(sample)

    assert idx.shape.batch == (count,)
    assert found.shape == (count,)
    assert not jnp.any(found)  # Initially all should be not found


def test_hash_table_insert():
    count = 1000
    batch = 4000
    table: HashTable = HashTable.build(HashValueAB, 1, int(1e4))

    sample = HashValueAB.random((count,))

    # Check initial state
    _, old_found = table.lookup_parallel(sample)
    assert not jnp.any(old_found)

    # Insert states
    # Insert states
    current_count = sample.shape.batch[0] if sample.shape.batch else 1
    pad_amount = batch - current_count
    batched_sample = sample.pad((0, pad_amount))
    filled = jnp.zeros((batch,), dtype=jnp.bool_).at[:count].set(True)
    table, inserted, _, _ = table.parallel_insert(batched_sample, filled)

    # Verify insertion
    _, found = table.lookup_parallel(sample)
    assert jnp.all(found)  # All states should be found after insertion
    assert jnp.mean(inserted) > 0  # Some states should have been inserted


def test_same_state_insert_at_batch():
    batch = 5000
    table: HashTable = HashTable.build(HashValueAB, 1, int(1e5))

    num = 10
    counts = 0
    all_samples = []
    for i in range(num):
        key = jax.random.PRNGKey(i)
        samples = HashValueAB.random((batch,))
        cloned_sample_num = jax.random.randint(key, (), 1, batch // 2)
        cloned_sample_idx = jax.random.randint(key, (cloned_sample_num,), 0, batch)
        cloned_sample_idx = jnp.sort(cloned_sample_idx)
        new_clone_idx = jax.random.randint(key, (cloned_sample_num,), 0, batch)

        # Create deliberate duplicates within the batch
        samples = samples.at[new_clone_idx].set(samples[cloned_sample_idx])
        bytesed = jax.vmap(lambda x: x.bytes)(samples)
        unique_count = jnp.unique(bytesed, axis=0).shape[0]
        # after this, some states are duplicated
        all_samples.append(samples)

        table, updatable, unique, idxs = table.parallel_insert(samples)
        counts += jnp.sum(updatable)

        # Verify uniqueness tracking
        bytesed = jax.vmap(lambda x: x.bytes)(idxs)
        unique_idxs = jnp.unique(bytesed, axis=0)
        assert (
            unique_idxs.shape[0] == unique_count
        ), f"unique_idxs.shape: {unique_idxs.shape}, unique_count: {unique_count}"
        assert unique_idxs.shape[0] == jnp.sum(unique), "Unique index mismatch"
        assert jnp.all(
            jnp.unique(unique_idxs, axis=0) == unique_idxs
        ), "Duplicate indices in unique set"

        # Verify inserted states exist in table
        _, found = table.lookup_parallel(samples)
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
        idx, found = table.lookup_parallel(samples)
        assert jnp.all(found), "Cross-batch state missing"
        contents = table[idx]
        assert jnp.all(
            jax.vmap(lambda x, y: x == y)(contents, samples)
        ), "Inserted states not found in table"


def test_large_hash_table():
    count = int(1e7)
    batch = int(1e4)
    table: HashTable = HashTable.build(HashValueAB, 1, count)

    sample = HashValueAB.random((count,))
    hash, bytes = jax.vmap(lambda x: x.hash_with_uint32ed(0))(sample)
    unique_bytes = jnp.unique(bytes, axis=0, return_index=True)[1]
    unique_bytes_len = unique_bytes.shape[0]
    unique_hash = jnp.unique(hash, axis=0, return_index=True)[1]
    unique_hash_len = unique_hash.shape[0]
    print(f"unique_bytes_len: {unique_bytes_len}, unique_hash_len: {unique_hash_len}")

    # Insert in batches
    inserted_count = 0
    for i in range(0, count, batch):
        batch_sample = sample[i : i + batch]
        table, inserted, _, _ = table.parallel_insert(
            batch_sample, jnp.ones(len(batch_sample), dtype=jnp.bool_)
        )
        inserted_count += jnp.sum(inserted)

    assert (
        inserted_count == unique_bytes_len
    ), f"inserted_count: {inserted_count}, unique_bytes_len: {unique_bytes_len}, unique_hash_len: {unique_hash_len}"

    # Verify all states can be found
    _, found = table.lookup_parallel(sample)
    assert jnp.mean(found) == 1.0  # All states should be found


def test_default_value_insertion():
    """Test that default values can be inserted and found correctly."""
    table: HashTable = HashTable.build(HashValueAB, 1, int(1e4))

    # Create a default value instance
    default_value = HashValueAB.default()

    # Insert the default value
    table, inserted, idx = table.insert(default_value)

    # Assert that the insertion was successful
    assert inserted, "Default value should be inserted successfully"

    # Look up the inserted default value
    lookup_idx, found = table.lookup(default_value)

    # Assert that the lookup is successful
    assert found, "Default value should be found after insertion"

    # Verify the retrieved value matches the inserted value
    retrieved_value = table[lookup_idx]
    assert (
        retrieved_value == default_value
    ), "Retrieved value should match the inserted default value"

    # Test with parallel operations as well - use a fresh table
    fresh_table: HashTable = HashTable.build(HashValueAB, 1, int(1e4))
    default_batch = HashValueAB.default((5,))  # Create a batch of default values

    # Insert batch of default values
    fresh_table, updatable, unique_filled, batch_idx = fresh_table.parallel_insert(default_batch)

    # At least one should be updatable (the first unique default value)
    assert jnp.any(updatable), "At least one default value should be updatable in batch"

    # Look up the batch
    batch_lookup_idx, batch_found = fresh_table.lookup_parallel(default_batch)

    # All should be found
    assert jnp.all(batch_found), "All default values should be found after batch insertion"


def test_uint32ed_padding_for_non_multiple_of_four_bytes():
    payload = jnp.asarray(jnp.arange(47, dtype=jnp.uint8).block_until_ready())
    sample = OddBytesValue47(payload=payload)

    uint32ed = sample.uint32ed

    expected_words = (47 + 3) // 4
    assert uint32ed.shape == (expected_words,)
    assert uint32ed.dtype == jnp.uint32
