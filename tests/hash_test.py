import jax
import jax.numpy as jnp

from xtructure import FieldDescriptor, HashTable, xtructure_dataclass


@xtructure_dataclass
class XtructureValue:
    a: FieldDescriptor(jnp.uint8)  # type: ignore
    b: FieldDescriptor(jnp.uint32, (1, 2))  # type: ignore


def test_hash_table_lookup():
    count = 1000
    sample = XtructureValue.random((count,))
    table = HashTable.build(XtructureValue, 1, int(1e4))

    lookup = jax.jit(lambda table, sample: HashTable.lookup(table, sample))
    idx, table_idx, found = jax.vmap(lookup, in_axes=(None, 0))(table, sample)

    assert idx.shape == (count,)
    assert table_idx.shape == (count,)
    assert found.shape == (count,)
    assert not jnp.any(found)  # Initially all should be not found


def test_hash_table_insert():
    count = 1000
    batch = 4000
    table = HashTable.build(XtructureValue, 1, int(1e4))

    sample = XtructureValue.random((count,))

    lookup = jax.jit(lambda table, sample: HashTable.lookup(table, sample))
    parallel_insert = jax.jit(
        lambda table, sample, filled: HashTable.parallel_insert(table, sample, filled)
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


def test_same_state_insert_at_batch():
    batch = 5000
    table = HashTable.build(XtructureValue, 1, int(1e5))
    parallel_insert = jax.jit(
        lambda table, sample, filled: HashTable.parallel_insert(table, sample, filled)
    )
    lookup = jax.jit(lambda table, sample: HashTable.lookup(table, sample))

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
        samples = samples.at[new_clone_idx].set(samples[cloned_sample_idx])
        h, bytesed = jax.vmap(lambda x: x.hash(0))(samples)
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
            jax.vmap(lambda x, y: x == y)(contents, samples)
        ), "Inserted states not found in table"


def test_large_hash_table():
    count = int(1e7)
    batch = int(1e4)
    table = HashTable.build(XtructureValue, 1, count)

    sample = XtructureValue.random((count,))
    hash, bytes = jax.vmap(lambda x: x.hash(0))(sample)
    unique_bytes = jnp.unique(bytes, axis=0, return_index=True)[1]
    unique_bytes_len = unique_bytes.shape[0]
    unique_hash = jnp.unique(hash, axis=0, return_index=True)[1]
    unique_hash_len = unique_hash.shape[0]
    print(f"unique_bytes_len: {unique_bytes_len}, unique_hash_len: {unique_hash_len}")

    parallel_insert = jax.jit(
        lambda table, sample, filled: HashTable.parallel_insert(table, sample, filled)
    )
    lookup = jax.jit(lambda table, sample: HashTable.lookup(table, sample))

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
