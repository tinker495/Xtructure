import jax
import jax.numpy as jnp
import pytest

from tests.testdata import HashValueAB, OddBytesValue47
from xtructure import HashTable
from xtructure.core.xtructure_decorators.pytree_adapters import hash as hash_adapter
from xtructure.hashtable.hash_utils import get_new_idx_byterized_batched


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
    batched_sample = sample.padding_as_batch((batch,))
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


def test_parallel_insert_probes_bucket_overflow_without_losing_values():
    """A batch wider than an initial bucket must spill through double hashing."""
    count = 24
    bucket_size = 2
    table: HashTable = HashTable.build(
        HashValueAB,
        seed=17,
        capacity=32,
        bucket_size=bucket_size,
        hash_size_multiplier=1,
    )
    samples = HashValueAB.random((count,), key=jax.random.PRNGKey(23))
    probe = get_new_idx_byterized_batched(samples, table._capacity, table.seed)
    initial_counts = jnp.bincount(probe.index, length=table._capacity)

    assert int(jnp.max(initial_counts)) > bucket_size

    table, inserted, unique, hash_idx = table.parallel_insert(samples)
    lookup_idx, found = table.lookup_parallel(samples)

    assert int(table.size) == count
    assert bool(jnp.all(inserted))
    assert bool(jnp.all(unique))
    assert bool(jnp.all(found))
    assert int(jnp.unique(hash_idx.index).shape[0]) == count
    assert bool(jnp.all(lookup_idx.index == hash_idx.index))


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


def test_hash_byte_encoding_rejects_unknown_dtype_kind():
    with pytest.raises(TypeError, match="DType Kind"):
        hash_adapter.tree_to_uint32(jnp.asarray([1 + 2j], dtype=jnp.complex64))


def test_hash_decorator_attaches_module_level_functions():
    sample = OddBytesValue47(payload=jnp.asarray(jnp.arange(47, dtype=jnp.uint8)))

    assert OddBytesValue47.bytes.fget is hash_adapter.tree_to_bytes
    assert OddBytesValue47.uint32ed.fget is hash_adapter._uint32ed
    assert OddBytesValue47.hash is hash_adapter._h
    assert OddBytesValue47.hash_with_uint32ed is hash_adapter._h_with_uint32ed

    assert jnp.array_equal(sample.bytes, hash_adapter.tree_to_bytes(sample))
    assert jnp.array_equal(sample.uint32ed, hash_adapter.tree_to_uint32(sample))
    assert sample.hash(0) == hash_adapter._h(sample, 0)


def test_uint32ed_is_instance_layout_aware():
    expected_words = (47 + 3) // 4  # 12

    single = OddBytesValue47(payload=jnp.arange(47, dtype=jnp.uint8))
    assert single.uint32ed.shape == (expected_words,)

    batched = OddBytesValue47.default((3,))
    assert batched.uint32ed.shape == (3, expected_words)

    vmap_result = jax.vmap(lambda x: x.uint32ed)(batched)
    assert jnp.array_equal(batched.uint32ed, vmap_result)


def test_hash_is_instance_layout_aware():
    single = OddBytesValue47(payload=jnp.arange(47, dtype=jnp.uint8))
    batched = OddBytesValue47.default((3,))

    assert jnp.ndim(single.hash(0)) == 0

    assert batched.hash(0).shape == (3,)

    vmap_result = jax.vmap(lambda x: x.hash(0))(batched)
    assert jnp.array_equal(batched.hash(0), vmap_result)


def test_uint32ed_unstructured_raises():
    # TODO: Construct an UNSTRUCTURED instance once the construction path is known.
    pytest.skip("UNSTRUCTURED construction path TBD")


def _assert_tables_bit_identical(a: HashTable, b: HashTable) -> None:
    assert int(a.size) == int(b.size)
    assert bool(jnp.all(a.fingerprints == b.fingerprints))
    assert bool(jnp.all(a.bucket_fill_levels == b.bucket_fill_levels))
    assert bool(jnp.all(jax.vmap(lambda x, y: x == y)(a.table, b.table)))


def test_lookup_parallel_with_probe_matches_lookup_parallel():
    """The probe-returning lookup must return the same (idx, found) as the plain one."""
    batch = 2048
    table: HashTable = HashTable.build(HashValueAB, 7, int(1e5))
    samples = HashValueAB.random((batch,))
    filled = jax.random.bernoulli(jax.random.PRNGKey(0), 0.8, (batch,))

    table, *_ = table.parallel_insert(samples, filled)

    idx_plain, found_plain = table.lookup_parallel(samples, filled)
    idx_probe, found_probe, probe = table.lookup_parallel_with_probe(samples, filled)

    assert bool(jnp.all(idx_plain.index == idx_probe.index))
    assert bool(jnp.all(found_plain == found_probe))
    # Probe fields are exactly the shared hash-pass products.
    assert probe.index.shape == (batch,)
    assert probe.step.shape == (batch,)
    assert probe.fingerprint.shape == (batch,)
    assert probe.uint32ed.ndim == 2 and probe.uint32ed.shape[0] == batch


def test_parallel_insert_with_probe_is_bit_identical():
    """parallel_insert(probe=...) must yield bit-identical table state and idx vs recompute.

    Runs several batches with deliberate intra-batch duplicates on two independently
    built tables: one always recomputes the hash pass, the other threads the probe
    produced by lookup_parallel_with_probe. The resulting tables, returned masks and
    HashIdx must match exactly (the whole point of the no-rehash optimisation is that
    reusing the probe changes nothing observable).
    """
    batch = 4000
    table_recompute: HashTable = HashTable.build(HashValueAB, 3, int(1e5))
    table_probe: HashTable = HashTable.build(HashValueAB, 3, int(1e5))

    for i in range(6):
        key = jax.random.PRNGKey(i)
        samples = HashValueAB.random((batch,))
        # Inject intra-batch duplicates so dedup logic is exercised.
        dup_src = jax.random.randint(key, (batch // 3,), 0, batch)
        dup_dst = jax.random.randint(jax.random.fold_in(key, 1), (batch // 3,), 0, batch)
        samples = samples.at[dup_dst].set(samples[dup_src])
        filled = jax.random.bernoulli(jax.random.fold_in(key, 2), 0.9, (batch,))

        # Recompute path: plain lookup (discards probe) then plain insert.
        _, _ = table_recompute.lookup_parallel(samples, filled)
        table_recompute, ins_r, uniq_r, hidx_r = table_recompute.parallel_insert(samples, filled)

        # Probe path: probe-returning lookup then insert reusing the probe.
        _, _, probe = table_probe.lookup_parallel_with_probe(samples, filled)
        table_probe, ins_p, uniq_p, hidx_p = table_probe.parallel_insert(
            samples, filled, None, probe
        )

        assert bool(jnp.all(ins_r == ins_p)), f"inserted mask diverged at batch {i}"
        assert bool(jnp.all(uniq_r == uniq_p)), f"unique mask diverged at batch {i}"
        assert bool(jnp.all(hidx_r.index == hidx_p.index)), f"HashIdx diverged at batch {i}"
        _assert_tables_bit_identical(table_recompute, table_probe)


def test_parallel_insert_with_probe_respects_unique_key():
    """The probe path must stay bit-identical when a unique_key tie-breaker is supplied."""
    batch = 3000
    table_recompute: HashTable = HashTable.build(HashValueAB, 11, int(1e5))
    table_probe: HashTable = HashTable.build(HashValueAB, 11, int(1e5))

    key = jax.random.PRNGKey(99)
    samples = HashValueAB.random((batch,))
    dup_src = jax.random.randint(key, (batch // 2,), 0, batch)
    dup_dst = jax.random.randint(jax.random.fold_in(key, 1), (batch // 2,), 0, batch)
    samples = samples.at[dup_dst].set(samples[dup_src])
    filled = jnp.ones((batch,), dtype=jnp.bool_)
    unique_key = jax.random.uniform(jax.random.fold_in(key, 3), (batch,))

    table_recompute, ins_r, uniq_r, hidx_r = table_recompute.parallel_insert(
        samples, filled, unique_key
    )
    _, _, probe = table_probe.lookup_parallel_with_probe(samples, filled)
    table_probe, ins_p, uniq_p, hidx_p = table_probe.parallel_insert(
        samples, filled, unique_key, probe
    )

    assert bool(jnp.all(ins_r == ins_p))
    assert bool(jnp.all(uniq_r == uniq_p))
    assert bool(jnp.all(hidx_r.index == hidx_p.index))
    _assert_tables_bit_identical(table_recompute, table_probe)


def test_parallel_insert_probe_shape_mismatch_raises():
    """A probe whose batch length disagrees with the states must fail fast, not recompute."""
    from xtructure import HashTableProbe

    batch = 512
    table: HashTable = HashTable.build(HashValueAB, 1, int(1e4))
    samples = HashValueAB.random((batch,))
    filled = jnp.ones((batch,), dtype=jnp.bool_)
    _, _, probe = table.lookup_parallel_with_probe(samples, filled)

    truncated = HashTableProbe(
        index=probe.index[: batch // 2],
        step=probe.step,
        uint32ed=probe.uint32ed,
        fingerprint=probe.fingerprint,
    )
    with pytest.raises(ValueError):
        table.parallel_insert(samples, filled, None, truncated)
