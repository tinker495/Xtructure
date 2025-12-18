import jax
import jax.numpy as jnp
import numpy as np

from xtructure import FieldDescriptor, HashTable, xtructure_dataclass


@xtructure_dataclass
class XtructureValue:
    a: FieldDescriptor(jnp.uint8)  # type: ignore
    b: FieldDescriptor(jnp.uint32, (1, 2))  # type: ignore


@xtructure_dataclass
class OddBytesValue:
    payload: FieldDescriptor(jnp.uint8, (47,))  # type: ignore


def test_hash_table_lookup():
    count = 1000
    sample = XtructureValue.random((count,))
    table: HashTable = HashTable.build(XtructureValue, 1, int(1e4))

    idx, found = table.lookup_parallel(sample)

    assert idx.shape.batch == (count,)
    assert found.shape == (count,)
    assert not jnp.any(found)  # Initially all should be not found


def test_hash_table_insert():
    count = 1000
    batch = 4000
    table: HashTable = HashTable.build(XtructureValue, 1, int(1e4))

    sample = XtructureValue.random((count,))

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
    table: HashTable = HashTable.build(XtructureValue, 1, int(1e5))

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
    table: HashTable = HashTable.build(XtructureValue, 1, count)

    sample = XtructureValue.random((count,))
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
    table: HashTable = HashTable.build(XtructureValue, 1, int(1e4))

    # Create a default value instance
    default_value = XtructureValue.default()

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
    fresh_table: HashTable = HashTable.build(XtructureValue, 1, int(1e4))
    default_batch = XtructureValue.default((5,))  # Create a batch of default values

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
    sample = OddBytesValue(payload=payload)

    uint32ed = sample.uint32ed

    expected_words = (47 + 3) // 4
    assert uint32ed.shape == (expected_words,)
    assert uint32ed.dtype == jnp.uint32


def test_hash_table_multi_round_bulk_insert_and_integrity():
    """Stress test: multiple rounds of bulk insert, then bulk lookup + content integrity checks.

    This is stricter than other tests:
    - Inserts in multiple rounds (table grows over time)
    - Includes deliberate duplicates both within-round and across-round
    - Verifies: found==True, retrieved values equal inserted values, and size matches unique count
    """

    def _concat_xtruct(values_list):
        # Concatenate a list of Xtructurable batches along axis=0
        return jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *values_list)

    # Keep this large enough to stress probing + repeated inserts, but small enough for CI.
    rounds = 6
    batch = 20000
    cap = int(3e5)

    table: HashTable = HashTable.build(XtructureValue, 1, cap)

    total_inserted = 0
    replay_batches = []
    # Cache some real previously-inserted values to inject as cross-round duplicates.
    cached_prev_values = None

    for r in range(rounds):
        key = jax.random.PRNGKey(1234 + r)

        # Fresh random batch
        samples = XtructureValue.random((batch,))

        # Create within-batch duplicates: copy some items onto other indices
        num_dup = jax.random.randint(key, (), batch // 10, batch // 3)
        src = jax.random.randint(key, (num_dup,), 0, batch)
        dst = jax.random.randint(key, (num_dup,), 0, batch)
        samples = samples.at[dst].set(samples[src])

        # Create cross-round duplicates by overwriting a prefix with previously inserted values
        if cached_prev_values is not None:
            reuse = jax.random.randint(key, (), 200, 2000)
            prev_n = cached_prev_values.shape.batch[0]
            reuse = jnp.minimum(reuse, prev_n)
            reuse_idx = jax.random.randint(key, (reuse,), 0, prev_n)
            samples = samples.at[:reuse].set(cached_prev_values[reuse_idx])

        # Insert this round
        table, inserted, _, _ = table.parallel_insert(samples)
        total_inserted += int(jnp.sum(inserted))

        # Immediate integrity: everything in `samples` should be findable and equal
        idx_lookup, found = table.lookup_parallel(samples)
        assert jnp.all(found), f"round {r}: some inserted states not found"
        contents = table[idx_lookup]
        assert jnp.all(
            jax.vmap(lambda x, y: x == y)(contents, samples)
        ), f"round {r}: content mismatch"

        replay_batches.append(samples)

        # Refresh cached previous values for next round: take a random subset of this round
        keep = jax.random.randint(key, (), 2000, 8000)
        keep_idx = jax.random.randint(key, (keep,), 0, batch)
        cached_prev_values = samples[keep_idx]

    # Final strict check: bulk re-lookup of the concatenation of ALL rounds' inserted samples
    replay = _concat_xtruct(replay_batches)
    idx, found = table.lookup_parallel(replay)
    assert jnp.all(found), "final replay lookup: missing keys"
    contents = table[idx]
    assert jnp.all(
        jax.vmap(lambda x, y: x == y)(contents, replay)
    ), "final replay lookup: content mismatch"

    # Size should match total insertions reported as inserted across rounds
    assert table.size == total_inserted, f"table.size {table.size} != expected {total_inserted}"


@xtructure_dataclass
class MixedValue:
    f: FieldDescriptor(jnp.float32, (3,))  # type: ignore
    i: FieldDescriptor(jnp.int32)  # type: ignore
    m: FieldDescriptor(jnp.bool_, (2, 2))  # type: ignore


def test_hash_table_reinsert_idempotent_and_size_stable():
    """Re-inserting the same data should not increase size after the first insertion."""
    batch = 30000
    table: HashTable = HashTable.build(XtructureValue, 1, int(2e5))
    samples = XtructureValue.random((batch,))

    table, inserted1, _, _ = table.parallel_insert(samples)
    size1 = int(table.size)
    assert size1 == int(jnp.sum(inserted1))

    # Re-insert identical batch
    table2, inserted2, _, _ = table.parallel_insert(samples)
    assert int(table2.size) == size1, "Size should not grow on re-insert"
    assert int(jnp.sum(inserted2)) == 0, "Second insert should report no new insertions"

    idx, found = table2.lookup_parallel(samples)
    assert jnp.all(found)
    assert jnp.all(jax.vmap(lambda x, y: x == y)(table2[idx], samples))


def test_hash_table_unique_key_priority_among_duplicates():
    """If duplicates exist in a batch, unique_key should pick the smallest key as representative."""
    n = 8000
    # Make exact duplicates: pairwise [0..n/2) duplicated into [n/2..n)
    half = n // 2
    base = XtructureValue.random((half,))
    dup = base  # exact
    samples = base.padding_as_batch((n,)).at[half:].set(dup)

    # unique_key: for each pair, make the second copy have a smaller key
    keys = jnp.arange(n, dtype=jnp.float32)
    keys = keys.at[:half].set(10.0)  # larger
    keys = keys.at[half:].set(1.0)  # smaller -> should win

    table: HashTable = HashTable.build(XtructureValue, 1, int(1e5))
    table, inserted, unique_filled, idxs = table.parallel_insert(samples, unique_key=keys)

    # For each pair, exactly one should be unique, and it should be the second half (smaller key).
    assert int(jnp.sum(unique_filled)) == half
    assert jnp.all(~unique_filled[:half])
    assert jnp.all(unique_filled[half:])

    # Inserted should also be only the representatives (first time table is empty).
    assert int(jnp.sum(inserted)) == half

    # All should be found; duplicates should map to same slot indices.
    _, found = table.lookup_parallel(samples)
    assert jnp.all(found)
    bytesed = jax.vmap(lambda x: x.bytes)(idxs)
    unique_idxs = jnp.unique(bytesed, axis=0)
    assert unique_idxs.shape[0] == half


def test_hash_table_max_probes_small_does_not_corrupt_or_oob():
    """With very small max_probes, inserts may fail, but must not corrupt size or crash."""
    batch = 20000
    # Intentionally small and dense to trigger probe exhaustion
    table: HashTable = HashTable.build(
        XtructureValue, 1, int(2000), slots_per_bucket=32, hash_size_multiplier=1, max_probes=2
    )
    samples = XtructureValue.random((batch,))

    table2, inserted, _, _ = table.parallel_insert(samples)
    inserted_n = int(jnp.sum(inserted))
    assert int(table2.size) == inserted_n

    # For those reported inserted, they must be findable and equal.
    inserted_idx = jnp.where(inserted, size=inserted_n, fill_value=0)[0]
    inserted_samples = samples[inserted_idx]
    idx, found = table2.lookup_parallel(inserted_samples)
    assert jnp.all(found)
    assert jnp.all(jax.vmap(lambda x, y: x == y)(table2[idx], inserted_samples))


def test_hash_table_fingerprint_collision_fallback_path_correctness():
    """Force frequent fingerprint collisions and ensure lookup remains correct (fallback path works).

    This tries to create a bucket where multiple different keys share the same (weakened) fingerprint.
    Lookup must still find the correct key.
    """
    import xtructure.hashtable.hashtable as ht

    # Monkeypatch fingerprint mixer to drastically increase collisions.
    orig_mix = ht._mix_fingerprint

    def weak_mix(primary, secondary, length):
        del secondary, length
        return jnp.asarray(primary, dtype=jnp.uint32) & jnp.uint32(0xFF)

    try:
        ht._mix_fingerprint = weak_mix  # type: ignore[attr-defined]
        jax.clear_caches()

        # Smallish table so buckets get multiple entries; slots_per_bucket=32 (slab-like)
        table: HashTable = HashTable.build(
            XtructureValue, 1, int(4096), slots_per_bucket=32, hash_size_multiplier=1
        )
        batch = 6000
        samples = XtructureValue.random((batch,))

        table, inserted, _, _ = table.parallel_insert(samples)
        # Work only with inserted keys to avoid dup noise
        inserted_n = int(jnp.sum(inserted))
        assert inserted_n > 1000
        inserted_idx = jnp.where(inserted, size=inserted_n, fill_value=0)[0]
        inserted_samples = samples[inserted_idx]

        # Compute bucket indices + weak fingerprints
        buckets, _, _, fps = jax.vmap(ht.get_new_idx_byterized, in_axes=(0, None, None))(
            inserted_samples, table._capacity, table.seed
        )
        buckets_h = np.asarray(jax.device_get(buckets)).astype(np.int64)
        fps_h = np.asarray(jax.device_get(fps)).astype(np.uint32)

        # Find a (bucket, fp) group with >=2 different keys
        key_bytes = np.asarray(jax.device_get(jax.vmap(lambda x: x.bytes)(inserted_samples)))
        group = {}
        for i in range(buckets_h.shape[0]):
            k = (int(buckets_h[i]), int(fps_h[i]))
            group.setdefault(k, []).append(i)
        target = None
        for k, idxs in group.items():
            if len(idxs) >= 2:
                # ensure actually different values (bytes differ)
                if not np.array_equal(key_bytes[idxs[0]], key_bytes[idxs[1]]):
                    target = idxs[:2]
                    break
        assert target is not None, "Could not find a collision group; try increasing batch"

        # Query the 2nd item; if lookup only checked first fp match, it could fail.
        query = inserted_samples[target[1]]
        idx, found = table.lookup(query)
        assert bool(found)
        retrieved = table[idx]
        assert retrieved == query

    finally:
        ht._mix_fingerprint = orig_mix  # type: ignore[attr-defined]
        jax.clear_caches()


def test_hash_table_mixed_dtype_shape_integrity():
    """Exercise hash table with a different structured value (dtypes + shapes)."""
    batch = 10000
    table: HashTable = HashTable.build(MixedValue, 1, int(8e4))
    samples = MixedValue.random((batch,))

    table, inserted, _, _ = table.parallel_insert(samples)
    assert int(table.size) == int(jnp.sum(inserted))

    idx, found = table.lookup_parallel(samples)
    assert jnp.all(found)
    assert jnp.all(jax.vmap(lambda x, y: x == y)(table[idx], samples))
