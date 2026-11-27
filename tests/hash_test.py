import jax
import jax.numpy as jnp
import pytest

from tests.testdata import HashValueAB, OddBytesValue47
from xtructure import FieldDescriptor, HashTable, HashTableProbe, Xtructurable
from xtructure.core import hash as hash_adapter
from xtructure.hashtable.hash_utils import (
    _compute_unique_mask_from_uint32eds,
    get_new_idx_byterized_batched,
)


def test_dedupe_reused_hash_collision_falls_back_to_exact_rows():
    rows = jnp.array([[1, 2], [1, 2], [3, 4], [5, 6]], dtype=jnp.uint32)
    filled = jnp.ones((4,), dtype=jnp.bool_)
    unique_key = jnp.array([2.0, 1.0, 0.0, 0.0], dtype=jnp.float32)
    colliding_hash = jnp.zeros((4,), dtype=jnp.uint32)

    unique, representatives = _compute_unique_mask_from_uint32eds(
        rows,
        filled,
        unique_key,
        row_hash=colliding_hash,
    )

    assert jnp.array_equal(unique, jnp.array([False, True, True, True]))
    assert jnp.array_equal(representatives, jnp.array([1, 1, 2, 3]))


def test_hash_table_lookup():
    count = 1000
    sample = HashValueAB.random((count,))
    table: HashTable = HashTable.build(HashValueAB, 1, int(1e4))

    idx, found = table.lookup_parallel(sample)

    assert idx.shape == (count,)
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


@pytest.mark.parametrize("restore", [False, True])
def test_eager_bytes_accepts_python_value_validator(restore):
    # Author: tinker495. Eager encoding must validate concrete values before JIT,
    # including data restored without a cache by a raw PyTree operation.
    def nonnegative(value):
        if jnp.any(value < 0):
            raise ValueError("negative value")

    class Values(Xtructurable, bitpack="off"):
        value: FieldDescriptor.tensor(dtype=jnp.uint8, shape=(2,), validator=nonnegative)

    sample = Values(value=jnp.array([1, 2], dtype=jnp.uint8))
    if restore:
        sample = jax.tree_util.tree_map(lambda value: value, sample)
    assert jnp.array_equal(sample.bytes, jnp.array([1, 2], dtype=jnp.uint8))


@pytest.mark.parametrize(
    "wrap",
    [
        pytest.param(lambda value: value, id="direct"),
        pytest.param(lambda value: [value], id="list"),
        pytest.param(lambda value: (value,), id="tuple"),
        pytest.param(lambda value: {"items": [value]}, id="dict"),
    ],
)
def test_tree_to_bytes_rejects_invalid_structured_nodes(wrap):
    # Author: tinker495. Containers must not bypass the data contract when raw
    # PyTree reconstruction changes a declared uint8 field to uint16.
    sample = OddBytesValue47(payload=jnp.arange(47, dtype=jnp.uint8))
    invalid = jax.tree_util.tree_map(lambda value: value.astype(jnp.uint16), sample)
    with pytest.raises(TypeError, match="expected dtype"):
        hash_adapter.tree_to_bytes(wrap(invalid))


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


def test_parallel_insert_assume_unique_is_bit_identical():
    batch = 4096
    samples = HashValueAB.random((batch,), key=jax.random.PRNGKey(123))
    samples = samples.at[batch // 2 :].set(samples[: batch // 2])
    filled = jnp.ones((batch,), dtype=jnp.bool_)
    unique_key = jax.random.uniform(jax.random.PRNGKey(124), (batch,))

    regular: HashTable = HashTable.build(HashValueAB, 13, int(1e5))
    fast: HashTable = HashTable.build(HashValueAB, 13, int(1e5))
    probe = get_new_idx_byterized_batched(samples, regular._capacity, regular.seed)
    unique, _ = _compute_unique_mask_from_uint32eds(
        probe.uint32ed,
        filled,
        unique_key,
        row_hash=probe.fingerprint,
    )

    regular, inserted_regular, unique_regular, idx_regular = regular.parallel_insert(
        samples, unique, unique_key, probe
    )
    fast, inserted_fast, unique_fast, idx_fast = fast.parallel_insert(
        samples,
        unique,
        unique_key,
        probe,
        assume_unique=True,
    )

    assert bool(jnp.all(inserted_regular == inserted_fast))
    assert bool(jnp.all(unique_regular == unique_fast))
    assert bool(jnp.all(idx_regular.index == idx_fast.index))
    _assert_tables_bit_identical(regular, fast)


def test_lookup_parallel_deduplicates_and_restores_duplicates():
    # Author: tinker495. Parallel lookup must deduplicate active duplicates
    # internally while restoring accurate (HashIdx, found) answers to every
    # active row and returning zeroed results for inactive or empty batches.
    distinct = HashValueAB.default((3,))
    distinct = distinct.replace(
        a=jnp.array([10, 20, 30], dtype=jnp.uint8),
        b=jnp.array([[[1, 2]], [[3, 4]], [[5, 6]]], dtype=jnp.uint32),
    )

    table: HashTable = HashTable.build(HashValueAB, 7, 128)
    table, inserted, _, inserted_idxs = table.parallel_insert(distinct[:2])
    assert bool(jnp.all(inserted))

    # Layout:
    # 0: s0 (inactive)
    # 1..3: s0 (active duplicates, present in table)
    # 4..5: s2 (active duplicates, missing from table)
    # 6: s1 (active singleton, present in table)
    # 7: s1 (inactive)
    order = jnp.array([0, 0, 0, 0, 2, 2, 1, 1], dtype=jnp.int32)
    batch = distinct[order]
    filled = jnp.array([False, True, True, True, True, True, True, False])

    idx, found = table.lookup_parallel(batch, filled)
    assert jnp.array_equal(
        found,
        jnp.array([False, True, True, True, False, False, True, False]),
    )
    assert int(idx.index[1]) == int(inserted_idxs.index[0])
    assert int(idx.index[2]) == int(inserted_idxs.index[0])
    assert int(idx.index[3]) == int(inserted_idxs.index[0])
    assert int(idx.index[4]) == int(idx.index[5])
    assert int(idx.index[6]) == int(inserted_idxs.index[1])
    assert int(idx.index[0]) == 0
    assert int(idx.index[7]) == 0

    for disabled in (False, jnp.bool_(False), jnp.zeros_like(filled)):
        empty_idx, empty_found = table.lookup_parallel(batch, disabled)
        assert jnp.array_equal(empty_idx.index, jnp.zeros((8,), dtype=empty_idx.index.dtype))
        assert not bool(jnp.any(empty_found))

    empty_batch = HashValueAB.default((0,))
    empty_idx, empty_found, empty_probe = table.lookup_parallel_with_probe(empty_batch)
    assert empty_idx.shape == (0,)
    assert empty_found.shape == (0,)
    assert empty_probe.index.shape == (0,)


def test_lookup_parallel_probe_reuse_and_assume_unique_are_bit_identical():
    # Author: tinker495. Search loops thread HashTableProbe from
    # lookup_parallel_with_probe into subsequent lookups and parallel_insert, and
    # set assume_unique=True after upstream xnp.unique_mask deduplication; all
    # paths must stay bit-identical while rejecting mismatched probes at trace time.
    batch = 512
    samples = HashValueAB.random((batch,), key=jax.random.PRNGKey(321))
    samples = samples.at[batch // 2 :].set(samples[: batch // 2])
    filled = jax.random.bernoulli(jax.random.PRNGKey(322), p=0.8, shape=(batch,))

    seed_table: HashTable = HashTable.build(HashValueAB, 17, int(1e4))
    seed_table, _, _, _ = seed_table.parallel_insert(samples[: batch // 4])

    idx_plain, found_plain = seed_table.lookup_parallel(samples, filled)
    idx_probe, found_probe, probe = seed_table.lookup_parallel_with_probe(samples, filled)
    assert isinstance(probe, HashTableProbe)
    assert jnp.array_equal(idx_plain.index, idx_probe.index)
    assert jnp.array_equal(found_plain, found_probe)

    idx_reused, found_reused = seed_table.lookup_parallel(samples, filled, probe=probe)
    assert jnp.array_equal(idx_plain.index, idx_reused.index)
    assert jnp.array_equal(found_plain, found_reused)

    unique, _ = _compute_unique_mask_from_uint32eds(
        probe.uint32ed,
        filled,
        None,
        row_hash=probe.fingerprint,
    )
    idx_dedup_regular, found_dedup_regular = seed_table.lookup_parallel(
        samples,
        unique,
        probe=probe,
    )
    idx_dedup_fast, found_dedup_fast, probe_fast = seed_table.lookup_parallel_with_probe(
        samples,
        unique,
        probe=probe,
        assume_unique=True,
    )
    assert jnp.array_equal(idx_dedup_regular.index, idx_dedup_fast.index)
    assert jnp.array_equal(found_dedup_regular, found_dedup_fast)
    assert jnp.array_equal(probe.uint32ed, probe_fast.uint32ed)

    table_a, ins_a, uniq_a, out_idx_a = seed_table.parallel_insert(samples, filled)
    table_b, ins_b, uniq_b, out_idx_b = seed_table.parallel_insert(
        samples,
        filled,
        probe=probe,
    )
    assert jnp.array_equal(ins_a, ins_b)
    assert jnp.array_equal(uniq_a, uniq_b)
    assert jnp.array_equal(out_idx_a.index, out_idx_b.index)
    _assert_tables_bit_identical(table_a, table_b)

    wrong_probe = probe._replace(index=probe.index[:10])
    with pytest.raises(ValueError, match="HashTableProbe.index shape"):
        seed_table.lookup_parallel(samples, False, probe=wrong_probe)
