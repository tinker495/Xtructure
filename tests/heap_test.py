import random

import jax
import jax.numpy as jnp
import pytest

from xtructure import BGPQ, FieldDescriptor, xtructure_dataclass


@xtructure_dataclass
class XtructureValue:
    """
    This class is a dataclass that represents a hash table heap value.
    It has two fields:
    1. index: hashtable index
    2. table_index: cuckoo table index
    """

    a: FieldDescriptor(jnp.uint8)  # type: ignore
    b: FieldDescriptor(jnp.uint32, (1, 2))  # type: ignore
    c: FieldDescriptor(jnp.float32, (1, 2, 3))  # type: ignore


@jax.jit
def key_gen(x: XtructureValue) -> float:
    uint32_hash = x.hash()
    key = uint32_hash % (2**12) / (2**8)
    return key.astype(jnp.float32)


@pytest.fixture
def heap_setup():
    batch_size = int(1e4)
    max_size = int(2e7)
    heap = BGPQ.build(max_size, batch_size, XtructureValue, jnp.float32)

    _key_gen = jax.jit(jax.vmap(key_gen))

    return heap, batch_size, max_size, _key_gen


def test_heap_initialization(heap_setup):
    heap, batch_size, max_size, _key_gen = heap_setup
    assert heap is not None
    assert heap.size == 0
    assert heap.batch_size == batch_size


@pytest.mark.parametrize(
    "N", [128, 256, 311, 512, 707] + [random.randint(1, 700) for _ in range(5)]
)
def test_heap_insert_and_delete_batch_size(heap_setup, N):
    heap, batch_size, max_size, _key_gen = heap_setup
    rnd_key = jax.random.PRNGKey(random.randint(0, 1000000))

    # Test batch insertion
    total_size = 0
    for i in range(0, N, 1):
        rnd_key, seed1 = jax.random.split(rnd_key, 2)
        value = XtructureValue.random(shape=(batch_size,), key=seed1)
        key = _key_gen(value)
        heap = heap.insert(key, value)
        total_size += batch_size
        assert heap.size == total_size, (
            f"Expected size {total_size}, got {heap.size},"
            f"heap.heap_size: {heap.heap_size}, heap.buffer_size: {heap.buffer_size}"
        )

    stacked_val = heap.val_store[:N]
    stacked_key = heap.key_store[:N]

    stacked_val_key = jax.vmap(_key_gen)(stacked_val)
    isclose = jnp.isclose(stacked_key, stacked_val_key)
    assert jnp.all(isclose), (
        f"inserted keys and values mismatch, this means that insert is corrupted"
        f"Key and value mismatch, \nstacked_key: \n{stacked_key[jnp.where(~isclose)]},"
        f"\nstacked_val_key: \n{stacked_val_key[jnp.where(~isclose)]},"
        f"\nstacked_val: \n{stacked_val[jnp.where(~isclose)][:3]},"
        f"\nidexs: \n{jnp.where(~isclose)}"
    )

    # Test batch deletion
    all_keys = []
    last_maximum_key = -jnp.inf
    while heap.size > 0:
        heap, min_key, min_val = BGPQ.delete_mins(heap)
        filled = jnp.isfinite(min_key)
        assert jnp.any(filled), (
            f"delete_mins is corrupted"
            f"No keys to delete, \nheap: \n{heap},"
            f"\nheap.size: \n{heap.size},"
        )

        # check key and value matching
        isclose = jnp.isclose(min_key, _key_gen(min_val)) | ~filled
        assert jnp.all(isclose), (
            f"delete_mins is corrupted"
            f"Key and value mismatch, \nmin_key: \n{min_key},"
            f"\nmin_val_key: \n{_key_gen(min_val)},"
            f"\nidexs: \n{jnp.where(~isclose)}"
        )
        all_keys.append(min_key)
        is_larger = min_key >= last_maximum_key
        assert jnp.sum(~is_larger) < 1, (
            f"delete_mins is corrupted"
            f"Key is not in ascending order, \nmin_key: \n{min_key},"
            f"\nlast_maximum_key: \n{last_maximum_key},"
        )
        last_maximum_key = jnp.max(min_key)

    all_keys = jnp.concatenate(all_keys)
    diff = all_keys[1:] - all_keys[:-1]
    decreasing = diff < 0
    # Verify that elements are in ascending order
    assert jnp.sum(decreasing) < 1, (
        f"Keys are not in ascending order: {decreasing}"
        f"\nfailed_idxs: {jnp.where(decreasing)}"
        f"\nincorrect_keys: ({all_keys[jnp.where(decreasing)[0]]},"
        f"{all_keys[jnp.where(decreasing)[0] + 1]})"
    )


@pytest.mark.parametrize(
    "N", [128, 256, 311, 512, 707] + [random.randint(1, 700) for _ in range(5)]
)
def test_heap_insert_and_delete_random_size(heap_setup, N):
    heap, batch_size, max_size, _key_gen = heap_setup
    rnd_key = jax.random.PRNGKey(random.randint(0, 1000000))

    # Test batch insertion
    total_size = 0
    for i in range(0, N, 1):
        rnd_key, seed1, seed2 = jax.random.split(rnd_key, 3)
        size = jax.random.randint(seed1, minval=1, maxval=8, shape=()) * batch_size // 8
        value = XtructureValue.random(shape=(size,), key=seed2)
        key = _key_gen(value)
        key, value = BGPQ.make_batched(key, value, batch_size)
        heap = heap.insert(key, value)
        total_size += size
        assert heap.size == total_size, (
            f"Expected size {total_size}, got {heap.size},"
            f"heap.heap_size: {heap.heap_size}, heap.buffer_size: {heap.buffer_size}"
        )

    stacked_val = heap.val_store[: total_size // batch_size]
    stacked_key = heap.key_store[: total_size // batch_size]

    stacked_val_key = jax.vmap(_key_gen)(stacked_val)
    isclose = jnp.isclose(stacked_key, stacked_val_key)
    assert jnp.all(isclose), (
        f"inserted keys and values mismatch, this means that insert is corrupted"
        f"Key and value mismatch, \nstacked_key: \n{stacked_key[jnp.where(~isclose)]},"
        f"\nstacked_val_key: \n{stacked_val_key[jnp.where(~isclose)]},"
        f"\nstacked_val: \n{stacked_val[jnp.where(~isclose)][:3]},"
        f"\nidexs: \n{jnp.where(~isclose)}"
    )

    # Test batch deletion
    all_keys = []
    last_maximum_key = -jnp.inf
    while heap.size > 0:
        heap, min_key, min_val = BGPQ.delete_mins(heap)
        filled = jnp.isfinite(min_key)
        assert jnp.any(filled), (
            f"delete_mins is corrupted"
            f"No keys to delete, \nheap: \n{heap},"
            f"\nheap.size: \n{heap.size},"
        )

        # check key and value matching
        isclose = jnp.isclose(min_key, _key_gen(min_val)) | ~filled
        assert jnp.all(isclose), (
            f"delete_mins is corrupted"
            f"Key and value mismatch, \nmin_key: \n{min_key},"
            f"\nmin_val_key: \n{_key_gen(min_val)},"
            f"\nidexs: \n{jnp.where(~isclose)}"
        )
        all_keys.append(min_key)
        is_larger = min_key >= last_maximum_key
        assert jnp.sum(~is_larger) < 1, (
            f"delete_mins is corrupted"
            f"Key is not in ascending order, \nmin_key: \n{min_key},"
            f"\nlast_maximum_key: \n{last_maximum_key},"
        )
        last_maximum_key = jnp.max(min_key)

    all_keys = jnp.concatenate(all_keys)
    diff = all_keys[1:] - all_keys[:-1]
    decreasing = diff < 0
    # Verify that elements are in ascending order
    assert jnp.sum(decreasing) < 1, (
        f"Keys are not in ascending order: {decreasing}"
        f"\nfailed_idxs: {jnp.where(decreasing)}"
        f"\nincorrect_keys: ({all_keys[jnp.where(decreasing)[0]]},"
        f"{all_keys[jnp.where(decreasing)[0] + 1]})"
    )


def test_heap_overflow_eviction():
    """
    Test that the heap correctly evicts the lowest priority elements (largest keys)
    when it is full, even if those elements are not on the default insertion path.
    """
    batch_size = 4
    # branch_size = 3 -> total_size = 12
    total_size = 12
    heap = BGPQ.build(total_size, batch_size, XtructureValue, jnp.float32)

    # Initialize keys for 3 batches
    # 1. 100s (Worst elements)
    k1 = jnp.full((batch_size,), 100.0, dtype=jnp.float32)
    v1 = XtructureValue.default((batch_size,))

    # 2. 80s (Best elements, will be at Root)
    k2 = jnp.full((batch_size,), 80.0, dtype=jnp.float32)
    v2 = XtructureValue.default((batch_size,))

    # 3. 90s (Middle elements)
    k3 = jnp.full((batch_size,), 90.0, dtype=jnp.float32)
    v3 = XtructureValue.default((batch_size,))

    # Insert to fill the heap
    # Order matters to place 100s off the default path
    # Path for size=0 -> 0
    heap = heap.insert(k3, v3)  # Root: 90s
    # Path for size=1 -> 1 (child)
    heap = heap.insert(k2, v2)  # Root: 80s, Node 1: 90s
    # Path for size=2 -> 2 (child)
    heap = heap.insert(k1, v1)  # Root: 80s, Node 1: 90s, Node 2: 100s

    # Verify state before overflow
    assert jnp.all(heap.key_store[0] == 80.0)
    assert jnp.all(heap.key_store[1] == 90.0)
    assert jnp.all(heap.key_store[2] == 100.0)
    assert heap.heap_size == 2  # 0-based index of last node (so size is 3 branches)

    # Now insert 95s (Better than 100s, worse than 80s and 90s)
    k_new = jnp.full((batch_size,), 95.0, dtype=jnp.float32)
    v_new = XtructureValue.default((batch_size,))

    heap = heap.insert(k_new, v_new)

    # Expectation: 100s should be evicted, 95s should be retained.
    # Since 100s were at Node 2, and 95s > 80s, 95s should not replace root.
    # It should eventually find its way to replace 100s.

    # Check all keys in the heap
    all_keys = heap.key_store.flatten()
    # Filter out Infs if any (shouldn't be any)
    valid_keys = all_keys[jnp.isfinite(all_keys)]

    assert jnp.all(valid_keys <= 95.0), f"Found keys > 95.0: {valid_keys[valid_keys > 95.0]}"
    assert jnp.sum(valid_keys == 95.0) == batch_size, "New keys (95.0) not found in heap"
    assert jnp.sum(valid_keys == 100.0) == 0, "Old worst keys (100.0) failed to be evicted"

    # Also verify structure (implementation detail check, but good for debugging)
    # 95s should be at Node 2 now
    assert jnp.all(heap.key_store[2] == 95.0)
