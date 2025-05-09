import chex
import jax
import jax.numpy as jnp
import pytest

from Xtructure import KEY_DTYPE
from Xtructure import BGPQ, xtructure_dataclass
from Xtructure.field_descriptors import FieldDescriptor


@xtructure_dataclass
class XtructureValue:
    """
    This class is a dataclass that represents a hash table heap value.
    It has two fields:
    1. index: hashtable index
    2. table_index: cuckoo table index
    """

    a: FieldDescriptor(jnp.uint8) # type: ignore
    b: FieldDescriptor(jnp.uint32, (1, 2)) # type: ignore
    c: FieldDescriptor(jnp.float32, (1, 2, 3)) # type: ignore


def rotl(x, n):
    """Rotate left operation for 32-bit integers."""
    return (x << n) | (x >> (32 - n))


@jax.jit
def xxhash(x, seed):
    """
    Implementation of xxHash algorithm for 32-bit integers.
    Args:
        x: Input value to hash
        seed: Seed value for hash function
    Returns:
        32-bit hash value
    """
    prime_1 = jnp.uint32(0x9E3779B1)
    prime_2 = jnp.uint32(0x85EBCA77)
    prime_3 = jnp.uint32(0xC2B2AE3D)
    prime_5 = jnp.uint32(0x165667B1)
    acc = jnp.uint32(seed) + prime_5
    for _ in range(4):
        lane = x & 255
        acc = acc + lane * prime_5
        acc = rotl(acc, 11) * prime_1
        x = x >> 8
    acc = acc ^ (acc >> 15)
    acc = acc * prime_2
    acc = acc ^ (acc >> 13)
    acc = acc * prime_3
    acc = acc ^ (acc >> 16)
    return acc


def heap_key_builder(x: XtructureValue):
    @jax.jit
    def _to_bytes(x):
        """Convert input to byte array."""
        return jax.lax.bitcast_convert_type(x, jnp.uint8).reshape(-1)

    @jax.jit
    def _byterize(x):
        """Convert entire state tree to flattened byte array."""
        x = jax.tree_util.tree_map(_to_bytes, x)
        x, _ = jax.tree_util.tree_flatten(x)
        return jnp.concatenate(x)

    default_bytes = _byterize(x.default())
    bytes_len = default_bytes.shape[0]
    # Calculate padding needed to make byte length multiple of 4
    pad_len = jnp.where(bytes_len % 4 != 0, 4 - (bytes_len % 4), 0)

    if pad_len > 0:

        def _to_uint32s(bytes):
            """Convert padded bytes to uint32 array."""
            x_padded = jnp.pad(bytes, (pad_len, 0), mode="constant", constant_values=0)
            x_reshaped = jnp.reshape(x_padded, (-1, 4))
            return jax.vmap(lambda x: jax.lax.bitcast_convert_type(x, jnp.uint32))(
                x_reshaped
            ).reshape(-1)

    else:

        def _to_uint32s(bytes):
            """Convert bytes directly to uint32 array."""
            x_reshaped = jnp.reshape(bytes, (-1, 4))
            return jax.vmap(lambda x: jax.lax.bitcast_convert_type(x, jnp.uint32))(
                x_reshaped
            ).reshape(-1)

    def _keys(x):
        bytes = _byterize(x)
        uint32ed = _to_uint32s(bytes)

        def scan_body(seed, x):
            result = xxhash(x, seed)
            return result, result

        hash_value, _ = jax.lax.scan(scan_body, 1, uint32ed)
        hash_value = (hash_value % (2**12)) / (2**8)
        return hash_value.astype(KEY_DTYPE)

    return jax.jit(_keys)


@pytest.fixture
def heap_setup():
    batch_size = 128
    max_size = 100000
    heap = BGPQ.build(max_size, batch_size, XtructureValue)

    _key_gen = heap_key_builder(XtructureValue)
    _key_gen = jax.jit(jax.vmap(_key_gen))

    return heap, batch_size, max_size, _key_gen


def test_heap_initialization(heap_setup):
    heap, batch_size, max_size, _key_gen = heap_setup
    assert heap is not None
    assert heap.size == 0
    assert heap.batch_size == batch_size


def test_heap_insert_and_delete_batch_size(heap_setup):
    heap, batch_size, max_size, _key_gen = heap_setup

    # Test batch insertion
    for i in range(0, 512, 1):

        value = XtructureValue.random(shape=(batch_size,), key=jax.random.PRNGKey(i))
        key = _key_gen(value)
        heap = BGPQ.insert(heap, key, value)

    assert heap.size == 512 * batch_size, f"Expected size 512 * batch_size, got {heap.size}"

    stacked_val = heap.val_store[:512]
    stacked_key = heap.key_store[:512]

    stacked_val_key = jax.vmap(_key_gen)(stacked_val)
    isclose = jnp.isclose(stacked_key, stacked_val_key)
    assert jnp.all(isclose), (
        f"inserted keys and values mismatch, this means that insert is corrupted"
        f"Key and value mismatch, \nstacked_key: \n{stacked_key},"
        f"\nstacked_val_key: \n{stacked_val_key},"
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
        assert jnp.sum(~is_larger) <= 1, (  # TODO: fix this
            f"delete_mins is corrupted"
            f"Key is not in ascending order, \nmin_key: \n{min_key},"
            f"\nlast_maximum_key: \n{last_maximum_key},"
        )
        last_maximum_key = jnp.max(min_key)

    all_keys = jnp.concatenate(all_keys)
    diff = all_keys[1:] - all_keys[:-1]
    decreasing = diff < 0
    # Verify that elements are in ascending order
    assert jnp.sum(decreasing) <= 1, (  # TODO: fix this
        f"Keys are not in ascending order: {decreasing}"
        f"\nfailed_idxs: {jnp.where(decreasing)}"
        f"\nincorrect_keys: ({all_keys[jnp.where(decreasing)[0]]},"
        f"{all_keys[jnp.where(decreasing)[0] + 1]})"
    )


def test_heap_insert_and_delete_random_size(heap_setup):
    heap, batch_size, max_size, _key_gen = heap_setup

    # Test batch insertion
    all_sizes = []
    for i in range(0, 512, 1):

        size = jax.random.randint(
            jax.random.PRNGKey(i), minval=batch_size - 10, maxval=batch_size, shape=()
        )
        value = XtructureValue.random(shape=(size,), key=jax.random.PRNGKey(i))
        key = _key_gen(value)
        key, value = BGPQ.make_batched(key, value, batch_size)
        heap = BGPQ.insert(heap, key, value)
        all_sizes.append(size)

    all_sizes = jnp.array(all_sizes)
    total_size = jnp.sum(all_sizes)
    assert heap.size == total_size, f"Expected size {total_size}, got {heap.size}"

    stacked_val = heap.val_store[: total_size // batch_size]
    stacked_key = heap.key_store[: total_size // batch_size]

    stacked_val_key = jax.vmap(_key_gen)(stacked_val)
    isclose = jnp.isclose(stacked_key, stacked_val_key)
    assert jnp.all(isclose), (
        f"inserted keys and values mismatch, this means that insert is corrupted"
        f"Key and value mismatch, \nstacked_key: \n{stacked_key},"
        f"\nstacked_val_key: \n{stacked_val_key},"
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
        assert jnp.sum(~is_larger) <= 1, (  # TODO: fix this
            f"delete_mins is corrupted"
            f"Key is not in ascending order, \nmin_key: \n{min_key},"
            f"\nlast_maximum_key: \n{last_maximum_key},"
        )
        last_maximum_key = jnp.max(min_key)

    all_keys = jnp.concatenate(all_keys)
    diff = all_keys[1:] - all_keys[:-1]
    decreasing = diff < 0
    # Verify that elements are in ascending order
    assert jnp.sum(decreasing) <= 1, (  # TODO: fix this
        f"Keys are not in ascending order: {decreasing}"
        f"\nfailed_idxs: {jnp.where(decreasing)}"
        f"\nincorrect_keys: ({all_keys[jnp.where(decreasing)[0]]},"
        f"{all_keys[jnp.where(decreasing)[0] + 1]})"
    )