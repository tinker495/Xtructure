"""Tests for packed (bit/byte) serialization of xtructure dataclasses."""

import os

import chex
import jax
import jax.numpy as jnp
import pytest

from xtructure import FieldDescriptor, xtructure_dataclass


@xtructure_dataclass
class PackedLeafData:
    # Lots of booleans + small ints to make packing worthwhile.
    flags: FieldDescriptor.tensor(dtype=jnp.bool_, shape=(4096,), pack_bits=1)
    small3: FieldDescriptor.tensor(dtype=jnp.uint8, shape=(4096,), pack_bits=3)
    small5: FieldDescriptor.tensor(dtype=jnp.uint8, shape=(1024,), pack_bits=5)
    # A raw field (exact bytes preserved).
    floats: FieldDescriptor.tensor(dtype=jnp.float32, shape=(256,))


@xtructure_dataclass
class PackedNestedData:
    inner: FieldDescriptor.scalar(dtype=PackedLeafData)
    ids: FieldDescriptor.tensor(dtype=jnp.uint16, shape=(512,), pack_bits=8)  # values must fit in 8 bits


@pytest.fixture
def temp_files(tmp_path):
    normal = tmp_path / "normal.npz"
    packed = tmp_path / "packed.npz"
    return str(normal), str(packed)


def _make_instance(key):
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    flags = jax.random.bernoulli(k1, shape=(4096,))
    small3 = jax.random.randint(k2, shape=(4096,), minval=0, maxval=8, dtype=jnp.uint8)
    small5 = jax.random.randint(k3, shape=(1024,), minval=0, maxval=32, dtype=jnp.uint8)
    floats = jax.random.uniform(k4, shape=(256,), dtype=jnp.float32)
    inner = PackedLeafData(flags=flags, small3=small3, small5=small5, floats=floats)
    ids = jax.random.randint(k5, shape=(512,), minval=0, maxval=256, dtype=jnp.uint16)
    return PackedNestedData(inner=inner, ids=ids)


def test_to_packed_and_from_packed_roundtrip():
    original = _make_instance(jax.random.PRNGKey(0))
    packed = original.to_packed()
    restored = PackedNestedData.from_packed(packed)
    chex.assert_trees_all_equal(original, restored)


def test_save_packed_and_load_packed_roundtrip(temp_files):
    normal_path, packed_path = temp_files
    original = _make_instance(jax.random.PRNGKey(1))

    original.save(normal_path)
    original.save_packed(packed_path)

    loaded = PackedNestedData.load_packed(packed_path)
    chex.assert_trees_all_equal(original, loaded)

    # Packed file should be smaller or equal (typically much smaller here).
    assert os.path.getsize(packed_path) <= os.path.getsize(normal_path)


def test_pack_bits_range_validation_raises():
    # small3 has pack_bits=3 (valid range 0..7). Introduce 8 to force failure.
    inner = PackedLeafData(
        flags=jnp.zeros((4096,), dtype=jnp.bool_),
        small3=jnp.concatenate([jnp.zeros((4095,), dtype=jnp.uint8), jnp.array([8], dtype=jnp.uint8)]),
        small5=jnp.zeros((1024,), dtype=jnp.uint8),
        floats=jnp.zeros((256,), dtype=jnp.float32),
    )
    bad = PackedNestedData(inner=inner, ids=jnp.zeros((512,), dtype=jnp.uint16))
    with pytest.raises(ValueError, match="pack_bits=3"):
        bad.to_packed(validate_range=True)

