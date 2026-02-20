"""Tests for saving and loading xtructure dataclasses."""

import os

import chex
import jax
import numpy as np
import pytest

from tests.testdata import NestedData, SimpleData, VectorData
from xtructure.io.io import (
    _bitpack_keys,
    _flatten_instance_for_save,
    _unflatten_data_for_load,
    load,
    save,
)

# --- Test Fixtures ---


@pytest.fixture
def temp_file(tmp_path):
    """Provides a temporary file path for saving and loading."""
    file_path = tmp_path / "test_data.npz"
    yield str(file_path)
    # Cleanup is handled by pytest's tmp_path fixture


# --- Test Cases ---


@pytest.mark.parametrize(
    "cls,shape,seed",
    [
        (SimpleData, (), 0),
        (VectorData, (10,), 1),
        (NestedData, (), 2),
        (NestedData, (5, 2), 3),
    ],
)
def test_save_and_load_roundtrip(temp_file, cls, shape, seed):
    """Generic save/load roundtrip over multiple dataclass types and batch shapes."""
    key = jax.random.PRNGKey(seed)
    original_instance = cls.random(shape=shape, key=key)

    original_instance.save(temp_file)
    loaded_instance = cls.load(temp_file)

    assert isinstance(loaded_instance, cls)
    chex.assert_trees_all_equal(original_instance, loaded_instance)


def test_load_nonexistent_file():
    """Tests that loading a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        SimpleData.load("non_existent_file_12345.npz")


def test_load_invalid_file(tmp_path):
    """Tests that loading a file without proper metadata raises ValueError."""
    invalid_file = tmp_path / "invalid.npz"
    # Save a numpy file without the required metadata keys
    import numpy as np

    np.savez_compressed(invalid_file, data=np.array([1, 2, 3]))

    with pytest.raises(ValueError, match="File is missing necessary xtructure metadata"):
        SimpleData.load(str(invalid_file))


def test_save_to_new_directory(tmp_path):
    """Tests that save function can create a new directory."""
    dir_path = tmp_path / "new_dir"
    file_path = dir_path / "test.npz"

    assert not os.path.exists(dir_path)

    instance = SimpleData.default()
    instance.save(str(file_path))

    assert os.path.exists(file_path)


# --- Direct import tests ---


def test_direct_save_and_load(tmp_path):
    """Calls save() and load() directly (not via dataclass methods)."""
    file_path = str(tmp_path / "direct.npz")
    key = jax.random.PRNGKey(42)
    original = SimpleData.random(shape=(), key=key)

    save(file_path, original)
    loaded = load(file_path)

    assert isinstance(loaded, SimpleData)
    chex.assert_trees_all_equal(original, loaded)


def test_flatten_roundtrip(tmp_path):
    """Tests _flatten_instance_for_save produces expected top-level dict keys."""
    key = jax.random.PRNGKey(7)
    instance = SimpleData.random(shape=(), key=key)

    flat = _flatten_instance_for_save(instance, packed=False)

    assert "id" in flat, "Expected 'id' key in flattened output"
    assert "value" in flat, "Expected 'value' key in flattened output"
    assert isinstance(flat["id"], np.ndarray)
    assert isinstance(flat["value"], np.ndarray)

    # Round-trip through _unflatten_data_for_load
    reconstructed_fields = _unflatten_data_for_load(SimpleData, flat)
    reconstructed = SimpleData(**reconstructed_fields)
    chex.assert_trees_all_equal(instance, reconstructed)


def test_bitpack_keys_format():
    """Tests _bitpack_keys returns the four expected key strings."""
    data_key, shape_key, bits_key, dtype_key = _bitpack_keys("some.field")

    assert data_key.startswith("some.field.")
    assert shape_key.startswith("some.field.")
    assert bits_key.startswith("some.field.")
    assert dtype_key.startswith("some.field.")

    assert data_key.endswith(".data")
    assert shape_key.endswith(".shape")
    assert bits_key.endswith(".bits")
    assert dtype_key.endswith(".dtype")

    # All four keys must be distinct
    assert len({data_key, shape_key, bits_key, dtype_key}) == 4


def test_save_non_xtructure_raises(tmp_path):
    """Tests that saving a non-xtructure object raises TypeError."""
    file_path = str(tmp_path / "bad.npz")

    class PlainObject:
        pass

    with pytest.raises(TypeError):
        save(file_path, PlainObject())


def test_load_missing_metadata_raises(tmp_path):
    """Tests that load() raises ValueError when metadata keys are absent."""
    invalid_file = tmp_path / "no_meta.npz"
    np.savez_compressed(invalid_file, some_array=np.array([1, 2, 3]))

    with pytest.raises(ValueError, match="File is missing necessary xtructure metadata"):
        load(str(invalid_file))
