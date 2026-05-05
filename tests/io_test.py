"""Tests for saving and loading xtructure dataclasses."""

import os

import chex
import jax
import pytest

from tests.testdata import NestedData, SimpleData, VectorData

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


def test_load_raises_key_error_when_field_data_missing(tmp_path):
    """A .npz with valid metadata but missing field arrays must raise KeyError, not silent dropout."""
    import numpy as np

    invalid_file = tmp_path / "missing_fields.npz"
    np.savez_compressed(
        str(invalid_file),
        __xtructure_class_module__=np.array(SimpleData.__module__),
        __xtructure_class_name__=np.array(SimpleData.__name__),
    )

    with pytest.raises(KeyError, match="Missing field"):
        SimpleData.load(str(invalid_file))


def test_load_tolerates_legacy_extra_dtype_metadata(temp_file):
    """Older xtructure releases wrote a `<field>.__xtructure_bitpack__.dtype` extra key.

    Newer .npz files no longer write it, but legacy .npz inputs must still load
    cleanly: numpy ignores extra keys, and the loader must not depend on it.
    """
    import numpy as np

    instance = VectorData.random(shape=(4,), key=jax.random.PRNGKey(0))
    instance.save(temp_file)

    # Re-pack the .npz with an additional spurious legacy `*.__xtructure_bitpack__.dtype` entry
    # to simulate a file written by older releases.
    with np.load(temp_file, allow_pickle=False) as data:
        contents = {key: np.array(data[key]) for key in data.files}
    contents["bogus_legacy.__xtructure_bitpack__.dtype"] = np.array("uint8", dtype=np.str_)
    np.savez_compressed(temp_file, **contents)

    loaded = VectorData.load(temp_file)
    chex.assert_trees_all_equal(instance, loaded)
