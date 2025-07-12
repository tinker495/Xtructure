"""Tests for saving and loading xtructure dataclasses."""

import os

import chex
import jax
import jax.numpy as jnp
import pytest

from xtructure import FieldDescriptor, xtructure_dataclass

# --- Test Dataclasses (re-defined here for a self-contained test file) ---


@xtructure_dataclass
class SimpleData:
    id: FieldDescriptor[jnp.uint32]
    value: FieldDescriptor[jnp.float32]


@xtructure_dataclass
class VectorData:
    position: FieldDescriptor[jnp.float32, (3,)]
    velocity: FieldDescriptor[jnp.float32, (3,)]


@xtructure_dataclass
class NestedData:
    simple: FieldDescriptor[SimpleData]
    vector: FieldDescriptor[VectorData]


# --- Test Fixtures ---


@pytest.fixture
def temp_file(tmp_path):
    """Provides a temporary file path for saving and loading."""
    file_path = tmp_path / "test_data.npz"
    yield str(file_path)
    # Cleanup is handled by pytest's tmp_path fixture


# --- Test Cases ---


def test_save_and_load_simple_instance(temp_file):
    """Tests saving and loading a single, non-batched instance."""
    key = jax.random.PRNGKey(0)
    original_instance = SimpleData.random(key=key)

    # Save the instance using the new method
    original_instance.save(temp_file)

    # Load the instance using the new class method
    loaded_instance = SimpleData.load(temp_file)

    # Verify the type and data
    assert isinstance(loaded_instance, SimpleData)
    chex.assert_trees_all_equal(original_instance, loaded_instance)


def test_save_and_load_batched_instance(temp_file):
    """Tests saving and loading a batched instance."""
    key = jax.random.PRNGKey(1)
    original_instance = VectorData.random(shape=(10,), key=key)

    # Save
    original_instance.save(temp_file)

    # Load
    loaded_instance = VectorData.load(temp_file)

    # Verify
    assert isinstance(loaded_instance, VectorData)
    assert loaded_instance.shape.batch == (10,)
    chex.assert_trees_all_equal(original_instance, loaded_instance)


def test_save_and_load_nested_instance(temp_file):
    """Tests saving and loading an instance with nested xtructure dataclasses."""
    key = jax.random.PRNGKey(2)
    original_instance = NestedData.random(key=key)

    # Save
    original_instance.save(temp_file)

    # Load
    loaded_instance = NestedData.load(temp_file)

    # Verify
    assert isinstance(loaded_instance, NestedData)
    assert isinstance(loaded_instance.simple, SimpleData)
    assert isinstance(loaded_instance.vector, VectorData)
    chex.assert_trees_all_equal(original_instance, loaded_instance)


def test_save_and_load_nested_batched_instance(temp_file):
    """Tests saving and loading a batched instance with nested structures."""
    key = jax.random.PRNGKey(3)
    original_instance = NestedData.random(shape=(5, 2), key=key)

    # Save
    original_instance.save(temp_file)

    # Load
    loaded_instance = NestedData.load(temp_file)

    # Verify
    assert loaded_instance.shape.batch == (5, 2)
    assert loaded_instance.simple.shape.batch == (5, 2)
    assert loaded_instance.vector.position.shape == (5, 2, 3)
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
