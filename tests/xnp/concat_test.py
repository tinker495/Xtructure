"""Tests for xnp.concatenate behavior."""

import jax.numpy as jnp
import pytest

from tests.xnp.shared_data import SimpleData, VectorData
from xtructure import numpy as xnp


def test_concat_single_dataclasses():
    """Test concatenating SINGLE structured dataclasses."""
    data1 = SimpleData.default()
    data1 = data1.replace(id=jnp.array(1), value=jnp.array(1.0))
    data2 = SimpleData.default()
    data2 = data2.replace(id=jnp.array(2), value=jnp.array(2.0))
    data3 = SimpleData.default()
    data3 = data3.replace(id=jnp.array(3), value=jnp.array(3.0))

    result = xnp.concatenate([data1, data2, data3])

    assert result.structured_type.name == "BATCHED"
    assert result.shape.batch == (3,)
    assert jnp.array_equal(result.id, jnp.array([1, 2, 3]))
    assert jnp.array_equal(result.value, jnp.array([1.0, 2.0, 3.0]))


def test_concat_batched_dataclasses():
    """Test concatenating BATCHED structured dataclasses."""
    data1 = SimpleData.default(shape=(2,))
    data1 = data1.replace(id=jnp.array([1, 2]), value=jnp.array([1.0, 2.0]))
    data2 = SimpleData.default(shape=(3,))
    data2 = data2.replace(id=jnp.array([3, 4, 5]), value=jnp.array([3.0, 4.0, 5.0]))

    result = xnp.concatenate([data1, data2])

    assert result.structured_type.name == "BATCHED"
    assert result.shape.batch == (5,)
    assert jnp.array_equal(result.id, jnp.array([1, 2, 3, 4, 5]))
    assert jnp.array_equal(result.value, jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]))


def test_concat_vector_dataclasses():
    """Test concatenating dataclasses with vector fields."""
    data1 = VectorData.default(shape=(2,))
    data1 = data1.replace(
        position=jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        velocity=jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
    )
    data2 = VectorData.default(shape=(1,))
    data2 = data2.replace(
        position=jnp.array([[7.0, 8.0, 9.0]]), velocity=jnp.array([[0.7, 0.8, 0.9]])
    )

    result = xnp.concatenate([data1, data2])

    assert result.structured_type.name == "BATCHED"
    assert result.shape.batch == (3,)
    expected_pos = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    expected_vel = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    assert jnp.allclose(result.position, expected_pos)
    assert jnp.allclose(result.velocity, expected_vel)


def test_concat_empty_list():
    """Test that concatenating empty list raises ValueError."""
    with pytest.raises(ValueError, match="Cannot concatenate empty list"):
        xnp.concatenate([])


def test_concat_single_item():
    """Test that concatenating single item returns the item itself."""
    data = SimpleData.default()
    result = xnp.concatenate([data])
    assert result is data


def test_concat_incompatible_types():
    """Test that concatenating different types raises ValueError."""
    simple_data = SimpleData.default()
    vector_data = VectorData.default()

    with pytest.raises(ValueError, match="All dataclasses must be of the same type"):
        xnp.concatenate([simple_data, vector_data])
