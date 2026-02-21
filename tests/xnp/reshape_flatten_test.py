"""Tests for reshape and flatten helpers."""

import jax.numpy as jnp
import pytest

from tests.xnp.shared_data import SimpleData, VectorData
from xtructure import numpy as xnp


def test_reshape_wrapper():
    """Test that the reshape wrapper function works like the built-in method."""
    data = SimpleData.default(shape=(6,))
    data = data.replace(id=jnp.arange(6), value=jnp.arange(6, dtype=jnp.float32))

    result_wrapper = xnp.reshape(data, (2, 3))
    result_builtin = data.reshape((2, 3))

    assert jnp.array_equal(result_wrapper.id, result_builtin.id)
    assert jnp.array_equal(result_wrapper.value, result_builtin.value)
    assert result_wrapper.shape.batch == result_builtin.shape.batch == (2, 3)


def test_reshape_with_minus_one():
    """Test reshape with -1 to automatically calculate dimensions."""
    data = SimpleData.default(shape=(12,))
    data = data.replace(id=jnp.arange(12), value=jnp.arange(12, dtype=jnp.float32))

    result1 = xnp.reshape(data, (2, -1))
    assert result1.shape.batch == (2, 6)
    assert jnp.array_equal(result1.id, jnp.arange(12).reshape(2, 6))
    assert jnp.array_equal(
        result1.value, jnp.arange(12, dtype=jnp.float32).reshape(2, 6)
    )

    result2 = xnp.reshape(data, (-1, 3))
    assert result2.shape.batch == (4, 3)
    assert jnp.array_equal(result2.id, jnp.arange(12).reshape(4, 3))
    assert jnp.array_equal(
        result2.value, jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    )

    result3 = xnp.reshape(data, (-1,))
    assert result3.shape.batch == (12,)
    assert jnp.array_equal(result3.id, jnp.arange(12))
    assert jnp.array_equal(result3.value, jnp.arange(12, dtype=jnp.float32))


def test_reshape_with_minus_one_2d():
    """Test reshape with -1 on 2D data."""
    data = SimpleData.default(shape=(8, 3))
    data = data.replace(
        id=jnp.arange(24).reshape(8, 3),
        value=jnp.arange(24, dtype=jnp.float32).reshape(8, 3),
    )

    result1 = xnp.reshape(data, (2, -1, 3))
    assert result1.shape.batch == (2, 4, 3)
    expected_id = jnp.arange(24).reshape(2, 4, 3)
    expected_value = jnp.arange(24, dtype=jnp.float32).reshape(2, 4, 3)
    assert jnp.array_equal(result1.id, expected_id)
    assert jnp.array_equal(result1.value, expected_value)

    result2 = xnp.reshape(data, (4, -1))
    assert result2.shape.batch == (4, 6)
    expected_id = jnp.arange(24).reshape(4, 6)
    expected_value = jnp.arange(24, dtype=jnp.float32).reshape(4, 6)
    assert jnp.array_equal(result2.id, expected_id)
    assert jnp.array_equal(result2.value, expected_value)


def test_reshape_with_minus_one_errors():
    """Test that reshape with invalid -1 usage raises appropriate errors."""
    data = SimpleData.default(shape=(10,))
    data = data.replace(id=jnp.arange(10), value=jnp.arange(10, dtype=jnp.float32))

    with pytest.raises(ValueError, match="Only one -1 is allowed in new_shape"):
        xnp.reshape(data, (-1, -1))

    with pytest.raises(
        ValueError,
        match="Total length 10 is not divisible by the product of other dimensions 3",
    ):
        xnp.reshape(data, (3, -1))

    with pytest.raises(
        ValueError, match="Cannot infer -1 dimension when other dimensions are 0"
    ):
        xnp.reshape(data, (0, -1))


def test_reshape_with_minus_one_vector_data():
    """Test reshape with -1 on vector data."""
    data = VectorData.default(shape=(12,))
    data = data.replace(
        position=jnp.arange(36, dtype=jnp.float32).reshape(12, 3),
        velocity=jnp.arange(36, dtype=jnp.float32).reshape(12, 3) + 100,
    )

    result = xnp.reshape(data, (3, -1))
    assert result.shape.batch == (3, 4)
    assert result.position.shape == (3, 4, 3)
    assert result.velocity.shape == (3, 4, 3)

    expected_position = jnp.arange(36, dtype=jnp.float32).reshape(3, 4, 3)
    expected_velocity = (jnp.arange(36, dtype=jnp.float32) + 100).reshape(3, 4, 3)
    assert jnp.array_equal(result.position, expected_position)
    assert jnp.array_equal(result.velocity, expected_velocity)


def test_flatten_wrapper():
    """Test that xnp.flatten calls the existing dataclass flatten method."""
    dc = SimpleData.default(shape=(2, 3))
    dc = dc.replace(
        id=jnp.arange(6).reshape(2, 3),
        value=jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
    )

    result = xnp.flatten(dc)
    expected = dc.flatten()

    assert jnp.array_equal(result.id, expected.id)
    assert jnp.array_equal(result.value, expected.value)
