"""Tests for xnp.pad behavior across padding scenarios."""

import jax.numpy as jnp
import pytest

from tests.xnp.shared_data import SimpleData, VectorData
from xtructure import numpy as xnp


def test_pad_single_to_batched():
    """Test padding a SINGLE dataclass to create a batched version."""
    data = SimpleData.default()
    data = data.replace(id=jnp.array(42), value=jnp.array(3.14))

    result = xnp.pad(data, (0, 4))

    assert result.structured_type.name == "BATCHED"
    assert result.shape.batch == (5,)
    expected_id = jnp.array(
        [42, 4294967295, 4294967295, 4294967295, 4294967295], dtype=jnp.uint32
    )
    expected_value = jnp.array(
        [3.14, jnp.inf, jnp.inf, jnp.inf, jnp.inf], dtype=jnp.float32
    )
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.allclose(result.value, expected_value)


def test_pad_batched_axis_0():
    """Test padding a BATCHED dataclass along axis 0."""
    data = SimpleData.default((3,))
    data = data.replace(
        id=jnp.array([1, 2, 3], dtype=jnp.uint32), value=jnp.array([1.0, 2.0, 3.0])
    )

    result = xnp.pad(data, (0, 2))

    assert result.structured_type.name == "BATCHED"
    assert result.shape.batch == (5,)
    expected_id = jnp.array([1, 2, 3, 4294967295, 4294967295], dtype=jnp.uint32)
    expected_value = jnp.array([1.0, 2.0, 3.0, jnp.inf, jnp.inf], dtype=jnp.float32)
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_pad_single_before_after_inserts_correctly():
    """SINGLE padding with (before, after) inserts at the correct offset."""
    data = SimpleData.default()
    data = data.replace(
        id=jnp.array(42, dtype=jnp.uint32), value=jnp.array(3.14, dtype=jnp.float32)
    )

    result = xnp.pad(data, (2, 1))

    assert result.structured_type.name == "BATCHED"
    assert result.shape.batch == (4,)
    expected_id = jnp.array([4294967295, 4294967295, 42, 4294967295], dtype=jnp.uint32)
    expected_value = jnp.array([jnp.inf, jnp.inf, 3.14, jnp.inf], dtype=jnp.float32)
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.allclose(result.value, expected_value)


def test_pad_batched_before_after_inserts_correctly():
    """BATCHED padding with (before, after) inserts at the correct offset."""
    data = SimpleData.default((3,))
    data = data.replace(
        id=jnp.array([1, 2, 3], dtype=jnp.uint32), value=jnp.array([1.0, 2.0, 3.0])
    )

    result = xnp.pad(data, (2, 1))

    assert result.structured_type.name == "BATCHED"
    assert result.shape.batch == (6,)
    expected_id = jnp.array(
        [4294967295, 4294967295, 1, 2, 3, 4294967295], dtype=jnp.uint32
    )
    expected_value = jnp.array(
        [jnp.inf, jnp.inf, 1.0, 2.0, 3.0, jnp.inf], dtype=jnp.float32
    )
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.allclose(result.value, expected_value)


def test_pad_method_alias():
    """Test that .pad() instance method works as alias to xnp.pad."""
    data = SimpleData.default((2,))
    data = data.replace(
        id=jnp.array([1, 2], dtype=jnp.uint32), value=jnp.array([1.0, 2.0])
    )

    result_xnp = xnp.pad(data, (0, 2))
    result_method = data.pad((0, 2))

    assert jnp.array_equal(result_xnp.id, result_method.id)
    assert jnp.array_equal(result_xnp.value, result_method.value)
    assert result_xnp.shape.batch == result_method.shape.batch


def test_pad_batched_with_constant_values():
    """Test padding with custom constant values."""
    data = SimpleData.default((2,))
    data = data.replace(id=jnp.array([1, 2]), value=jnp.array([1.0, 2.0]))

    result = xnp.pad(data, (0, 2), constant_values=99)

    assert result.shape.batch == (4,)
    assert jnp.array_equal(result.id, jnp.array([1, 2, 99, 99], dtype=jnp.uint32))
    assert jnp.array_equal(
        result.value, jnp.array([1.0, 2.0, 99.0, 99.0], dtype=jnp.float32)
    )


def test_pad_batched_target_shape():
    """Test padding to a target batch shape."""
    data = SimpleData.default((2, 3))

    result = xnp.pad(data, [(0, 2), (0, 2)])

    assert result.shape.batch == (4, 5)


def test_pad_tensor_fields_multidim_batch_default_constant():
    """Default constant padding should work for tensor fields with batch_ndim > 1."""
    data = VectorData.default((2, 3))
    position = jnp.arange(18, dtype=jnp.float32).reshape(2, 3, 3)
    velocity = position + 0.5
    data = data.replace(position=position, velocity=velocity)

    result = xnp.pad(data, [(1, 0), (0, 2)])

    assert result.shape.batch == (3, 5)
    assert jnp.array_equal(result.position[1:3, 0:3], position)
    assert jnp.array_equal(result.velocity[1:3, 0:3], velocity)
    assert jnp.all(jnp.isinf(result.position[0]))
    assert jnp.all(jnp.isinf(result.velocity[0]))
    assert jnp.all(jnp.isinf(result.position[:, 3:]))
    assert jnp.all(jnp.isinf(result.velocity[:, 3:]))


def test_pad_no_change_needed():
    """Test that padding with zero padding returns the same instance."""
    data = SimpleData.default((3,))
    result = xnp.pad(data, (0, 0))
    assert result is data


def test_pad_invalid_pad_width():
    """Test that invalid pad_width raises ValueError."""
    data = SimpleData.default((3,))

    with pytest.raises(
        ValueError, match="pad_width must be int, sequence of int, or sequence of pairs"
    ):
        xnp.pad(data, "invalid")
