"""Tests for the xnp.take helpers."""

import jax.numpy as jnp

from tests.xnp.shared_data import SimpleData, VectorData
from xtructure import numpy as xnp


def test_take_basic():
    """Test basic take functionality."""
    data = SimpleData.default((5,))
    data = data.replace(
        id=jnp.array([1, 2, 3, 4, 5], dtype=jnp.uint32),
        value=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32),
    )

    indices = jnp.array([0, 2, 4])
    result = xnp.take(data, indices)

    expected_id = jnp.array([1, 3, 5], dtype=jnp.uint32)
    expected_value = jnp.array([1.0, 3.0, 5.0], dtype=jnp.float32)
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_take_with_negative_indices():
    """Test take with negative indices."""
    data = SimpleData.default((4,))
    data = data.replace(
        id=jnp.array([1, 2, 3, 4], dtype=jnp.uint32),
        value=jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
    )

    indices = jnp.array([-1, -2])
    result = xnp.take(data, indices)

    expected_id = jnp.array([4, 3], dtype=jnp.uint32)
    expected_value = jnp.array([4.0, 3.0], dtype=jnp.float32)
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_take_2d_axis_0():
    """Test take on 2D dataclass along axis 0."""
    data = VectorData.default((3, 2))
    data = data.replace(
        position=jnp.array(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
            ],
            dtype=jnp.float32,
        ),
        velocity=jnp.array(
            [
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
                [[1.3, 1.4, 1.5], [1.6, 1.7, 1.8]],
            ],
            dtype=jnp.float32,
        ),
    )

    indices = jnp.array([0, 2])
    result = xnp.take(data, indices, axis=0)

    expected_position = jnp.array(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]],
        dtype=jnp.float32,
    )
    expected_velocity = jnp.array(
        [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[1.3, 1.4, 1.5], [1.6, 1.7, 1.8]]],
        dtype=jnp.float32,
    )

    assert jnp.array_equal(result.position, expected_position)
    assert jnp.array_equal(result.velocity, expected_velocity)


def test_take_2d_axis_1():
    """Test take on 2D dataclass along axis 1."""
    data = VectorData.default((2, 3))
    data = data.replace(
        position=jnp.array(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
            ],
            dtype=jnp.float32,
        ),
        velocity=jnp.array(
            [
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
                [[1.0, 1.1, 1.2], [1.3, 1.4, 1.5], [1.6, 1.7, 1.8]],
            ],
            dtype=jnp.float32,
        ),
    )

    indices = jnp.array([0, 2])
    result = xnp.take(data, indices, axis=1)

    expected_position = jnp.array(
        [[[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]], [[10.0, 11.0, 12.0], [16.0, 17.0, 18.0]]],
        dtype=jnp.float32,
    )
    expected_velocity = jnp.array(
        [[[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]], [[1.0, 1.1, 1.2], [1.6, 1.7, 1.8]]],
        dtype=jnp.float32,
    )

    assert jnp.array_equal(result.position, expected_position)
    assert jnp.array_equal(result.velocity, expected_velocity)


def test_take_single_element():
    """Test take with single element."""
    data = SimpleData.default((3,))
    data = data.replace(
        id=jnp.array([1, 2, 3], dtype=jnp.uint32),
        value=jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
    )

    indices = jnp.array([1])
    result = xnp.take(data, indices)

    expected_id = jnp.array([2], dtype=jnp.uint32)
    expected_value = jnp.array([2.0], dtype=jnp.float32)
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_take_empty_indices():
    """Test take with empty indices."""
    data = SimpleData.default((3,))
    data = data.replace(
        id=jnp.array([1, 2, 3], dtype=jnp.uint32),
        value=jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
    )

    indices = jnp.array([], dtype=jnp.int32)
    result = xnp.take(data, indices)

    expected_id = jnp.array([], dtype=jnp.uint32)
    expected_value = jnp.array([], dtype=jnp.float32)
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_take_equivalent_to_jnp_take():
    """Test that xnp.take produces same result as manual jnp.take."""
    data = SimpleData.default((4,))
    data = data.replace(
        id=jnp.array([1, 2, 3, 4], dtype=jnp.uint32),
        value=jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
    )

    indices = jnp.array([0, 2, 3])

    result_xnp = xnp.take(data, indices)
    result_manual = SimpleData(
        id=jnp.take(data.id, indices), value=jnp.take(data.value, indices)
    )

    assert jnp.array_equal(result_xnp.id, result_manual.id)
    assert jnp.array_equal(result_xnp.value, result_manual.value)


def test_take_along_axis_axis1():
    """Test take_along_axis along axis 1 matches jnp.take_along_axis."""
    data = SimpleData.default((2, 3))
    data = data.replace(
        id=jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.uint32),
        value=jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32),
    )
    indices = jnp.array([[2, 1, 0], [0, 2, 1]], dtype=jnp.int32)

    result = xnp.take_along_axis(data, indices, axis=1)

    expected_id = jnp.take_along_axis(data.id, indices, axis=1)
    expected_value = jnp.take_along_axis(data.value, indices, axis=1)

    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_take_along_axis_axis0():
    """Test take_along_axis along axis 0 with non-trivial index pattern."""
    data = SimpleData.default((3, 2))
    data = data.replace(
        id=jnp.array([[1, 2], [3, 4], [5, 6]], dtype=jnp.uint32),
        value=jnp.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=jnp.float32),
    )
    indices = jnp.array([[2, 0], [1, 2]], dtype=jnp.int32)

    result = xnp.take_along_axis(data, indices, axis=0)

    expected_id = jnp.take_along_axis(data.id, indices, axis=0)
    expected_value = jnp.take_along_axis(data.value, indices, axis=0)

    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_take_along_axis_broadcasts_field_dims_axis1():
    """Indices without field dims should broadcast across each leaf."""
    data = VectorData.default((2, 4))
    position = jnp.arange(2 * 4 * 3, dtype=jnp.float32).reshape(2, 4, 3)
    velocity = position + 0.5
    data = data.replace(position=position, velocity=velocity)

    indices = jnp.array([[3, 2, 1, 0], [0, 1, 2, 3]], dtype=jnp.int32)

    result = xnp.take_along_axis(data, indices, axis=1)

    broadcast_idx = jnp.broadcast_to(indices[..., None], data.position.shape)
    expected_position = jnp.take_along_axis(data.position, broadcast_idx, axis=1)
    expected_velocity = jnp.take_along_axis(data.velocity, broadcast_idx, axis=1)

    assert jnp.array_equal(result.position, expected_position)
    assert jnp.array_equal(result.velocity, expected_velocity)


def test_take_along_axis_broadcasts_field_dims_axis0():
    """Broadcast indices for fields when gathering along axis 0."""
    data = VectorData.default((3, 2))
    position = jnp.arange(3 * 2 * 3, dtype=jnp.float32).reshape(3, 2, 3)
    velocity = position - 1.0
    data = data.replace(position=position, velocity=velocity)

    indices = jnp.array([[2, 0], [1, 1]], dtype=jnp.int32)

    result = xnp.take_along_axis(data, indices, axis=0)

    target_shape = (indices.shape[0],) + data.position.shape[1:]
    broadcast_idx = jnp.broadcast_to(indices[..., None], target_shape)
    expected_position = jnp.take_along_axis(data.position, broadcast_idx, axis=0)
    expected_velocity = jnp.take_along_axis(data.velocity, broadcast_idx, axis=0)

    assert jnp.array_equal(result.position, expected_position)
    assert jnp.array_equal(result.velocity, expected_velocity)
