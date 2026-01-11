"""Tests for xnp.transpose and xnp.swapaxes helpers."""

import jax.numpy as jnp

from tests.xnp.shared_data import NestedData, SimpleData, VectorData
from xtructure import numpy as xnp


def test_transpose_2d_basic():
    """Test basic transpose on 2D dataclass."""
    data = SimpleData.default(shape=(2, 3))
    data = data.replace(
        id=jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.uint32),
        value=jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32),
    )

    result = xnp.transpose(data)

    assert result.shape.batch == (3, 2)
    expected_id = jnp.array([[1, 4], [2, 5], [3, 6]], dtype=jnp.uint32)
    expected_value = jnp.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=jnp.float32)
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_transpose_3d_basic():
    """Test basic transpose on 3D dataclass."""
    data = SimpleData.default(shape=(2, 3, 4))
    data = data.replace(
        id=jnp.arange(24, dtype=jnp.uint32).reshape(2, 3, 4),
        value=jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4),
    )

    result = xnp.transpose(data)

    assert result.shape.batch == (4, 3, 2)

    expected_id = jnp.transpose(jnp.arange(24, dtype=jnp.uint32).reshape(2, 3, 4))
    expected_value = jnp.transpose(jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4))
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_transpose_with_custom_axes():
    """Test transpose with custom axes order."""
    data = SimpleData.default(shape=(2, 3, 4))
    data = data.replace(
        id=jnp.arange(24, dtype=jnp.uint32).reshape(2, 3, 4),
        value=jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4),
    )

    result = xnp.transpose(data, axes=(2, 0, 1))

    assert result.shape.batch == (4, 2, 3)

    expected_id = jnp.transpose(jnp.arange(24, dtype=jnp.uint32).reshape(2, 3, 4), axes=(2, 0, 1))
    expected_value = jnp.transpose(
        jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4), axes=(2, 0, 1)
    )
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_transpose_vector_dataclass():
    """Test transpose on dataclass with vector fields."""
    data = VectorData.default(shape=(2, 3))
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

    result = xnp.transpose(data)

    assert result.position.shape == (3, 2, 3)
    assert result.velocity.shape == (3, 2, 3)
    assert result.position.shape[2] == 3
    assert result.velocity.shape[2] == 3


def test_transpose_1d_no_change():
    """Test transpose on 1D dataclass (should be no-op)."""
    data = SimpleData.default(shape=(3,))
    data = data.replace(
        id=jnp.array([1, 2, 3], dtype=jnp.uint32),
        value=jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
    )

    result = xnp.transpose(data)

    assert result.shape.batch == (3,)
    assert jnp.array_equal(result.id, data.id)
    assert jnp.array_equal(result.value, data.value)


def test_transpose_equivalent_to_jnp_transpose():
    """Test that xnp.transpose produces same result as manual jnp.transpose."""
    data = SimpleData.default(shape=(2, 3))
    data = data.replace(
        id=jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.uint32),
        value=jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32),
    )

    result_xnp = xnp.transpose(data)
    result_manual = SimpleData(id=jnp.transpose(data.id), value=jnp.transpose(data.value))

    assert jnp.array_equal(result_xnp.id, result_manual.id)
    assert jnp.array_equal(result_xnp.value, result_manual.value)


def test_transpose_with_axes_equivalent_to_jnp():
    """Test that xnp.transpose with axes produces same result as manual jnp.transpose."""
    data = SimpleData.default(shape=(2, 3, 4))
    data = data.replace(
        id=jnp.arange(24, dtype=jnp.uint32).reshape(2, 3, 4),
        value=jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4),
    )

    axes = (2, 0, 1)

    result_xnp = xnp.transpose(data, axes=axes)
    result_manual = SimpleData(
        id=jnp.transpose(data.id, axes=axes), value=jnp.transpose(data.value, axes=axes)
    )

    assert jnp.array_equal(result_xnp.id, result_manual.id)
    assert jnp.array_equal(result_xnp.value, result_manual.value)


def test_transpose_nested_dataclass():
    """Test transpose on nested dataclass."""
    simple = SimpleData.default(shape=(2, 2))
    simple = simple.replace(
        id=jnp.array([[1, 2], [3, 4]], dtype=jnp.uint32),
        value=jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
    )
    vector = VectorData.default(shape=(2, 2))
    vector = vector.replace(
        position=jnp.array(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ],
            dtype=jnp.float32,
        ),
        velocity=jnp.array(
            [
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
            ],
            dtype=jnp.float32,
        ),
    )
    data = NestedData.default(shape=(2, 2))
    data = data.replace(simple=simple, vector=vector)

    result = xnp.transpose(data)

    assert result.simple.id.shape == (2, 2)
    assert result.simple.value.shape == (2, 2)
    assert result.vector.position.shape == (2, 2, 3)
    assert result.vector.velocity.shape == (2, 2, 3)

    expected_simple_id = jnp.transpose(data.simple.id)
    expected_simple_value = jnp.transpose(data.simple.value)
    assert jnp.array_equal(result.simple.id, expected_simple_id)
    assert jnp.array_equal(result.simple.value, expected_simple_value)


def test_swap_axes_2d_basic():
    """Test basic swapaxes on 2D dataclass."""
    data = SimpleData.default(shape=(2, 3))
    data = data.replace(
        id=jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.uint32),
        value=jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32),
    )

    result = xnp.swapaxes(data, 0, 1)

    assert result.shape.batch == (3, 2)
    expected_id = jnp.array([[1, 4], [2, 5], [3, 6]], dtype=jnp.uint32)
    expected_value = jnp.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=jnp.float32)
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_swap_axes_3d_basic():
    """Test basic swapaxes on 3D dataclass."""
    data = SimpleData.default(shape=(2, 3, 4))
    data = data.replace(
        id=jnp.arange(24, dtype=jnp.uint32).reshape(2, 3, 4),
        value=jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4),
    )

    result = xnp.swapaxes(data, 0, 2)

    assert result.shape.batch == (4, 3, 2)

    expected_id = jnp.swapaxes(jnp.arange(24, dtype=jnp.uint32).reshape(2, 3, 4), 0, 2)
    expected_value = jnp.swapaxes(jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4), 0, 2)
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_swap_axes_with_negative_indices():
    """Test swapaxes with negative indices."""
    data = SimpleData.default(shape=(2, 3, 4))
    data = data.replace(
        id=jnp.arange(24, dtype=jnp.uint32).reshape(2, 3, 4),
        value=jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4),
    )

    result = xnp.swapaxes(data, -1, -2)

    assert result.shape.batch == (2, 4, 3)

    expected_id = jnp.swapaxes(jnp.arange(24, dtype=jnp.uint32).reshape(2, 3, 4), -1, -2)
    expected_value = jnp.swapaxes(jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4), -1, -2)
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_swap_axes_mixed_positive_negative():
    """Test swapaxes with mixed positive and negative indices."""
    data = SimpleData.default(shape=(2, 3, 4))
    data = data.replace(
        id=jnp.arange(24, dtype=jnp.uint32).reshape(2, 3, 4),
        value=jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4),
    )

    result = xnp.swapaxes(data, 1, -1)

    assert result.shape.batch == (2, 4, 3)

    expected_id = jnp.swapaxes(jnp.arange(24, dtype=jnp.uint32).reshape(2, 3, 4), 1, -1)
    expected_value = jnp.swapaxes(jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4), 1, -1)
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_swap_axes_vector_dataclass():
    """Test swapaxes on dataclass with vector fields."""
    data = VectorData.default(shape=(2, 3))
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

    result = xnp.swapaxes(data, 0, 1)

    assert result.shape.batch == (3, 2)
    expected_position = jnp.array(
        [
            [[1.0, 2.0, 3.0], [10.0, 11.0, 12.0]],
            [[4.0, 5.0, 6.0], [13.0, 14.0, 15.0]],
            [[7.0, 8.0, 9.0], [16.0, 17.0, 18.0]],
        ],
        dtype=jnp.float32,
    )
    expected_velocity = jnp.array(
        [
            [[0.1, 0.2, 0.3], [1.0, 1.1, 1.2]],
            [[0.4, 0.5, 0.6], [1.3, 1.4, 1.5]],
            [[0.7, 0.8, 0.9], [1.6, 1.7, 1.8]],
        ],
        dtype=jnp.float32,
    )
    assert jnp.array_equal(result.position, expected_position)
    assert jnp.array_equal(result.velocity, expected_velocity)


def test_swap_axes_1d_no_change():
    """Test swapaxes on 1D dataclass (should be no-op)."""
    data = SimpleData.default(shape=(3,))
    data = data.replace(
        id=jnp.array([1, 2, 3], dtype=jnp.uint32),
        value=jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
    )

    result = xnp.swapaxes(data, 0, 0)

    assert result.shape.batch == (3,)
    assert jnp.array_equal(result.id, data.id)
    assert jnp.array_equal(result.value, data.value)


def test_swap_axes_equivalent_to_jnp_swapaxes():
    """Test that xnp.swapaxes produces same result as manual jnp.swapaxes."""
    data = SimpleData.default(shape=(2, 3))
    data = data.replace(
        id=jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.uint32),
        value=jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32),
    )

    result_xnp = xnp.swapaxes(data, 0, 1)
    result_manual = SimpleData(id=jnp.swapaxes(data.id, 0, 1), value=jnp.swapaxes(data.value, 0, 1))

    assert jnp.array_equal(result_xnp.id, result_manual.id)
    assert jnp.array_equal(result_xnp.value, result_manual.value)


def test_swap_axes_nested_dataclass():
    """Test swapaxes on nested dataclass."""
    simple = SimpleData.default(shape=(2, 2))
    simple = simple.replace(
        id=jnp.array([[1, 2], [3, 4]], dtype=jnp.uint32),
        value=jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
    )
    vector = VectorData.default(shape=(2, 2))
    vector = vector.replace(
        position=jnp.array(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ],
            dtype=jnp.float32,
        ),
        velocity=jnp.array(
            [
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
            ],
            dtype=jnp.float32,
        ),
    )
    data = NestedData.default(shape=(2, 2))
    data = data.replace(simple=simple, vector=vector)

    result = xnp.swapaxes(data, 0, 1)

    assert result.shape.batch == (2, 2)
    expected_simple_id = jnp.array([[1, 3], [2, 4]], dtype=jnp.uint32)
    expected_simple_value = jnp.array([[1.0, 3.0], [2.0, 4.0]], dtype=jnp.float32)
    assert jnp.array_equal(result.simple.id, expected_simple_id)
    assert jnp.array_equal(result.simple.value, expected_simple_value)


def test_swap_axes_integration_with_other_ops():
    """Test swapaxes integration with other xnp operations."""
    data = SimpleData.default(shape=(2, 3))
    data = data.replace(
        id=jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.uint32),
        value=jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32),
    )

    swapped = xnp.swapaxes(data, 0, 1)
    assert swapped.shape.batch == (3, 2)

    taken = xnp.take(swapped, jnp.array([0, 2]), axis=0)
    assert taken.shape.batch == (2, 2)
    expected_id = jnp.array([[1, 4], [3, 6]], dtype=jnp.uint32)
    assert jnp.array_equal(taken.id, expected_id)

    condition = jnp.array([[True, False], [False, True], [True, True]])
    fallback_value = jnp.array(999, dtype=jnp.uint32)
    filtered = xnp.where(condition, swapped, fallback_value)
    assert filtered.shape.batch == (3, 2)
    expected_filtered_id = jnp.array([[1, 999], [999, 5], [3, 6]], dtype=jnp.uint32)
    assert jnp.array_equal(filtered.id, expected_filtered_id)


def test_transpose_integration_with_other_ops():
    """Test transpose integration with other xnp operations."""
    data = SimpleData.default(shape=(2, 3))
    data = data.replace(
        id=jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.uint32),
        value=jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32),
    )

    transposed = xnp.transpose(data)
    assert transposed.shape.batch == (3, 2)

    taken = xnp.take(transposed, jnp.array([0, 2]), axis=0)
    assert taken.shape.batch == (2, 2)
    expected_id = jnp.array([[1, 4], [3, 6]], dtype=jnp.uint32)
    assert jnp.array_equal(taken.id, expected_id)

    condition = jnp.array([[True, False], [False, True], [True, True]])
    fallback_value = jnp.array(999, dtype=jnp.uint32)
    filtered = xnp.where(condition, transposed, fallback_value)
    assert filtered.shape.batch == (3, 2)
    expected_filtered_id = jnp.array([[1, 999], [999, 5], [3, 6]], dtype=jnp.uint32)
    assert jnp.array_equal(filtered.id, expected_filtered_id)


def test_transpose_swap_axes_equivalence():
    """Test that transpose and swapaxes produce equivalent results for 2D arrays."""
    data = SimpleData.default(shape=(2, 3))
    data = data.replace(
        id=jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.uint32),
        value=jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32),
    )

    result_transpose = xnp.transpose(data)
    result_swap = xnp.swapaxes(data, 0, 1)

    assert jnp.array_equal(result_transpose.id, result_swap.id)
    assert jnp.array_equal(result_transpose.value, result_swap.value)
    assert result_transpose.shape.batch == result_swap.shape.batch
