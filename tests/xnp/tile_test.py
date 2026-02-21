"""Tests ensuring xnp.tile behaves like numpy.tile across dataclasses."""

import jax.numpy as jnp

from tests.xnp.shared_data import NestedData, SimpleData, VectorData
from xtructure import numpy as xnp


def test_tile_single_dataclass():
    """Test tiling a single dataclass to create a batch."""
    data = SimpleData.default()
    data = data.replace(id=jnp.array(42), value=jnp.array(3.14))

    result = xnp.tile(data, 3)

    assert result.structured_type.name == "BATCHED"
    assert result.shape.batch == (3,)
    assert jnp.array_equal(result.id, jnp.array([42, 42, 42], dtype=jnp.uint32))
    assert jnp.array_equal(
        result.value, jnp.array([3.14, 3.14, 3.14], dtype=jnp.float32)
    )


def test_tile_batched_dataclass_1d():
    """Test tiling a 1D batched dataclass."""
    data = SimpleData.default((2,))
    data = data.replace(
        id=jnp.array([1, 2], dtype=jnp.uint32),
        value=jnp.array([1.0, 2.0], dtype=jnp.float32),
    )

    result = xnp.tile(data, 2)

    assert result.structured_type.name == "BATCHED"
    assert result.shape.batch == (4,)
    assert jnp.array_equal(result.id, jnp.array([1, 2, 1, 2], dtype=jnp.uint32))
    assert jnp.array_equal(
        result.value, jnp.array([1.0, 2.0, 1.0, 2.0], dtype=jnp.float32)
    )


def test_tile_batched_dataclass_2d():
    """Test tiling a 2D batched dataclass."""
    data = SimpleData.default((2, 2))
    data = data.replace(
        id=jnp.array([[1, 2], [3, 4]], dtype=jnp.uint32),
        value=jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
    )

    result = xnp.tile(data, (2, 3))

    assert result.structured_type.name == "BATCHED"
    assert result.shape.batch == (4, 6)
    expected_id = jnp.array(
        [
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
        ],
        dtype=jnp.uint32,
    )
    expected_value = expected_id.astype(jnp.float32)
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_tile_vector_dataclass():
    """Test tiling a dataclass with vector fields."""
    data = VectorData.default((2,))
    data = data.replace(
        position=jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        velocity=jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
    )

    result = xnp.tile(data, 2)

    assert result.position.shape == (2, 6)
    assert result.velocity.shape == (2, 6)
    expected_position = jnp.array(
        [
            [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 4.0, 5.0, 6.0],
        ]
    )
    expected_velocity = jnp.array(
        [
            [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6, 0.4, 0.5, 0.6],
        ]
    )
    assert jnp.allclose(result.position, expected_position)
    assert jnp.allclose(result.velocity, expected_velocity)


def test_tile_complex_repetition_pattern():
    """Test tiling with complex repetition patterns."""
    data = SimpleData.default((2, 3))
    data = data.replace(
        id=jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.uint32),
        value=jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32),
    )

    result = xnp.tile(data, (1, 2, 1))

    assert result.structured_type.name == "BATCHED"
    assert result.shape.batch == (1, 4, 3)
    expected_id = jnp.array(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [1, 2, 3],
                [4, 5, 6],
            ]
        ],
        dtype=jnp.uint32,
    )
    expected_value = expected_id.astype(jnp.float32)
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_tile_nested_dataclass():
    """Test tiling a nested dataclass."""
    simple = SimpleData.default()
    simple = simple.replace(id=jnp.array(1), value=jnp.array(1.0))
    vector = VectorData.default()
    vector = vector.replace(
        position=jnp.array([1.0, 2.0, 3.0]), velocity=jnp.array([0.1, 0.2, 0.3])
    )
    data = NestedData.default()
    data = data.replace(simple=simple, vector=vector)

    result = xnp.tile(data, 2)

    assert result.simple.id.shape == (2,)
    assert result.simple.value.shape == (2,)
    assert result.vector.position.shape == (6,)
    assert result.vector.velocity.shape == (6,)

    assert jnp.array_equal(result.simple.id, jnp.array([1, 1], dtype=jnp.uint32))
    assert jnp.array_equal(
        result.simple.value, jnp.array([1.0, 1.0], dtype=jnp.float32)
    )
    assert jnp.allclose(
        result.vector.position, jnp.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    )
    assert jnp.allclose(
        result.vector.velocity, jnp.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3])
    )


def test_tile_equivalent_to_jnp_tile():
    """Test that xnp.tile produces same result as manual jnp.tile."""
    data = SimpleData.default((2, 2))
    data = data.replace(
        id=jnp.array([[1, 2], [3, 4]], dtype=jnp.uint32),
        value=jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
    )

    reps = (2, 3)

    result_xnp = xnp.tile(data, reps)
    result_manual = SimpleData(
        id=jnp.tile(data.id, reps), value=jnp.tile(data.value, reps)
    )

    assert jnp.array_equal(result_xnp.id, result_manual.id)
    assert jnp.array_equal(result_xnp.value, result_manual.value)


def test_tile_single_repetition():
    """Test tiling with single integer repetition."""
    data = SimpleData.default((2,))
    data = data.replace(
        id=jnp.array([1, 2], dtype=jnp.uint32),
        value=jnp.array([1.0, 2.0], dtype=jnp.float32),
    )

    result = xnp.tile(data, 3)

    assert result.shape.batch == (6,)
    assert jnp.array_equal(result.id, jnp.array([1, 2, 1, 2, 1, 2], dtype=jnp.uint32))
    assert jnp.array_equal(
        result.value, jnp.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0], dtype=jnp.float32)
    )


def test_tile_tuple_repetition():
    """Test tiling with tuple repetition."""
    data = SimpleData.default((2,))
    data = data.replace(
        id=jnp.array([1, 2], dtype=jnp.uint32),
        value=jnp.array([1.0, 2.0], dtype=jnp.float32),
    )

    result = xnp.tile(data, (2, 3))

    assert result.shape.batch == (2, 6)
    expected_id = jnp.array(
        [
            [1, 2, 1, 2, 1, 2],
            [1, 2, 1, 2, 1, 2],
        ],
        dtype=jnp.uint32,
    )
    expected_value = expected_id.astype(jnp.float32)
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_tile_empty_batch():
    """Test tiling with empty batch."""
    data = SimpleData.default((0,))

    result = xnp.tile(data, 3)

    assert result.shape.batch == (0,)
    assert jnp.array_equal(result.id, jnp.array([], dtype=jnp.uint32))
    assert jnp.array_equal(result.value, jnp.array([], dtype=jnp.float32))


def test_tile_zero_repetition():
    """Test tiling with zero repetition."""
    data = SimpleData.default((2,))
    data = data.replace(
        id=jnp.array([1, 2], dtype=jnp.uint32),
        value=jnp.array([1.0, 2.0], dtype=jnp.float32),
    )

    result = xnp.tile(data, 0)

    assert result.shape.batch == (0,)
    assert jnp.array_equal(result.id, jnp.array([], dtype=jnp.uint32))
    assert jnp.array_equal(result.value, jnp.array([], dtype=jnp.float32))


def test_tile_mixed_zero_repetition():
    """Test tiling with mixed zero and non-zero repetition."""
    data = SimpleData.default((2, 2))
    data = data.replace(
        id=jnp.array([[1, 2], [3, 4]], dtype=jnp.uint32),
        value=jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
    )

    result = xnp.tile(data, (2, 0))

    assert result.shape.batch == (4, 0)
    assert jnp.array_equal(result.id, jnp.array([], dtype=jnp.uint32).reshape(4, 0))
    assert jnp.array_equal(result.value, jnp.array([], dtype=jnp.float32).reshape(4, 0))


def test_tile_integration_with_other_ops():
    """Test tile integration with other xnp operations."""
    data = SimpleData.default((2,))
    data = data.replace(
        id=jnp.array([1, 2], dtype=jnp.uint32),
        value=jnp.array([1.0, 2.0], dtype=jnp.float32),
    )

    tiled = xnp.tile(data, 3)
    assert tiled.shape.batch == (6,)

    taken = xnp.take(tiled, jnp.array([0, 2, 4]))
    assert taken.shape.batch == (3,)
    assert jnp.array_equal(taken.id, jnp.array([1, 1, 1], dtype=jnp.uint32))

    condition = jnp.array([True, False, True, False, True, False])
    fallback_value = jnp.array(999, dtype=jnp.uint32)
    filtered = xnp.where(condition, tiled, fallback_value)
    assert filtered.shape.batch == (6,)
    expected_id = jnp.array([1, 999, 1, 999, 1, 999], dtype=jnp.uint32)
    assert jnp.array_equal(filtered.id, expected_id)
