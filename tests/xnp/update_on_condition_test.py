"""Tests covering the `xnp.update_on_condition` helper."""

import jax.numpy as jnp
import pytest

from tests.xnp.shared_data import NestedData, SimpleData, VectorData
from xtructure import numpy as xnp


def _make_simple_default(shape):
    instance = SimpleData.default(shape)
    return instance.replace(
        id=jnp.zeros(instance.shape.batch, dtype=jnp.uint32),
        value=jnp.zeros(instance.shape.batch, dtype=jnp.float32),
    )


def test_update_on_condition_basic():
    """Test update_on_condition with a simple dataclass."""
    original = _make_simple_default((5,))
    indices = jnp.array([0, 2, 4])
    condition = jnp.array([True, True, True])
    values_to_set = 1.0

    result = xnp.update_on_condition(original, indices, condition, values_to_set)

    expected_id = jnp.array([1, 0, 1, 0, 1], dtype=jnp.uint32)
    expected_value = jnp.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=jnp.float32)
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_update_on_condition_duplicate_indices_first_wins():
    """Test update_on_condition with duplicate indices - first True wins."""
    original = _make_simple_default((5,))
    indices = jnp.array([0, 2, 0])
    condition = jnp.array([True, True, True])
    values_to_set = jnp.array([1.0, 2.0, 3.0])

    result = xnp.update_on_condition(original, indices, condition, values_to_set)

    expected_id = jnp.array([1, 0, 2, 0, 0], dtype=jnp.uint32)
    expected_value = jnp.array([1.0, 0.0, 2.0, 0.0, 0.0], dtype=jnp.float32)
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_update_on_condition_advanced_indexing():
    """Test update_on_condition with advanced indexing."""
    original = VectorData.default((2, 3))
    original = original.replace(
        position=jnp.zeros((2, 3, 3), dtype=jnp.float32),
        velocity=jnp.zeros((2, 3, 3), dtype=jnp.float32),
    )
    indices = (jnp.array([0, 1, 0]), jnp.array([1, 2, 1]))
    condition = jnp.array([True, True, False])
    values_to_set = 5.0

    result = xnp.update_on_condition(original, indices, condition, values_to_set)

    expected_position = jnp.array(
        [
            [[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [5.0, 5.0, 5.0]],
        ],
        dtype=jnp.float32,
    )
    expected_velocity = expected_position.copy()

    assert jnp.array_equal(result.position, expected_position)
    assert jnp.array_equal(result.velocity, expected_velocity)


def test_update_on_condition_all_false():
    """Test update_on_condition when all conditions are False."""
    original = SimpleData.default((5,))
    original = original.replace(
        id=jnp.arange(5, dtype=jnp.uint32), value=jnp.arange(5, dtype=jnp.float32)
    )
    indices = jnp.array([0, 1, 2])
    condition = jnp.array([False, False, False])
    values_to_set = 99.0

    result = xnp.update_on_condition(original, indices, condition, values_to_set)

    assert jnp.array_equal(result.id, original.id)
    assert jnp.array_equal(result.value, original.value)


def test_update_on_condition_scalar_value():
    """Test update_on_condition with scalar values."""
    original = SimpleData.default((4,))
    original = original.replace(
        id=jnp.ones(4, dtype=jnp.uint32), value=jnp.ones(4, dtype=jnp.float32)
    )
    indices = jnp.array([1, 3])
    condition = jnp.array([True, True])
    values_to_set = 7.0

    result = xnp.update_on_condition(original, indices, condition, values_to_set)

    expected_id = jnp.array([1, 7, 1, 7], dtype=jnp.uint32)
    expected_value = jnp.array([1.0, 7.0, 1.0, 7.0], dtype=jnp.float32)
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_update_on_condition_array_values():
    """Test update_on_condition with array values."""
    original = _make_simple_default((5,))
    indices = jnp.array([0, 2, 4, 0])
    condition = jnp.array([True, True, False, True])
    values_to_set = jnp.array([10.0, 20.0, 30.0, 40.0])

    result = xnp.update_on_condition(original, indices, condition, values_to_set)

    expected_id = jnp.array([10, 0, 20, 0, 0], dtype=jnp.uint32)
    expected_value = jnp.array([10.0, 0.0, 20.0, 0.0, 0.0], dtype=jnp.float32)
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_update_on_condition_dataclass_values_first_true_wins():
    """Dataclass values respect first-true-wins semantics across fields."""
    original = _make_simple_default((4,))
    indices = jnp.array([2, 2, 1, 2], dtype=jnp.int32)
    condition = jnp.array([False, True, True, True])

    updates = SimpleData.default((4,))
    updates = updates.replace(
        id=jnp.array([10, 20, 30, 40], dtype=jnp.uint32),
        value=jnp.array([-1.0, -2.0, -3.0, -4.0], dtype=jnp.float32),
    )

    result = xnp.update_on_condition(original, indices, condition, updates)

    expected_id = jnp.array([0, 30, 20, 0], dtype=jnp.uint32)
    expected_value = jnp.array([0.0, -3.0, -2.0, 0.0], dtype=jnp.float32)
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_update_on_condition_advanced_indices_duplicate_first_true():
    """Advanced tuple indices still enforce first true wins after flattening."""
    original = VectorData.default((2, 2))
    original = original.replace(
        position=jnp.zeros((2, 2, 3), dtype=jnp.float32),
        velocity=jnp.zeros((2, 2, 3), dtype=jnp.float32),
    )

    row = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    col = jnp.array([1, 1, 0, 0], dtype=jnp.int32)
    condition = jnp.array([False, True, True, True])
    values_to_set = jnp.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
        ],
        dtype=jnp.float32,
    )

    result = xnp.update_on_condition(original, (row, col), condition, values_to_set)

    expected = jnp.zeros((2, 2, 3), dtype=jnp.float32)
    expected = expected.at[0, 1].set(values_to_set[1])
    expected = expected.at[1, 0].set(values_to_set[2])
    assert jnp.array_equal(result.position, expected)
    assert jnp.array_equal(result.velocity, expected)


def test_update_on_condition_nested_dataclass_first_true_wins_everywhere():
    """Nested dataclasses update every leaf with the same first-true policy."""
    base_simple = SimpleData.default((3,))
    base_simple = base_simple.replace(
        id=jnp.zeros(3, dtype=jnp.uint32), value=jnp.zeros(3, dtype=jnp.float32)
    )
    base_vector = VectorData.default((3,))
    base_vector = base_vector.replace(
        position=jnp.zeros((3, 3), dtype=jnp.float32),
        velocity=jnp.zeros((3, 3), dtype=jnp.float32),
    )
    original = NestedData.default((3,))
    original = original.replace(simple=base_simple, vector=base_vector)

    indices = jnp.array([2, 0, 2], dtype=jnp.int32)
    condition = jnp.array([True, True, True])

    updates_simple = SimpleData.default((3,))
    updates_simple = updates_simple.replace(
        id=jnp.array([11, 22, 33], dtype=jnp.uint32),
        value=jnp.array([1.1, 2.2, 3.3], dtype=jnp.float32),
    )
    updates_vector = VectorData.default((3,))
    updates_vector = updates_vector.replace(
        position=jnp.array(
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=jnp.float32
        ),
        velocity=jnp.array(
            [[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]], dtype=jnp.float32
        ),
    )
    updates = NestedData.default((3,))
    updates = updates.replace(simple=updates_simple, vector=updates_vector)

    result = xnp.update_on_condition(original, indices, condition, updates)

    assert jnp.array_equal(result.simple.id, jnp.array([22, 0, 11], dtype=jnp.uint32))
    assert jnp.allclose(
        result.simple.value, jnp.array([2.2, 0.0, 1.1], dtype=jnp.float32)
    )

    expected_position = jnp.zeros((3, 3), dtype=jnp.float32)
    expected_position = expected_position.at[0].set(updates_vector.position[1])
    expected_position = expected_position.at[2].set(updates_vector.position[0])
    expected_velocity = jnp.zeros((3, 3), dtype=jnp.float32)
    expected_velocity = expected_velocity.at[0].set(updates_vector.velocity[1])
    expected_velocity = expected_velocity.at[2].set(updates_vector.velocity[0])
    assert jnp.allclose(result.vector.position, expected_position)
    assert jnp.allclose(result.vector.velocity, expected_velocity)


def test_update_on_condition_array_basic():
    """Array inputs follow the same first-true-wins semantics."""
    original = jnp.zeros(5, dtype=jnp.float32)
    indices = jnp.array([0, 2, 4])
    condition = jnp.array([True, True, True])
    values_to_set = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)

    result = xnp.update_on_condition(original, indices, condition, values_to_set)

    expected = jnp.array([1.0, 0.0, 2.0, 0.0, 3.0], dtype=jnp.float32)
    assert jnp.array_equal(result, expected)


def test_update_on_condition_array_duplicate_indices_first_true():
    """Array inputs with duplicate indices keep the first True update."""
    original = jnp.zeros(3, dtype=jnp.float32)
    indices = jnp.array([1, 1, 1], dtype=jnp.int32)
    condition = jnp.array([True, True, True])
    values_to_set = jnp.array([9.0, 8.0, 7.0], dtype=jnp.float32)

    result = xnp.update_on_condition(original, indices, condition, values_to_set)

    expected = jnp.array([0.0, 9.0, 0.0], dtype=jnp.float32)
    assert jnp.array_equal(result, expected)


def test_update_on_condition_array_advanced_indices():
    """Tuple indices work for array inputs too."""
    original = jnp.zeros((2, 2), dtype=jnp.float32)
    row = jnp.array([0, 1, 1], dtype=jnp.int32)
    col = jnp.array([1, 0, 1], dtype=jnp.int32)
    condition = jnp.array([True, False, True])
    values_to_set = jnp.array([5.0, 6.0, 7.0], dtype=jnp.float32)

    result = xnp.update_on_condition(original, (row, col), condition, values_to_set)

    expected = jnp.zeros((2, 2), dtype=jnp.float32)
    expected = expected.at[0, 1].set(5.0)
    expected = expected.at[1, 1].set(7.0)
    assert jnp.array_equal(result, expected)


def test_update_on_condition_structure_mismatch_raises():
    """Different dataclass structures should raise before attempting updates."""
    original = SimpleData.default((2,))
    indices = jnp.array([0, 1], dtype=jnp.int32)
    condition = jnp.array([True, True])
    values_to_set = VectorData.default((2,))

    with pytest.raises((ValueError, TypeError)):
        xnp.update_on_condition(original, indices, condition, values_to_set)


def test_update_on_condition_condition_shape_mismatch_raises():
    """Condition arrays must match the flattened indices shape exactly."""
    original = SimpleData.default((3,))
    original = original.replace(
        id=jnp.arange(3, dtype=jnp.uint32), value=jnp.arange(3, dtype=jnp.float32)
    )
    indices = jnp.array([0, 1, 2], dtype=jnp.int32)
    condition = jnp.array([[True, False, True]], dtype=jnp.bool_)

    with pytest.raises(
        ValueError, match="`condition` shape .* must match `indices` shape"
    ):
        xnp.update_on_condition(original, indices, condition, 5.0)


def test_update_on_condition_non_broadcastable_values_raise():
    """Values that cannot broadcast to the update shape should fail loudly."""
    original = _make_simple_default((2,))
    indices = jnp.array([0, 1], dtype=jnp.int32)
    condition = jnp.array([True, True])
    bad_values = jnp.array([[9.0, 10.0]], dtype=jnp.float32)

    with pytest.raises((ValueError, TypeError)):
        xnp.update_on_condition(original, indices, condition, bad_values)


def test_update_on_condition_empty_condition_noop():
    """Empty update requests should leave the dataclass untouched."""
    original = SimpleData.default((2,))
    original = original.replace(
        id=jnp.array([5, 6], dtype=jnp.uint32),
        value=jnp.array([1.5, 2.5], dtype=jnp.float32),
    )
    indices = jnp.array([], dtype=jnp.int32)
    condition = jnp.array([], dtype=jnp.bool_)

    result = xnp.update_on_condition(original, indices, condition, -1.0)

    assert jnp.array_equal(result.id, original.id)
    assert jnp.array_equal(result.value, original.value)
