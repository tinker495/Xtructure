import jax
import jax.numpy as jnp
import pytest

from xtructure import FieldDescriptor, xtructure_dataclass
from xtructure import xtructure_numpy as xnp


# Test dataclasses for the new operations
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


def test_set_as_condition_basic():
    original_array = jnp.zeros(5)
    indices = jnp.array([0, 2, 4])
    condition = jnp.array([True, True, True])
    values_to_set = 1.0
    result = xnp.set_as_condition_on_array(original_array, indices, condition, values_to_set)
    expected = jnp.array([1.0, 0.0, 1.0, 0.0, 1.0])
    assert jnp.array_equal(result, expected)


def test_set_as_condition_duplicate_indices_first_wins():
    original_array = jnp.zeros(5)
    indices = jnp.array([0, 2, 0])
    condition = jnp.array([True, True, True])
    values_to_set = jnp.array([1.0, 2.0, 3.0])
    result = xnp.set_as_condition_on_array(original_array, indices, condition, values_to_set)
    expected = jnp.array([1.0, 0.0, 2.0, 0.0, 0.0])
    assert jnp.array_equal(result, expected)


def test_set_as_condition_advanced_indexing():
    original_array = jnp.zeros((2, 3))
    indices = (jnp.array([0, 1, 0]), jnp.array([1, 2, 1]))
    condition = jnp.array([True, True, False])
    values_to_set = 5.0
    result = xnp.set_as_condition_on_array(original_array, indices, condition, values_to_set)
    expected = jnp.array([[0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
    assert jnp.array_equal(result, expected)


def test_set_as_condition_all_false():
    original_array = jnp.arange(5)
    indices = jnp.array([0, 1, 2])
    condition = jnp.array([False, False, False])
    values_to_set = 99.0
    result = xnp.set_as_condition_on_array(original_array, indices, condition, values_to_set)
    assert jnp.array_equal(result, original_array)


def test_set_as_condition_scalar_value():
    original_array = jnp.ones(4)
    indices = jnp.array([1, 3])
    condition = jnp.array([True, True])
    values_to_set = 7.0
    result = xnp.set_as_condition_on_array(original_array, indices, condition, values_to_set)
    expected = jnp.array([1.0, 7.0, 1.0, 7.0])
    assert jnp.array_equal(result, expected)


def test_set_as_condition_array_values():
    original_array = jnp.zeros(5)
    indices = jnp.array([0, 2, 4, 0])
    condition = jnp.array([True, True, False, True])
    values_to_set = jnp.array([10.0, 20.0, 30.0, 40.0])
    result = xnp.set_as_condition_on_array(original_array, indices, condition, values_to_set)
    expected = jnp.array([10.0, 0.0, 20.0, 0.0, 0.0])
    assert jnp.array_equal(result, expected)


# Tests for concat function
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


# Tests for pad function
def test_pad_single_to_batched():
    """Test padding a SINGLE dataclass to create a batched version."""
    data = SimpleData.default()
    data = data.replace(id=jnp.array(42), value=jnp.array(3.14))

    result = xnp.pad(data, (0, 4))

    assert result.structured_type.name == "BATCHED"
    assert result.shape.batch == (5,)
    assert jnp.array_equal(result.id, jnp.array([42, 42, 42, 42, 42]))
    assert jnp.allclose(result.value, jnp.array([3.14, 3.14, 3.14, 3.14, 3.14]))


def test_pad_batched_axis_0():
    """Test padding a BATCHED dataclass along axis 0."""
    data = SimpleData.default(shape=(3,))
    data = data.replace(id=jnp.array([1, 2, 3], dtype=jnp.uint32), value=jnp.array([1.0, 2.0, 3.0]))

    result = xnp.pad(data, (0, 2))

    assert result.structured_type.name == "BATCHED"
    assert result.shape.batch == (5,)
    # Use the actual default fill values: uint32 max value and float32 inf
    expected_id = jnp.array([1, 2, 3, 4294967295, 4294967295], dtype=jnp.uint32)
    expected_value = jnp.array([1.0, 2.0, 3.0, jnp.inf, jnp.inf], dtype=jnp.float32)
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_pad_uses_existing_padding_as_batch():
    """Test that pad function uses the existing padding_as_batch method when appropriate."""
    data = SimpleData.default(shape=(2,))
    data = data.replace(id=jnp.array([1, 2], dtype=jnp.uint32), value=jnp.array([1.0, 2.0]))

    # This should use the existing padding_as_batch method
    result_xnp = xnp.pad(data, (0, 2))
    result_builtin = data.padding_as_batch((4,))

    # Results should be identical
    assert jnp.array_equal(result_xnp.id, result_builtin.id)
    assert jnp.array_equal(result_xnp.value, result_builtin.value)
    assert result_xnp.shape.batch == result_builtin.shape.batch


def test_pad_batched_with_constant_values():
    """Test padding with custom constant values."""
    data = SimpleData.default(shape=(2,))
    data = data.replace(id=jnp.array([1, 2]), value=jnp.array([1.0, 2.0]))

    result = xnp.pad(data, (0, 2), constant_values=99)

    assert result.shape.batch == (4,)
    assert jnp.array_equal(result.id, jnp.array([1, 2, 99, 99], dtype=jnp.uint32))
    assert jnp.array_equal(result.value, jnp.array([1.0, 2.0, 99.0, 99.0], dtype=jnp.float32))


def test_pad_batched_target_shape():
    """Test padding to a target batch shape."""
    data = SimpleData.default(shape=(2, 3))

    result = xnp.pad(data, [(0, 2), (0, 2)])

    assert result.shape.batch == (4, 5)


def test_pad_no_change_needed():
    """Test that padding with zero padding returns the same instance."""
    data = SimpleData.default(shape=(3,))
    result = xnp.pad(data, (0, 0))
    assert result is data


def test_pad_invalid_pad_width():
    """Test that invalid pad_width raises ValueError."""
    data = SimpleData.default(shape=(3,))

    # Test with invalid pad_width format
    with pytest.raises(
        ValueError, match="pad_width must be int, sequence of int, or sequence of pairs"
    ):
        xnp.pad(data, "invalid")


# Tests for stack function
def test_stack_single_dataclasses():
    """Test stacking SINGLE structured dataclasses."""
    data1 = SimpleData.default()
    data1 = data1.replace(id=jnp.array(1), value=jnp.array(1.0))
    data2 = SimpleData.default()
    data2 = data2.replace(id=jnp.array(2), value=jnp.array(2.0))

    result = xnp.stack([data1, data2])

    assert result.structured_type.name == "BATCHED"
    assert result.shape.batch == (2,)
    assert jnp.array_equal(result.id, jnp.array([1, 2]))
    assert jnp.array_equal(result.value, jnp.array([1.0, 2.0]))


def test_stack_batched_dataclasses():
    """Test stacking BATCHED structured dataclasses."""
    data1 = SimpleData.default(shape=(2,))
    data1 = data1.replace(id=jnp.array([1, 2]), value=jnp.array([1.0, 2.0]))
    data2 = SimpleData.default(shape=(2,))
    data2 = data2.replace(id=jnp.array([3, 4]), value=jnp.array([3.0, 4.0]))

    result = xnp.stack([data1, data2])

    assert result.structured_type.name == "BATCHED"
    assert result.shape.batch == (2, 2)
    expected_id = jnp.array([[1, 2], [3, 4]])
    expected_value = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_stack_axis_1():
    """Test stacking along axis 1."""
    data1 = SimpleData.default(shape=(2,))
    data1 = data1.replace(id=jnp.array([1, 2]), value=jnp.array([1.0, 2.0]))
    data2 = SimpleData.default(shape=(2,))
    data2 = data2.replace(id=jnp.array([3, 4]), value=jnp.array([3.0, 4.0]))

    result = xnp.stack([data1, data2], axis=1)

    assert result.shape.batch == (2, 2)
    expected_id = jnp.array([[1, 3], [2, 4]])
    expected_value = jnp.array([[1.0, 3.0], [2.0, 4.0]])
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_stack_single_item():
    """Test stacking a single item adds a dimension."""
    data = SimpleData.default()
    data = data.replace(id=jnp.array(42), value=jnp.array(3.14))

    result = xnp.stack([data])

    assert result.structured_type.name == "BATCHED"
    assert result.shape.batch == (1,)
    assert jnp.array_equal(result.id, jnp.array([42]))
    assert jnp.array_equal(result.value, jnp.array([3.14]))


def test_stack_empty_list():
    """Test that stacking empty list raises ValueError."""
    with pytest.raises(ValueError, match="Cannot stack empty list"):
        xnp.stack([])


def test_stack_incompatible_batch_shapes():
    """Test that stacking dataclasses with different batch shapes raises ValueError."""
    data1 = SimpleData.default(shape=(2,))
    data2 = SimpleData.default(shape=(3,))

    with pytest.raises(ValueError, match="All dataclasses must have the same batch shape"):
        xnp.stack([data1, data2])


# Tests for reshape wrapper function
def test_reshape_wrapper():
    """Test that the reshape wrapper function works like the built-in method."""
    data = SimpleData.default(shape=(6,))
    data = data.replace(id=jnp.arange(6), value=jnp.arange(6, dtype=jnp.float32))

    # Test both the wrapper and built-in method
    result_wrapper = xnp.reshape(data, (2, 3))
    result_builtin = data.reshape((2, 3))

    # Results should be identical
    assert jnp.array_equal(result_wrapper.id, result_builtin.id)
    assert jnp.array_equal(result_wrapper.value, result_builtin.value)
    assert result_wrapper.shape.batch == result_builtin.shape.batch == (2, 3)


# Tests for flatten wrapper function
def test_flatten_wrapper():
    """Test that xnp.flatten calls the existing dataclass flatten method"""
    dc = SimpleData.default(shape=(2, 3))
    dc = dc.replace(
        id=jnp.arange(6).reshape(2, 3), value=jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    )

    # Test our wrapper
    result = xnp.flatten(dc)

    # Test direct method call
    expected = dc.flatten()

    # Should be identical
    assert jnp.array_equal(result.id, expected.id)
    assert jnp.array_equal(result.value, expected.value)


def test_where_with_dataclasses():
    """Test xnp.where with two dataclasses"""
    dc1 = SimpleData.default(shape=(3,))
    dc1 = dc1.replace(id=jnp.array([1, 2, 3]), value=jnp.array([1.0, 2.0, 3.0]))
    dc2 = SimpleData.default(shape=(3,))
    dc2 = dc2.replace(id=jnp.array([10, 20, 30]), value=jnp.array([10.0, 20.0, 30.0]))
    condition = jnp.array([True, False, True])

    result = xnp.where(condition, dc1, dc2)

    expected_id = jnp.array([1, 20, 3])  # True->dc1, False->dc2, True->dc1
    expected_value = jnp.array([1.0, 20.0, 3.0])

    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_where_with_scalar():
    """Test xnp.where with dataclass and scalar fallback"""
    dc = SimpleData.default(shape=(3,))
    dc = dc.replace(id=jnp.array([1, 2, 3]), value=jnp.array([1.0, 2.0, 3.0]))
    condition = jnp.array([True, False, True])
    fallback = -1

    result = xnp.where(condition, dc, fallback)

    expected_id = jnp.array([1, -1, 3])  # True->dc.id, False->-1, True->dc.id
    expected_value = jnp.array([1.0, -1.0, 3.0])

    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_where_batched_dataclasses():
    """Test xnp.where with batched dataclasses"""
    dc1 = SimpleData.default(shape=(2,))
    dc1 = dc1.replace(id=jnp.array([1, 2]), value=jnp.array([1.0, 2.0]))
    dc2 = SimpleData.default(shape=(2,))
    dc2 = dc2.replace(id=jnp.array([10, 20]), value=jnp.array([10.0, 20.0]))
    condition = jnp.array([True, False])

    result = xnp.where(condition, dc1, dc2)

    expected_id = jnp.array([1, 20])  # First from dc1, second from dc2
    expected_value = jnp.array([1.0, 20.0])

    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_where_scalar_condition():
    """Test xnp.where with scalar boolean condition"""
    dc1 = SimpleData.default(shape=(3,))
    dc1 = dc1.replace(id=jnp.array([1, 2, 3]), value=jnp.array([1.0, 2.0, 3.0]))
    dc2 = SimpleData.default(shape=(3,))
    dc2 = dc2.replace(id=jnp.array([10, 20, 30]), value=jnp.array([10.0, 20.0, 30.0]))

    # Test with True condition
    result_true = xnp.where(True, dc1, dc2)
    assert jnp.array_equal(result_true.id, dc1.id)
    assert jnp.array_equal(result_true.value, dc1.value)

    # Test with False condition
    result_false = xnp.where(False, dc1, dc2)
    assert jnp.array_equal(result_false.id, dc2.id)
    assert jnp.array_equal(result_false.value, dc2.value)


def test_where_equivalent_to_tree_map():
    """Test that xnp.where produces same result as manual tree_map"""
    dc1 = SimpleData.default(shape=(3,))
    dc1 = dc1.replace(id=jnp.array([1, 2, 3]), value=jnp.array([1.0, 2.0, 3.0]))
    dc2 = SimpleData.default(shape=(3,))
    dc2 = dc2.replace(id=jnp.array([10, 20, 30]), value=jnp.array([10.0, 20.0, 30.0]))
    condition = jnp.array([True, False, True])

    # Using our xnp.where
    result_xnp = xnp.where(condition, dc1, dc2)

    # Using manual tree_map (the old way)
    result_manual = jax.tree_util.tree_map(lambda x, y: jnp.where(condition, x, y), dc1, dc2)

    assert jnp.array_equal(result_xnp.id, result_manual.id)
    assert jnp.array_equal(result_xnp.value, result_manual.value)


def test_where_scalar_equivalent_to_tree_map():
    """Test that xnp.where with scalar produces same result as manual tree_map"""
    dc = SimpleData.default(shape=(3,))
    dc = dc.replace(id=jnp.array([1, 2, 3]), value=jnp.array([1.0, 2.0, 3.0]))
    condition = jnp.array([True, False, True])
    fallback = -1

    # Using our xnp.where
    result_xnp = xnp.where(condition, dc, fallback)

    # Using manual tree_map (the old way)
    result_manual = jax.tree_util.tree_map(lambda x: jnp.where(condition, x, fallback), dc)

    assert jnp.array_equal(result_xnp.id, result_manual.id)
    assert jnp.array_equal(result_xnp.value, result_manual.value)


# Tests for unique_mask function
@xtructure_dataclass
class HashableData:
    """Test dataclass with hashable fields for unique_mask testing."""

    id: FieldDescriptor[jnp.uint32]
    value: FieldDescriptor[jnp.float32]


def test_unique_mask_basic_uniqueness():
    """Test basic uniqueness without cost consideration."""
    # Create data with some duplicates
    data = HashableData.default(shape=(5,))
    data = data.replace(
        id=jnp.array([1, 2, 1, 3, 2]),  # 1 and 2 are duplicates
        value=jnp.array([1.0, 2.0, 1.0, 3.0, 2.0]),
    )

    mask = xnp.unique_mask(data)

    # Should keep first occurrence of each unique element
    expected_mask = jnp.array([True, True, False, True, False])
    assert jnp.array_equal(mask, expected_mask)


def test_unique_mask_with_costs():
    """Test unique filtering with cost-based selection."""
    # Create data with duplicates but different costs
    data = HashableData.default(shape=(6,))
    data = data.replace(
        id=jnp.array([1, 2, 1, 3, 2, 1]),  # Multiple duplicates
        value=jnp.array([1.0, 2.0, 1.0, 3.0, 2.0, 1.0]),
    )

    # Costs: prefer lower costs (indices 0, 2, 5 have same id=1, costs 5.0, 2.0, 1.0)
    costs = jnp.array([5.0, 3.0, 2.0, 4.0, 7.0, 1.0])

    mask = xnp.unique_mask(data, key=costs)

    # Should keep: id=1 with cost=1.0 (index 5), id=2 with cost=3.0 (index 1), id=3 with cost=4.0 (index 3)
    expected_mask = jnp.array([False, True, False, True, False, True])
    assert jnp.array_equal(mask, expected_mask)


def test_unique_mask_tie_breaking():
    """Test tie-breaking when costs are equal."""
    data = HashableData.default(shape=(4,))
    data = data.replace(id=jnp.array([1, 2, 1, 2]), value=jnp.array([1.0, 2.0, 1.0, 2.0]))

    # Same costs for duplicates - should use index for tie-breaking (prefer lower index)
    costs = jnp.array([3.0, 4.0, 3.0, 4.0])

    mask = xnp.unique_mask(data, key=costs)

    # Should keep first occurrence when costs are equal
    expected_mask = jnp.array([True, True, False, False])
    assert jnp.array_equal(mask, expected_mask)


def test_unique_mask_infinite_costs():
    """Test that entries with infinite cost are excluded."""
    data = HashableData.default(shape=(4,))
    data = data.replace(id=jnp.array([1, 2, 1, 2]), value=jnp.array([1.0, 2.0, 1.0, 2.0]))

    # Some entries have infinite cost (invalid/padding entries)
    costs = jnp.array([1.0, jnp.inf, 2.0, 3.0])

    mask = xnp.unique_mask(data, key=costs)

    # Should exclude infinite cost entries and keep finite cost entries
    expected_mask = jnp.array([True, False, False, True])
    assert jnp.array_equal(mask, expected_mask)


def test_unique_mask_all_same():
    """Test when all elements are the same."""
    data = HashableData.default(shape=(3,))
    data = data.replace(id=jnp.array([1, 1, 1]), value=jnp.array([1.0, 1.0, 1.0]))

    costs = jnp.array([3.0, 1.0, 2.0])

    mask = xnp.unique_mask(data, key=costs)

    # Should keep only the one with minimum cost (index 1)
    expected_mask = jnp.array([False, True, False])
    assert jnp.array_equal(mask, expected_mask)


def test_unique_mask_all_unique():
    """Test when all elements are unique."""
    data = HashableData.default(shape=(3,))
    data = data.replace(id=jnp.array([1, 2, 3]), value=jnp.array([1.0, 2.0, 3.0]))

    costs = jnp.array([3.0, 1.0, 2.0])

    mask = xnp.unique_mask(data, key=costs)

    # Should keep all since they're all unique
    expected_mask = jnp.array([True, True, True])
    assert jnp.array_equal(mask, expected_mask)


def test_unique_mask_explicit_batch_len():
    """Test with explicitly provided batch_len."""
    data = HashableData.default(shape=(4,))
    data = data.replace(id=jnp.array([1, 2, 1, 3]), value=jnp.array([1.0, 2.0, 1.0, 3.0]))

    costs = jnp.array([2.0, 3.0, 1.0, 4.0])

    mask = xnp.unique_mask(data, key=costs, batch_len=4)

    # Should keep id=1 with cost=1.0 (index 2), id=2 (index 1), id=3 (index 3)
    expected_mask = jnp.array([False, True, True, True])
    assert jnp.array_equal(mask, expected_mask)


def test_unique_mask_no_key():
    """Test unique_mask without key returns first occurrence."""
    data = HashableData.default(shape=(5,))
    data = data.replace(id=jnp.array([1, 2, 1, 3, 2]), value=jnp.array([1.0, 2.0, 1.0, 3.0, 2.0]))

    mask = xnp.unique_mask(data, key=None)

    # Should return first occurrence of each unique element
    expected_mask = jnp.array([True, True, False, True, False])
    assert jnp.array_equal(mask, expected_mask)


def test_unique_mask_single_element():
    """Test with single element."""
    data = HashableData.default(shape=(1,))
    data = data.replace(id=jnp.array([1]), value=jnp.array([1.0]))

    mask = xnp.unique_mask(data)

    expected_mask = jnp.array([True])
    assert jnp.array_equal(mask, expected_mask)


def test_unique_mask_empty_batch():
    """Test with empty batch."""
    data = HashableData.default(shape=(0,))

    mask = xnp.unique_mask(data, batch_len=0)

    expected_mask = jnp.array([], dtype=jnp.bool_)
    assert jnp.array_equal(mask, expected_mask)


def test_unique_mask_error_no_uint32ed():
    """Test that error is raised when val doesn't have uint32ed attribute."""
    # Create a simple array without uint32ed
    invalid_data = jnp.array([1, 2, 3])

    with pytest.raises(ValueError, match="val must have a uint32ed attribute"):
        xnp.unique_mask(invalid_data)


def test_unique_mask_error_key_length_mismatch():
    """Test that error is raised when key length doesn't match batch_len."""
    data = HashableData.default(shape=(3,))
    data = data.replace(id=jnp.array([1, 2, 3]), value=jnp.array([1.0, 2.0, 3.0]))

    # Key length doesn't match batch length
    wrong_key = jnp.array([1.0, 2.0])  # length 2, but batch is 3

    with pytest.raises(ValueError, match="key length 2 must match batch_len 3"):
        xnp.unique_mask(data, key=wrong_key)


def test_unique_mask_complex_scenario():
    """Test a more complex scenario with multiple duplicates and costs."""
    data = HashableData.default(shape=(8,))
    data = data.replace(
        id=jnp.array([1, 2, 3, 1, 2, 3, 1, 4]),
        value=jnp.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 4.0]),
    )

    # Different costs for each occurrence
    costs = jnp.array([5.0, 3.0, 4.0, 2.0, 6.0, 1.0, 7.0, 2.0])

    mask = xnp.unique_mask(data, key=costs)

    # Should keep: id=1 with min cost=2.0 (index 3), id=2 with min cost=3.0 (index 1),
    #              id=3 with min cost=1.0 (index 5), id=4 with cost=2.0 (index 7)
    expected_mask = jnp.array([False, True, False, True, False, True, False, True])
    assert jnp.array_equal(mask, expected_mask)


def test_unique_mask_integration_with_other_ops():
    """Test unique_mask integration with other xnp operations."""
    # Create data with actual duplicates (same id AND value)
    data = HashableData.default(shape=(6,))
    data = data.replace(
        id=jnp.array([1, 2, 1, 3, 2, 1]),
        value=jnp.array([1.0, 2.0, 1.0, 3.0, 2.0, 1.0]),  # Same values for same ids
    )

    costs = jnp.array([3.0, 2.0, 1.0, 4.0, 5.0, 0.5])

    # Get unique mask
    mask = xnp.unique_mask(data, key=costs)

    # Use mask to filter data using xnp.where
    filtered_data = xnp.where(mask, data, HashableData.default())

    # Verify that we got the expected unique elements
    # id=1 (value=1.0): indices 0,2,5 with costs 3.0,1.0,0.5 -> keep index 5 (cost=0.5)
    # id=2 (value=2.0): indices 1,4 with costs 2.0,5.0 -> keep index 1 (cost=2.0)
    # id=3 (value=3.0): index 3 with cost 4.0 -> keep index 3
    assert jnp.sum(mask) == 3  # Should have 3 unique elements
    expected_mask = jnp.array([False, True, False, True, False, True])
    assert jnp.array_equal(mask, expected_mask)

    # Verify the filtered data has correct values where mask is True
    default_id = HashableData.default().id  # Get the actual default value
    expected_filtered_ids = jnp.where(mask, data.id, default_id)
    assert jnp.array_equal(filtered_data.id, expected_filtered_ids)
