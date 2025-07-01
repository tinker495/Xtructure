import jax
import jax.numpy as jnp
from xtructure import xtructure_numpy as xnp
from xtructure import FieldDescriptor, xtructure_dataclass
import pytest


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


def test_set_as_condition_duplicate_indices_last_wins():
    original_array = jnp.zeros(5)
    indices = jnp.array([0, 2, 0])
    condition = jnp.array([True, True, True])
    values_to_set = jnp.array([1.0, 2.0, 3.0])
    result = xnp.set_as_condition_on_array(original_array, indices, condition, values_to_set)
    expected = jnp.array([3.0, 0.0, 2.0, 0.0, 0.0])
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
    expected = jnp.array([40.0, 0.0, 20.0, 0.0, 0.0])
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
    
    result = xnp.concat([data1, data2, data3])
    
    assert result.structured_type.name == 'BATCHED'
    assert result.shape.batch == (3,)
    assert jnp.array_equal(result.id, jnp.array([1, 2, 3]))
    assert jnp.array_equal(result.value, jnp.array([1.0, 2.0, 3.0]))


def test_concat_batched_dataclasses():
    """Test concatenating BATCHED structured dataclasses."""
    data1 = SimpleData.default(shape=(2,))
    data1 = data1.replace(id=jnp.array([1, 2]), value=jnp.array([1.0, 2.0]))
    data2 = SimpleData.default(shape=(3,))
    data2 = data2.replace(id=jnp.array([3, 4, 5]), value=jnp.array([3.0, 4.0, 5.0]))
    
    result = xnp.concat([data1, data2])
    
    assert result.structured_type.name == 'BATCHED'
    assert result.shape.batch == (5,)
    assert jnp.array_equal(result.id, jnp.array([1, 2, 3, 4, 5]))
    assert jnp.array_equal(result.value, jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]))


def test_concat_vector_dataclasses():
    """Test concatenating dataclasses with vector fields."""
    data1 = VectorData.default(shape=(2,))
    data1 = data1.replace(
        position=jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        velocity=jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    )
    data2 = VectorData.default(shape=(1,))
    data2 = data2.replace(
        position=jnp.array([[7.0, 8.0, 9.0]]),
        velocity=jnp.array([[0.7, 0.8, 0.9]])
    )
    
    result = xnp.concat([data1, data2])
    
    assert result.structured_type.name == 'BATCHED'
    assert result.shape.batch == (3,)
    expected_pos = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    expected_vel = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    assert jnp.allclose(result.position, expected_pos)
    assert jnp.allclose(result.velocity, expected_vel)


def test_concat_empty_list():
    """Test that concatenating empty list raises ValueError."""
    with pytest.raises(ValueError, match="Cannot concatenate empty list"):
        xnp.concat([])


def test_concat_single_item():
    """Test that concatenating single item returns the item itself."""
    data = SimpleData.default()
    result = xnp.concat([data])
    assert result is data


def test_concat_incompatible_types():
    """Test that concatenating different types raises ValueError."""
    simple_data = SimpleData.default()
    vector_data = VectorData.default()
    
    with pytest.raises(ValueError, match="All dataclasses must be of the same type"):
        xnp.concat([simple_data, vector_data])


# Tests for pad function
def test_pad_single_to_batched():
    """Test padding a SINGLE dataclass to create a batched version."""
    data = SimpleData.default()
    data = data.replace(id=jnp.array(42), value=jnp.array(3.14))
    
    result = xnp.pad(data, target_size=5)
    
    assert result.structured_type.name == 'BATCHED'
    assert result.shape.batch == (5,)
    assert jnp.array_equal(result.id, jnp.array([42, 42, 42, 42, 42]))
    assert jnp.allclose(result.value, jnp.array([3.14, 3.14, 3.14, 3.14, 3.14]))


def test_pad_batched_axis_0():
    """Test padding a BATCHED dataclass along axis 0."""
    data = SimpleData.default(shape=(3,))
    data = data.replace(id=jnp.array([1, 2, 3]), value=jnp.array([1.0, 2.0, 3.0]))
    
    result = xnp.pad(data, target_size=5, axis=0)
    
    assert result.structured_type.name == 'BATCHED'
    assert result.shape.batch == (5,)
    # Use the actual default fill values: uint32 max value and float32 inf
    expected_id = jnp.array([1, 2, 3, 4294967295, 4294967295], dtype=jnp.uint32)
    expected_value = jnp.array([1.0, 2.0, 3.0, jnp.inf, jnp.inf], dtype=jnp.float32)
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_pad_uses_existing_padding_as_batch():
    """Test that pad function uses the existing padding_as_batch method when appropriate."""
    data = SimpleData.default(shape=(2,))
    data = data.replace(id=jnp.array([1, 2]), value=jnp.array([1.0, 2.0]))
    
    # This should use the existing padding_as_batch method
    result_xnp = xnp.pad(data, target_size=4)
    result_builtin = data.padding_as_batch((4,))
    
    # Results should be identical
    assert jnp.array_equal(result_xnp.id, result_builtin.id)
    assert jnp.array_equal(result_xnp.value, result_builtin.value)
    assert result_xnp.shape.batch == result_builtin.shape.batch


def test_pad_batched_with_constant_values():
    """Test padding with custom constant values."""
    data = SimpleData.default(shape=(2,))
    data = data.replace(id=jnp.array([1, 2]), value=jnp.array([1.0, 2.0]))
    
    result = xnp.pad(data, target_size=4, constant_values=99)
    
    assert result.shape.batch == (4,)
    assert jnp.array_equal(result.id, jnp.array([1, 2, 99, 99], dtype=jnp.uint32))
    assert jnp.array_equal(result.value, jnp.array([1.0, 2.0, 99.0, 99.0], dtype=jnp.float32))


def test_pad_batched_target_shape():
    """Test padding to a target batch shape."""
    data = SimpleData.default(shape=(2, 3))
    
    result = xnp.pad(data, target_size=(4, 5))
    
    assert result.shape.batch == (4, 5)


def test_pad_no_change_needed():
    """Test that padding to current size returns the same instance."""
    data = SimpleData.default(shape=(3,))
    result = xnp.pad(data, target_size=3)
    assert result is data


def test_pad_target_smaller_than_current():
    """Test that padding to smaller size raises ValueError."""
    data = SimpleData.default(shape=(5,))
    
    with pytest.raises(ValueError, match="Target size 3 is smaller than current size 5"):
        xnp.pad(data, target_size=3)


# Tests for stack function
def test_stack_single_dataclasses():
    """Test stacking SINGLE structured dataclasses."""
    data1 = SimpleData.default()
    data1 = data1.replace(id=jnp.array(1), value=jnp.array(1.0))
    data2 = SimpleData.default()
    data2 = data2.replace(id=jnp.array(2), value=jnp.array(2.0))
    
    result = xnp.stack([data1, data2])
    
    assert result.structured_type.name == 'BATCHED'
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
    
    assert result.structured_type.name == 'BATCHED'
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
    
    assert result.structured_type.name == 'BATCHED'
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
        id=jnp.arange(6).reshape(2, 3), 
        value=jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
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
    result_manual = jax.tree_util.tree_map(
        lambda x, y: jnp.where(condition, x, y), dc1, dc2
    )
    
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
    result_manual = jax.tree_util.tree_map(
        lambda x: jnp.where(condition, x, fallback), dc
    )
    
    assert jnp.array_equal(result_xnp.id, result_manual.id)
    assert jnp.array_equal(result_xnp.value, result_manual.value) 