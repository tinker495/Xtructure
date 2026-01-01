import dataclasses
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xtructure import FieldDescriptor, StructuredType, xtructure_dataclass
from xtructure.core.xtructure_decorators.indexing import add_indexing_methods


# Test data structures
@xtructure_dataclass
class SimpleData:
    id: FieldDescriptor.scalar(dtype=jnp.uint32)
    value: FieldDescriptor.scalar(dtype=jnp.float32)


@xtructure_dataclass
class VectorData:
    position: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))
    velocity: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))


@xtructure_dataclass
class MatrixData:
    matrix: FieldDescriptor.tensor(dtype=jnp.float32, shape=(2, 2))
    flags: FieldDescriptor.tensor(dtype=jnp.bool_, shape=(4,), fill_value=False)


@xtructure_dataclass
class NestedData:
    simple: FieldDescriptor.scalar(dtype=SimpleData)
    vector: FieldDescriptor.scalar(dtype=VectorData)


def test_dataclass_default():
    # Test default creation
    simple = SimpleData.default()
    assert simple.id.shape == ()
    assert simple.value.shape == ()
    assert simple.id.dtype == jnp.uint32
    assert simple.value.dtype == jnp.float32

    # Test batched creation
    batched = SimpleData.default(shape=(10,))
    assert batched.id.shape == (10,)
    assert batched.value.shape == (10,)


def test_dataclass_random():
    key = jax.random.PRNGKey(0)

    # Test random creation
    simple = SimpleData.random(key=key)
    assert simple.id.shape == ()
    assert simple.value.shape == ()

    # Test batched random creation
    batched = SimpleData.random(shape=(5,), key=key)
    assert batched.id.shape == (5,)
    assert batched.value.shape == (5,)


def test_vector_data():
    # Test vector data structure
    vector = VectorData.default()
    assert vector.position.shape == (3,)
    assert vector.velocity.shape == (3,)

    # Test batched vector data
    batched = VectorData.default(shape=(4,))
    assert batched.position.shape == (4, 3)
    assert batched.velocity.shape == (4, 3)


def test_matrix_data():
    # Test matrix data structure
    matrix = MatrixData.default()
    assert matrix.matrix.shape == (2, 2)
    assert matrix.flags.shape == (4,)

    # Test batched matrix data
    batched = MatrixData.default(shape=(3,))
    assert batched.matrix.shape == (3, 2, 2)
    assert batched.flags.shape == (3, 4)


def test_nested_data():
    # Test nested data structure
    nested = NestedData.default()
    assert nested.simple.id.shape == ()
    assert nested.simple.value.shape == ()
    assert nested.vector.position.shape == (3,)

    # Test batched nested data
    batched = NestedData.default(shape=(2,))
    assert batched.simple.id.shape == (2,)
    assert batched.simple.value.shape == (2,)
    assert batched.vector.position.shape == (2, 3)


def test_structured_type():
    # Test structured type property
    simple = SimpleData.default()
    assert simple.structured_type == StructuredType.SINGLE

    batched = SimpleData.default(shape=(5,))
    assert batched.structured_type == StructuredType.BATCHED
    assert batched.shape.batch == (5,)

    batched2d = SimpleData.default(shape=(5, 10))
    assert batched2d.structured_type == StructuredType.BATCHED
    assert batched2d.shape.batch == (5, 10)

    vector = VectorData.default(shape=(5, 10))
    assert vector.structured_type == StructuredType.BATCHED
    assert vector.shape.batch == (5, 10)

    matrix = MatrixData.default(shape=(5, 10))
    assert matrix.structured_type == StructuredType.BATCHED
    assert matrix.shape.batch == (5, 10)

    nested = NestedData.default(shape=(5, 10))
    assert nested.structured_type == StructuredType.BATCHED
    assert nested.shape.batch == (5, 10)


def test_len_semantics():
    # SINGLE -> 1
    single = SimpleData.default()
    assert len(single) == 1

    # BATCHED -> first batch dimension
    batched1d = SimpleData.default(shape=(5,))
    assert len(batched1d) == 5

    batched2d = SimpleData.default(shape=(5, 10))
    assert len(batched2d) == 5

    # UNSTRUCTURED -> error (batch size is ill-defined)
    unstructured = SimpleData(id=jnp.array(1), value=jnp.array([2.0, 3.0, 4.0]))
    assert unstructured.structured_type == StructuredType.UNSTRUCTURED
    with pytest.raises(TypeError):
        len(unstructured)


def test_reshape():
    # Test reshape functionality
    batched = SimpleData.default(shape=(10,))
    reshaped = batched.reshape((2, 5))
    assert reshaped.structured_type == StructuredType.BATCHED
    assert reshaped.shape.batch == (2, 5)
    assert reshaped.id.shape == (2, 5)
    assert reshaped.value.shape == (2, 5)

    batched2d = SimpleData.default(shape=(2, 3))
    reshaped2d = batched2d.reshape((6,))
    assert reshaped2d.structured_type == StructuredType.BATCHED
    assert reshaped2d.shape.batch == (6,)
    assert reshaped2d.id.shape == (6,)
    assert reshaped2d.value.shape == (6,)

    vector = VectorData.default(shape=(10,))
    reshaped_vector = vector.reshape((2, 5))
    assert reshaped_vector.structured_type == StructuredType.BATCHED
    assert reshaped_vector.shape.batch == (2, 5)
    assert reshaped_vector.position.shape == (2, 5, 3)
    assert reshaped_vector.velocity.shape == (2, 5, 3)

    vector2d = VectorData.default(shape=(2, 3))
    reshaped_vector2d = vector2d.reshape((6,))
    assert reshaped_vector2d.structured_type == StructuredType.BATCHED
    assert reshaped_vector2d.shape.batch == (6,)
    assert reshaped_vector2d.position.shape == (6, 3)
    assert reshaped_vector2d.velocity.shape == (6, 3)

    matrix = MatrixData.default(shape=(10,))
    reshaped_matrix = matrix.reshape((2, 5))
    assert reshaped_matrix.structured_type == StructuredType.BATCHED
    assert reshaped_matrix.shape.batch == (2, 5)
    assert reshaped_matrix.matrix.shape == (2, 5, 2, 2)

    matrix2d = MatrixData.default(shape=(2, 3))
    reshaped_matrix2d = matrix2d.reshape((6,))
    assert reshaped_matrix2d.structured_type == StructuredType.BATCHED
    assert reshaped_matrix2d.shape.batch == (6,)
    assert reshaped_matrix2d.matrix.shape == (6, 2, 2)
    assert reshaped_matrix2d.flags.shape == (6, 4)

    nested = NestedData.default(shape=(10,))
    reshaped_nested = nested.reshape((2, 5))
    assert reshaped_nested.structured_type == StructuredType.BATCHED
    assert reshaped_nested.shape.batch == (2, 5)
    assert reshaped_nested.simple.id.shape == (2, 5)
    assert reshaped_nested.simple.value.shape == (2, 5)

    nested2d = NestedData.default(shape=(2, 3))
    reshaped_nested2d = nested2d.reshape((6,))
    assert reshaped_nested2d.structured_type == StructuredType.BATCHED
    assert reshaped_nested2d.shape.batch == (6,)
    assert reshaped_nested2d.simple.id.shape == (6,)
    assert reshaped_nested2d.simple.value.shape == (6,)


def test_reshape_with_minus_one():
    """Test reshape with -1 to automatically calculate dimensions."""
    # Test SimpleData
    batched = SimpleData.default(shape=(12,))
    batched = batched.replace(
        id=jnp.arange(12, dtype=jnp.uint32), value=jnp.arange(12, dtype=jnp.float32)
    )

    # Test -1 at the end
    reshaped1 = batched.reshape((2, -1))
    assert reshaped1.structured_type == StructuredType.BATCHED
    assert reshaped1.shape.batch == (2, 6)
    assert reshaped1.id.shape == (2, 6)
    assert reshaped1.value.shape == (2, 6)

    # Test -1 at the beginning
    reshaped2 = batched.reshape((-1, 3))
    assert reshaped2.structured_type == StructuredType.BATCHED
    assert reshaped2.shape.batch == (4, 3)
    assert reshaped2.id.shape == (4, 3)
    assert reshaped2.value.shape == (4, 3)

    # Test -1 alone (flatten)
    reshaped3 = batched.reshape((-1,))
    assert reshaped3.structured_type == StructuredType.BATCHED
    assert reshaped3.shape.batch == (12,)
    assert reshaped3.id.shape == (12,)
    assert reshaped3.value.shape == (12,)

    # Test VectorData
    vector = VectorData.default(shape=(10,))
    vector = vector.replace(
        position=jnp.arange(30, dtype=jnp.float32).reshape(10, 3),
        velocity=jnp.arange(30, dtype=jnp.float32).reshape(10, 3) + 100,
    )

    # Test -1 reshape on vector data
    reshaped_vector = vector.reshape((2, -1))
    assert reshaped_vector.structured_type == StructuredType.BATCHED
    assert reshaped_vector.shape.batch == (2, 5)
    assert reshaped_vector.position.shape == (2, 5, 3)
    assert reshaped_vector.velocity.shape == (2, 5, 3)

    # Test MatrixData
    matrix = MatrixData.default(shape=(8,))
    matrix = matrix.replace(
        matrix=jnp.arange(32, dtype=jnp.float32).reshape(8, 2, 2),
        flags=jnp.arange(32, dtype=jnp.int32).reshape(8, 4).astype(jnp.bool_),
    )

    # Test -1 reshape on matrix data
    reshaped_matrix = matrix.reshape((2, -1))
    assert reshaped_matrix.structured_type == StructuredType.BATCHED
    assert reshaped_matrix.shape.batch == (2, 4)
    assert reshaped_matrix.matrix.shape == (2, 4, 2, 2)
    assert reshaped_matrix.flags.shape == (2, 4, 4)

    # Test NestedData
    nested = NestedData.default(shape=(6,))
    nested = nested.replace(
        simple=SimpleData(
            id=jnp.arange(6, dtype=jnp.uint32), value=jnp.arange(6, dtype=jnp.float32)
        ),
        vector=VectorData(
            position=jnp.arange(18, dtype=jnp.float32).reshape(6, 3),
            velocity=jnp.arange(18, dtype=jnp.float32).reshape(6, 3) + 100,
        ),
    )

    # Test -1 reshape on nested data
    reshaped_nested = nested.reshape((2, -1))
    assert reshaped_nested.structured_type == StructuredType.BATCHED
    assert reshaped_nested.shape.batch == (2, 3)
    assert reshaped_nested.simple.id.shape == (2, 3)
    assert reshaped_nested.simple.value.shape == (2, 3)
    assert reshaped_nested.vector.position.shape == (2, 3, 3)
    assert reshaped_nested.vector.velocity.shape == (2, 3, 3)


def test_reshape_with_minus_one_errors():
    """Test that reshape with invalid -1 usage raises appropriate errors."""
    batched = SimpleData.default(shape=(10,))
    batched = batched.replace(
        id=jnp.arange(10, dtype=jnp.uint32), value=jnp.arange(10, dtype=jnp.float32)
    )

    # Test multiple -1s (should raise error)
    try:
        batched.reshape((-1, -1))
        assert False, "Multiple -1s should have raised an error"
    except ValueError as e:
        assert "Only one -1 is allowed in new_shape" in str(e)

    # Test invalid shape that doesn't divide evenly
    try:
        batched.reshape((3, -1))
        assert False, "Invalid shape should have raised an error"
    except ValueError as e:
        assert "Total length 10 is not divisible by the product of other dimensions 3" in str(e)

    # Test with zero dimension (should raise error)
    try:
        batched.reshape((0, -1))
        assert False, "Zero dimension should have raised an error"
    except ValueError as e:
        assert "Cannot infer -1 dimension when other dimensions are 0" in str(e)


def test_flatten():
    # Test flatten functionality
    batched = SimpleData.default(shape=(2, 3))
    flattened = batched.flatten()
    print(flattened.structured_type)
    assert flattened.structured_type == StructuredType.BATCHED
    assert flattened.shape.batch == (6,)
    assert flattened.id.shape == (6,)
    assert flattened.value.shape == (6,)

    batched2d = SimpleData.default(shape=(2, 3))
    flattened2d = batched2d.flatten()
    assert flattened2d.structured_type == StructuredType.BATCHED
    assert flattened2d.shape.batch == (6,)
    assert flattened2d.id.shape == (6,)
    assert flattened2d.value.shape == (6,)

    vector = VectorData.default(shape=(2, 3))
    flattened_vector = vector.flatten()
    assert flattened_vector.structured_type == StructuredType.BATCHED
    assert flattened_vector.shape.batch == (6,)
    assert flattened_vector.position.shape == (6, 3)
    assert flattened_vector.velocity.shape == (6, 3)

    matrix = MatrixData.default(shape=(2, 3))
    flattened_matrix = matrix.flatten()
    assert flattened_matrix.structured_type == StructuredType.BATCHED
    assert flattened_matrix.shape.batch == (6,)
    assert flattened_matrix.matrix.shape == (6, 2, 2)
    assert flattened_matrix.flags.shape == (6, 4)

    nested = NestedData.default(shape=(2, 3))
    flattened_nested = nested.flatten()
    assert flattened_nested.structured_type == StructuredType.BATCHED
    assert flattened_nested.shape.batch == (6,)
    assert flattened_nested.simple.id.shape == (6,)
    assert flattened_nested.simple.value.shape == (6,)


def test_transpose_simple_data():
    data = SimpleData.default(shape=(2, 3))
    data = data.replace(
        id=jnp.arange(6, dtype=jnp.uint32).reshape(2, 3),
        value=jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
    )

    result = data.transpose()

    assert result.shape.batch == (3, 2)
    assert jnp.array_equal(result.id, jnp.transpose(data.id))
    assert jnp.array_equal(result.value, jnp.transpose(data.value))


def test_transpose_vector_data_batch_only():
    data = VectorData.default(shape=(2, 3))
    data = data.replace(
        position=jnp.arange(18, dtype=jnp.float32).reshape(2, 3, 3),
        velocity=jnp.arange(18, dtype=jnp.float32).reshape(2, 3, 3) + 100,
    )

    result = data.transpose()

    expected_position = jnp.transpose(data.position, axes=(1, 0, 2))
    expected_velocity = jnp.transpose(data.velocity, axes=(1, 0, 2))
    assert result.shape.batch == (3, 2)
    assert jnp.array_equal(result.position, expected_position)
    assert jnp.array_equal(result.velocity, expected_velocity)


def test_transpose_with_custom_axes():
    data = SimpleData.default(shape=(2, 3, 4))
    data = data.replace(
        id=jnp.arange(24, dtype=jnp.uint32).reshape(2, 3, 4),
        value=jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4),
    )
    axes = (2, 0, 1)

    result = data.transpose(axes=axes)

    assert result.shape.batch == (4, 2, 3)
    assert jnp.array_equal(result.id, jnp.transpose(data.id, axes=axes))
    assert jnp.array_equal(result.value, jnp.transpose(data.value, axes=axes))


def test_transpose_unstructured_raises():
    unstructured = SimpleData(id=jnp.array(1), value=jnp.array([2.0, 3.0, 4.0]))
    assert unstructured.structured_type == StructuredType.UNSTRUCTURED
    with pytest.raises(ValueError):
        unstructured.transpose()


def test_indexing():
    # Test indexing functionality
    batched = SimpleData.default(shape=(5,))
    single = batched[0]
    assert single.structured_type == StructuredType.SINGLE
    assert single.id.shape == ()
    assert single.value.shape == ()

    # Test slicing
    sliced = batched[1:3]
    assert sliced.structured_type == StructuredType.BATCHED
    assert sliced.shape.batch == (2,)
    assert sliced.id.shape == (2,)
    assert sliced.value.shape == (2,)


def test_unstructured_generation():

    unstructured = SimpleData(id=jnp.array(1), value=jnp.array([2.0, 3.0, 4.0]))
    assert unstructured.structured_type == StructuredType.UNSTRUCTURED
    assert unstructured.id.shape == ()
    assert unstructured.value.shape == (3,)

    batched_unstructured = SimpleData(
        id=jnp.array([1, 2]), value=jnp.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    )
    assert batched_unstructured.structured_type == StructuredType.UNSTRUCTURED
    assert batched_unstructured.id.shape == (2,)
    assert batched_unstructured.value.shape == (2, 3)

    try:
        batched_unstructured.reshape((2, 3))
        assert False, "unstructured data should not be reshaped"
    except ValueError:
        pass

    try:
        batched_unstructured.flatten()
        assert False, "flatten operation is only supported for BATCHED structured types"
    except ValueError:
        pass


def test_at_set_simple_data():
    # Test .at[...].set(...) for SimpleData
    original_data = SimpleData.default(shape=(3,))

    # Create new data to set
    data_to_set_scalar = SimpleData(
        id=jnp.array(100, dtype=jnp.uint32), value=jnp.array(99.9, dtype=jnp.float32)
    )

    # Update a single element with another SimpleData instance
    updated_data_single = original_data.at[1].set(data_to_set_scalar)

    assert updated_data_single.id[0] == original_data.id[0]
    assert updated_data_single.value[0] == original_data.value[0]
    assert updated_data_single.id[1] == data_to_set_scalar.id
    assert updated_data_single.value[1] == data_to_set_scalar.value
    assert updated_data_single.id[2] == original_data.id[2]
    assert updated_data_single.value[2] == original_data.value[2]

    # Ensure original data is unchanged
    assert original_data.id[1] != data_to_set_scalar.id
    assert original_data.value[1] != data_to_set_scalar.value

    # Update using a scalar value for all fields (if JAX supports it for specific dtype)
    # For SimpleData, id is uint32 and value is float32.
    # JAX .at[idx].set(scalar) will broadcast if the scalar is compatible.
    updated_data_scalar_id = original_data.at[0].set(jnp.uint32(50))
    assert updated_data_scalar_id.id[0] == 50
    # The value field should also be updated with 50 if broadcast works, or remain original if not.
    # Given current implementation, value_for_this_field = values_to_set (which is 50)
    # jnp.array(0.0, dtype=jnp.float32).at[()].set(50) would make it 50.0
    assert updated_data_scalar_id.value[0] == 50.0
    assert updated_data_scalar_id.id[1] == original_data.id[1]
    assert updated_data_scalar_id.value[1] == original_data.value[1]

    # Test setting with a slice
    slice_data_to_set = SimpleData.default(shape=(2,))  # id=0, value=0.0
    updated_data_slice = original_data.at[0:2].set(slice_data_to_set)
    assert updated_data_slice.id[0] == slice_data_to_set.id[0]
    assert updated_data_slice.value[0] == slice_data_to_set.value[0]
    assert updated_data_slice.id[1] == slice_data_to_set.id[1]
    assert updated_data_slice.value[1] == slice_data_to_set.value[1]
    assert updated_data_slice.id[2] == original_data.id[2]
    assert updated_data_slice.value[2] == original_data.value[2]


def test_at_set_vector_data():
    original_data = VectorData.default(
        shape=(3,)
    )  # position and velocity are (3,3) filled with 0.0

    # Data to set for a single batch element
    # position and velocity should be (3,)
    vector_to_set = VectorData(
        position=jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
        velocity=jnp.array([4.0, 5.0, 6.0], dtype=jnp.float32),
    )

    updated_data = original_data.at[1].set(vector_to_set)

    assert jnp.array_equal(updated_data.position[0], original_data.position[0])
    assert jnp.array_equal(updated_data.velocity[0], original_data.velocity[0])

    assert jnp.array_equal(updated_data.position[1], vector_to_set.position)
    assert jnp.array_equal(updated_data.velocity[1], vector_to_set.velocity)

    assert jnp.array_equal(updated_data.position[2], original_data.position[2])
    assert jnp.array_equal(updated_data.velocity[2], original_data.velocity[2])

    # Ensure original data is unchanged
    assert not jnp.array_equal(original_data.position[1], vector_to_set.position)

    # Test setting with a scalar (will broadcast to the (3,) shape of position/velocity)
    updated_data_scalar = original_data.at[0].set(jnp.float32(7.0))
    assert jnp.all(updated_data_scalar.position[0] == 7.0)
    assert jnp.all(updated_data_scalar.velocity[0] == 7.0)
    assert jnp.array_equal(updated_data_scalar.position[1], original_data.position[1])


def test_at_set_nested_data():
    original_data = NestedData.default(shape=(2,))
    # original_data.simple.id shape (2,), value (2,)
    # original_data.vector.position shape (2,3), velocity (2,3)

    # Create a single NestedData instance to set at one index
    # This instance itself should NOT be batched. Its internal fields are single.
    data_to_set_single_nested = (
        NestedData.default()
    )  # scalar id, value, (3,) position, (3,) velocity
    data_to_set_single_nested = data_to_set_single_nested.replace(
        simple=SimpleData(
            id=jnp.array(10, dtype=jnp.uint32), value=jnp.array(1.1, dtype=jnp.float32)
        ),
        vector=VectorData(
            position=jnp.ones(3, dtype=jnp.float32), velocity=jnp.ones(3, dtype=jnp.float32) * 2
        ),
    )

    updated_data = original_data.at[0].set(data_to_set_single_nested)

    # Check updated part
    assert updated_data.simple.id[0] == data_to_set_single_nested.simple.id
    assert updated_data.simple.value[0] == data_to_set_single_nested.simple.value
    assert jnp.array_equal(
        updated_data.vector.position[0], data_to_set_single_nested.vector.position
    )
    assert jnp.array_equal(
        updated_data.vector.velocity[0], data_to_set_single_nested.vector.velocity
    )

    # Check unchanged part
    assert updated_data.simple.id[1] == original_data.simple.id[1]
    assert updated_data.simple.value[1] == original_data.simple.value[1]
    assert jnp.array_equal(updated_data.vector.position[1], original_data.vector.position[1])
    assert jnp.array_equal(updated_data.vector.velocity[1], original_data.vector.velocity[1])

    # Ensure original data is unchanged
    assert original_data.simple.id[0] != data_to_set_single_nested.simple.id


# Create a separate test class for the indexing decorator
@dataclasses.dataclass
class IndexedData:
    x: jnp.ndarray
    y: jnp.ndarray


IndexedData = add_indexing_methods(IndexedData)


class TestIndexingDecorator(unittest.TestCase):
    def test_set_as_condition_duplicate_indices(self):
        """
        Tests that `set_as_condition` with duplicate indices correctly applies the update
        based on the first condition for that index.
        """
        instance = IndexedData(x=jnp.zeros(5), y=jnp.zeros(5))

        # Indices [0, 0], conditions [True, False]. The first update for index 0 is True,
        # so the update should be applied.
        indices = jnp.array([0, 0, 1])
        condition = jnp.array([True, False, True])

        updated_instance = instance.at[indices].set_as_condition(condition, 99)

        # The first update for index 0 is True, so it becomes 99.
        # The update for index 1 is True.
        expected_x = jnp.array([99.0, 99.0, 0.0, 0.0, 0.0])
        self.assertTrue(jnp.array_equal(updated_instance.x, expected_x))
        self.assertTrue(jnp.array_equal(updated_instance.y, expected_x))

    def test_set_as_condition_advanced_indexing_with_duplicates(self):
        """
        Tests `set_as_condition` with advanced indexing (tuple of arrays) and duplicate indices.
        The first condition should win.
        """
        instance = IndexedData(x=jnp.zeros((5, 5)), y=jnp.zeros((5, 5)))

        # Indices targeting (0,0), (0,0), and (1,1). Conditions are True, False, True.
        rows = jnp.array([0, 0, 1])
        cols = jnp.array([0, 0, 1])
        indices = (rows, cols)
        condition = jnp.array([True, False, True])

        updated_instance = instance.at[indices].set_as_condition(condition, 88)

        # first condition for (0,0) is True, so it's updated.
        # condition for (1,1) is True.
        expected_x = jnp.zeros((5, 5))
        expected_x = expected_x.at[0, 0].set(88)
        expected_x = expected_x.at[1, 1].set(88)

        self.assertTrue(jnp.array_equal(updated_instance.x, expected_x))
        self.assertTrue(jnp.array_equal(updated_instance.y, expected_x))

    def test_set_as_condition_with_scalar_value_broadcast(self):
        """
        Tests that a scalar value is correctly broadcasted across all `True` conditions.
        """
        instance = IndexedData(x=jnp.arange(10), y=jnp.arange(10))

        indices = jnp.array([1, 3, 5, 7])
        condition = jnp.array([True, False, True, False])

        updated_instance = instance.at[indices].set_as_condition(condition, -1)

        # Only indices 1 and 5 should be updated to -1.
        expected_x = jnp.array([0, -1, 2, 3, 4, -1, 6, 7, 8, 9])
        self.assertTrue(jnp.array_equal(updated_instance.x, expected_x))
        self.assertTrue(jnp.array_equal(updated_instance.y, expected_x))

    def test_set_as_condition_with_array_values(self):
        """
        Tests conditional set with an array of values, ensuring the first
        update for an index is the one that takes effect.
        """
        instance = IndexedData(x=jnp.zeros(4), y=jnp.zeros(4))

        indices = jnp.array([0, 1, 0, 2])  # Duplicate index 0
        condition = jnp.array([True, True, False, True])
        values_to_set = jnp.array([10, 20, 30, 40])

        updated_instance = instance.at[indices].set_as_condition(condition, values_to_set)

        # Expected:
        # Index 0 gets value 10 (first is True).
        # Index 1 gets value 20 (True).
        # Index 2 gets value 40 (True).
        expected_x = jnp.array([10.0, 20.0, 40.0, 0.0])

        self.assertTrue(jnp.array_equal(updated_instance.x, expected_x))
        self.assertTrue(jnp.array_equal(updated_instance.y, expected_x))

    def test_set_as_condition_extreme_randomized(self):
        """
        Tests `set_as_condition` with a large number of random updates,
        including duplicate indices and mixed conditions, to ensure robustness.
        """
        key = jax.random.PRNGKey(42)
        data_size = 1000
        num_updates = 5000  # More updates than data size to ensure many duplicates

        instance = IndexedData(
            x=jnp.zeros(data_size, dtype=jnp.float32),
            y=jnp.zeros(data_size, dtype=jnp.float32),
        )

        # Generate random indices, conditions, and values
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
        indices = jax.random.randint(subkey1, (num_updates,), 0, data_size)
        condition = jax.random.choice(subkey2, jnp.array([True, False]), (num_updates,))
        values_to_set = jax.random.normal(subkey3, (num_updates,), dtype=jnp.float32)

        # --- Calculate expected result using a clear, sequential logic ---
        # We use a standard Python dictionary and NumPy array to avoid JAX's
        # parallel execution nuances, establishing a ground truth.
        # The logic is "first True condition wins".
        expected_x = np.zeros(data_size, dtype=np.float32)

        # This dictionary will hold the final value for each index based on the first update rule.
        updates_to_apply = {}
        for i in range(len(indices)):
            idx = indices[i].item()
            if condition[i]:
                if idx not in updates_to_apply:
                    updates_to_apply[idx] = values_to_set[i].item()

        # Apply the determined updates to the numpy array
        for idx, value in updates_to_apply.items():
            expected_x[idx] = value

        # --- Perform the actual operation using the method under test ---
        updated_instance = instance.at[indices].set_as_condition(condition, values_to_set)

        # --- Compare results ---
        self.assertTrue(
            jnp.array_equal(updated_instance.x, jnp.array(expected_x)),
            "Randomized test with array of values failed for field 'x'",
        )
        self.assertTrue(
            jnp.array_equal(updated_instance.y, jnp.array(expected_x)),
            "Randomized test with array of values failed for field 'y'",
        )

    def test_set_as_condition_extreme_randomized_scalar(self):
        """
        Tests `set_as_condition` with a large number of random updates
        and a single scalar value to ensure correct broadcasting.
        """
        key = jax.random.PRNGKey(84)
        data_size = 1000
        num_updates = 5000
        scalar_value = 99.0

        instance = IndexedData(
            x=jnp.zeros(data_size, dtype=jnp.float32),
            y=jnp.zeros(data_size, dtype=jnp.float32),
        )

        # Generate random indices and conditions
        key, subkey1, subkey2 = jax.random.split(key, 3)
        indices = jax.random.randint(subkey1, (num_updates,), 0, data_size)
        condition = jax.random.choice(subkey2, jnp.array([True, False]), (num_updates,))

        # --- Calculate expected result ---
        expected_x = np.zeros(data_size, dtype=np.float32)

        # Find all unique indices where the FIRST condition is True
        updates_to_apply = {}
        for i in range(len(indices)):
            idx = indices[i].item()
            if condition[i]:
                if idx not in updates_to_apply:
                    updates_to_apply[idx] = scalar_value

        # Apply the scalar value to these indices
        for idx, value in updates_to_apply.items():
            expected_x[idx] = value

        # --- Perform the actual operation ---
        updated_instance = instance.at[indices].set_as_condition(condition, scalar_value)

        # --- Compare results ---
        self.assertTrue(
            jnp.array_equal(updated_instance.x, jnp.array(expected_x)),
            "Randomized test with scalar value failed for field 'x'",
        )
        self.assertTrue(
            jnp.array_equal(updated_instance.y, jnp.array(expected_x)),
            "Randomized test with scalar value failed for field 'y'",
        )


@xtructure_dataclass(validate=True)
class ValidatedScalarData:
    value: FieldDescriptor.scalar(dtype=jnp.float32)


@xtructure_dataclass(validate=True)
class ValidatedVectorData:
    vector: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))


@xtructure_dataclass(validate=True)
class ValidatedNestedData:
    simple: FieldDescriptor.scalar(dtype=SimpleData)


@xtructure_dataclass(validate=True)
class ValidatedWithPostInit:
    value: FieldDescriptor.scalar(dtype=jnp.float32)

    def __post_init__(self):
        self.value = self.value + 1.0


def test_validate_dtype_mismatch():
    ValidatedScalarData(value=jnp.array(1.0, dtype=jnp.float32))
    with pytest.raises(TypeError):
        ValidatedScalarData(value=jnp.array(1.0, dtype=jnp.int32))


def test_validate_shape_mismatch():
    ValidatedVectorData(vector=jnp.ones((2, 3), dtype=jnp.float32))
    with pytest.raises(ValueError):
        ValidatedVectorData(vector=jnp.ones((2,), dtype=jnp.float32))


def test_validate_nested_type():
    ValidatedNestedData(simple=SimpleData.default())
    with pytest.raises(TypeError):
        ValidatedNestedData(simple=VectorData.default())


def test_validate_preserves_existing_post_init():
    data = ValidatedWithPostInit(value=jnp.array(1.0, dtype=jnp.float32))
    assert jnp.array_equal(data.value, jnp.array(2.0, dtype=jnp.float32))


if __name__ == "__main__":
    unittest.main()
