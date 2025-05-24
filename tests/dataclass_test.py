import jax
import jax.numpy as jnp

from xtructure import FieldDescriptor, StructuredType, xtructure_dataclass


# Test data structures
@xtructure_dataclass
class SimpleData:
    id: FieldDescriptor[jnp.uint32]
    value: FieldDescriptor[jnp.float32]


@xtructure_dataclass
class VectorData:
    position: FieldDescriptor[jnp.float32, (3,)]
    velocity: FieldDescriptor[jnp.float32, (3,)]


@xtructure_dataclass
class MatrixData:
    matrix: FieldDescriptor[jnp.float32, (2, 2)]
    flags: FieldDescriptor[jnp.bool_, (4,), False]


@xtructure_dataclass
class NestedData:
    simple: FieldDescriptor[SimpleData]
    vector: FieldDescriptor[VectorData]


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
