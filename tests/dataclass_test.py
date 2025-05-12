import jax
import jax.numpy as jnp
import pytest
from Xtructure import xtructure_dataclass, FieldDescriptor, StructuredType

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
    assert batched.batch_shape == (5,)

    batched2d = SimpleData.default(shape=(5, 10))
    assert batched2d.structured_type == StructuredType.BATCHED
    assert batched2d.batch_shape == (5, 10)

    vector = VectorData.default(shape=(5, 10))
    assert vector.structured_type == StructuredType.BATCHED
    assert vector.batch_shape == (5, 10)

    matrix = MatrixData.default(shape=(5, 10))
    assert matrix.structured_type == StructuredType.BATCHED
    assert matrix.batch_shape == (5, 10)

    nested = NestedData.default(shape=(5, 10))
    assert nested.structured_type == StructuredType.BATCHED
    assert nested.batch_shape == (5, 10)

def test_reshape():
    # Test reshape functionality
    batched = SimpleData.default(shape=(10,))
    reshaped = batched.reshape((2, 5))
    assert reshaped.structured_type == StructuredType.BATCHED
    assert reshaped.batch_shape == (2, 5)
    assert reshaped.id.shape == (2, 5)
    assert reshaped.value.shape == (2, 5)

    batched2d = SimpleData.default(shape=(2, 3))
    reshaped2d = batched2d.reshape((6,))
    assert reshaped2d.structured_type == StructuredType.BATCHED
    assert reshaped2d.batch_shape == (6,)
    assert reshaped2d.id.shape == (6,)
    assert reshaped2d.value.shape == (6,)

    vector = VectorData.default(shape=(10,))
    reshaped_vector = vector.reshape((2, 5))
    assert reshaped_vector.structured_type == StructuredType.BATCHED
    assert reshaped_vector.batch_shape == (2, 5)
    assert reshaped_vector.position.shape == (2, 5, 3)
    assert reshaped_vector.velocity.shape == (2, 5, 3)
    
    vector2d = VectorData.default(shape=(2, 3))
    reshaped_vector2d = vector2d.reshape((6,))
    assert reshaped_vector2d.structured_type == StructuredType.BATCHED
    assert reshaped_vector2d.batch_shape == (6,)
    assert reshaped_vector2d.position.shape == (6, 3)
    assert reshaped_vector2d.velocity.shape == (6, 3)

    matrix = MatrixData.default(shape=(10,))
    reshaped_matrix = matrix.reshape((2, 5))
    assert reshaped_matrix.structured_type == StructuredType.BATCHED
    assert reshaped_matrix.batch_shape == (2, 5)
    assert reshaped_matrix.matrix.shape == (2, 5, 2, 2)

    matrix2d = MatrixData.default(shape=(2, 3))
    reshaped_matrix2d = matrix2d.reshape((6,))
    assert reshaped_matrix2d.structured_type == StructuredType.BATCHED
    assert reshaped_matrix2d.batch_shape == (6,)
    assert reshaped_matrix2d.matrix.shape == (6, 2, 2)
    assert reshaped_matrix2d.flags.shape == (6, 4)

    nested = NestedData.default(shape=(10,))
    reshaped_nested = nested.reshape((2, 5))
    assert reshaped_nested.structured_type == StructuredType.BATCHED
    assert reshaped_nested.batch_shape == (2, 5)
    assert reshaped_nested.simple.id.shape == (2, 5)
    assert reshaped_nested.simple.value.shape == (2, 5)

    nested2d = NestedData.default(shape=(2, 3))
    reshaped_nested2d = nested2d.reshape((6,))
    assert reshaped_nested2d.structured_type == StructuredType.BATCHED
    assert reshaped_nested2d.batch_shape == (6,)
    assert reshaped_nested2d.simple.id.shape == (6,)
    assert reshaped_nested2d.simple.value.shape == (6,)
    

def test_flatten():
    # Test flatten functionality
    batched = SimpleData.default(shape=(2, 3))
    flattened = batched.flatten()
    print(flattened.structured_type)
    assert flattened.structured_type == StructuredType.BATCHED
    assert flattened.batch_shape == (6,)
    assert flattened.id.shape == (6,)
    assert flattened.value.shape == (6,)

    batched2d = SimpleData.default(shape=(2, 3))
    flattened2d = batched2d.flatten()
    assert flattened2d.structured_type == StructuredType.BATCHED
    assert flattened2d.batch_shape == (6,)
    assert flattened2d.id.shape == (6,)
    assert flattened2d.value.shape == (6,)

    vector = VectorData.default(shape=(2, 3))
    flattened_vector = vector.flatten()
    assert flattened_vector.structured_type == StructuredType.BATCHED
    assert flattened_vector.batch_shape == (6,)
    assert flattened_vector.position.shape == (6, 3)
    assert flattened_vector.velocity.shape == (6, 3)

    matrix = MatrixData.default(shape=(2, 3))
    flattened_matrix = matrix.flatten()
    assert flattened_matrix.structured_type == StructuredType.BATCHED
    assert flattened_matrix.batch_shape == (6,)
    assert flattened_matrix.matrix.shape == (6, 2, 2)
    assert flattened_matrix.flags.shape == (6, 4)

    nested = NestedData.default(shape=(2, 3))
    flattened_nested = nested.flatten()
    assert flattened_nested.structured_type == StructuredType.BATCHED
    assert flattened_nested.batch_shape == (6,)
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
    assert sliced.batch_shape == (2,)
    assert sliced.id.shape == (2,)
    assert sliced.value.shape == (2,) 