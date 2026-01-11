import jax
import jax.numpy as jnp
import pytest

from tests.dataclass.fixtures import MatrixData, NestedData, SimpleData, VectorData
from xtructure import StructuredType, Xtructurable


def test_dataclass_default():
    simple = SimpleData.default()
    assert simple.id.shape == ()
    assert simple.value.shape == ()
    assert simple.id.dtype == jnp.uint32
    assert simple.value.dtype == jnp.float32

    batched = SimpleData.default(shape=(10,))
    assert batched.id.shape == (10,)
    assert batched.value.shape == (10,)


def test_dataclass_random():
    key = jax.random.PRNGKey(0)
    simple = SimpleData.random(key=key)
    assert simple.id.shape == ()
    assert simple.value.shape == ()

    batched = SimpleData.random(shape=(5,), key=key)
    assert batched.id.shape == (5,)
    assert batched.value.shape == (5,)


def test_vector_data():
    vector = VectorData.default()
    assert vector.position.shape == (3,)
    assert vector.velocity.shape == (3,)

    batched = VectorData.default(shape=(4,))
    assert batched.position.shape == (4, 3)
    assert batched.velocity.shape == (4, 3)


def test_matrix_data():
    matrix = MatrixData.default()
    assert matrix.matrix.shape == (2, 2)
    assert matrix.flags.shape == (4,)

    batched = MatrixData.default(shape=(3,))
    assert batched.matrix.shape == (3, 2, 2)
    assert batched.flags.shape == (3, 4)


def test_nested_data():
    nested = NestedData.default()
    assert nested.simple.id.shape == ()
    assert nested.simple.value.shape == ()
    assert nested.vector.position.shape == (3,)

    batched = NestedData.default(shape=(2,))
    assert batched.simple.id.shape == (2,)
    assert batched.simple.value.shape == (2,)
    assert batched.vector.position.shape == (2, 3)


def test_structured_type():
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


def test_batch_shape():
    single = SimpleData.default()
    assert single.batch_shape == ()

    batched = SimpleData.default(shape=(5, 10))
    assert batched.batch_shape == (5, 10)

    unstructured = SimpleData(id=jnp.array(1), value=jnp.array([2.0, 3.0, 4.0]))
    assert unstructured.structured_type == StructuredType.UNSTRUCTURED
    assert unstructured.batch_shape == -1


def test_len_semantics():
    single = SimpleData.default()
    assert len(single) == 1

    batched1d = SimpleData.default(shape=(5,))
    assert len(batched1d) == 5

    batched2d = SimpleData.default(shape=(5, 10))
    assert len(batched2d) == 5

    unstructured = SimpleData(id=jnp.array(1), value=jnp.array([2.0, 3.0, 4.0]))
    assert unstructured.structured_type == StructuredType.UNSTRUCTURED
    with pytest.raises(TypeError):
        len(unstructured)


def test_ndim_semantics():
    single = SimpleData.default()
    assert single.ndim == 0

    batched1d = SimpleData.default(shape=(5,))
    assert batched1d.ndim == 1

    batched2d = SimpleData.default(shape=(5, 10))
    assert batched2d.ndim == 2

    unstructured = SimpleData(id=jnp.array(1), value=jnp.array([2.0, 3.0, 4.0]))
    assert unstructured.structured_type == StructuredType.UNSTRUCTURED
    with pytest.raises(TypeError):
        _ = unstructured.ndim


def test_xtructurable_isinstance():
    assert isinstance(SimpleData, Xtructurable)
    assert isinstance(SimpleData.default(), Xtructurable)

    class Plain:
        pass

    assert not isinstance(Plain, Xtructurable)
    assert not isinstance(Plain(), Xtructurable)


def test_reshape():
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
    batched = SimpleData.default(shape=(12,))
    batched = batched.replace(
        id=jnp.arange(12, dtype=jnp.uint32), value=jnp.arange(12, dtype=jnp.float32)
    )

    reshaped1 = batched.reshape((2, -1))
    assert reshaped1.structured_type == StructuredType.BATCHED
    assert reshaped1.shape.batch == (2, 6)
    assert reshaped1.id.shape == (2, 6)
    assert reshaped1.value.shape == (2, 6)

    reshaped2 = batched.reshape((-1, 3))
    assert reshaped2.structured_type == StructuredType.BATCHED
    assert reshaped2.shape.batch == (4, 3)
    assert reshaped2.id.shape == (4, 3)
    assert reshaped2.value.shape == (4, 3)

    reshaped3 = batched.reshape((-1,))
    assert reshaped3.structured_type == StructuredType.BATCHED
    assert reshaped3.shape.batch == (12,)
    assert reshaped3.id.shape == (12,)
    assert reshaped3.value.shape == (12,)

    vector = VectorData.default(shape=(10,))
    vector = vector.replace(
        position=jnp.arange(30, dtype=jnp.float32).reshape(10, 3),
        velocity=jnp.arange(30, dtype=jnp.float32).reshape(10, 3) + 100,
    )

    reshaped_vector = vector.reshape((2, -1))
    assert reshaped_vector.structured_type == StructuredType.BATCHED
    assert reshaped_vector.shape.batch == (2, 5)
    assert reshaped_vector.position.shape == (2, 5, 3)
    assert reshaped_vector.velocity.shape == (2, 5, 3)

    matrix = MatrixData.default(shape=(8,))
    matrix = matrix.replace(
        matrix=jnp.arange(32, dtype=jnp.float32).reshape(8, 2, 2),
        flags=jnp.arange(32, dtype=jnp.int32).reshape(8, 4).astype(jnp.bool_),
    )

    reshaped_matrix = matrix.reshape((2, -1))
    assert reshaped_matrix.structured_type == StructuredType.BATCHED
    assert reshaped_matrix.shape.batch == (2, 4)
    assert reshaped_matrix.matrix.shape == (2, 4, 2, 2)
    assert reshaped_matrix.flags.shape == (2, 4, 4)

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

    reshaped_nested = nested.reshape((2, -1))
    assert reshaped_nested.structured_type == StructuredType.BATCHED
    assert reshaped_nested.shape.batch == (2, 3)
    assert reshaped_nested.simple.id.shape == (2, 3)
    assert reshaped_nested.simple.value.shape == (2, 3)
    assert reshaped_nested.vector.position.shape == (2, 3, 3)
    assert reshaped_nested.vector.velocity.shape == (2, 3, 3)


def test_reshape_with_minus_one_errors():
    batched = SimpleData.default(shape=(10,))
    batched = batched.replace(
        id=jnp.arange(10, dtype=jnp.uint32), value=jnp.arange(10, dtype=jnp.float32)
    )

    with pytest.raises(ValueError):
        batched.reshape((-1, -1))

    with pytest.raises(ValueError):
        batched.reshape((3, -1))

    with pytest.raises(ValueError):
        batched.reshape((0, -1))


def test_flatten():
    batched = SimpleData.default(shape=(2, 3))
    flattened = batched.flatten()
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

    with pytest.raises(ValueError):
        batched_unstructured.reshape((2, 3))

    with pytest.raises(ValueError):
        batched_unstructured.flatten()
