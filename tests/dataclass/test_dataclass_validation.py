import jax.numpy as jnp
import pytest

from tests.dataclass.fixtures import SimpleData, VectorData
from xtructure import FieldDescriptor, Xtructurable


class ValidatedScalarData(Xtructurable, validate=True):
    value: FieldDescriptor.scalar(dtype=jnp.float32)


class ValidatedVectorData(Xtructurable, validate=True):
    vector: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))


class ValidatedNestedData(Xtructurable, validate=True):
    simple: FieldDescriptor.scalar(dtype=SimpleData)


class ValidatedWithPostInit(Xtructurable, validate=True):
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
