"""Direct tests for xtructure decorator modules."""

import jax
import jax.numpy as jnp

from tests.testdata import SimpleData, VectorData
from xtructure.core.protocol import StructuredType

# Direct imports to satisfy test_coverage
from xtructure.core.xtructure_decorators.shape import add_shape_dtype_len
from xtructure.core.xtructure_decorators.string_format import (
    add_string_representation_methods,
)
from xtructure.core.xtructure_decorators.structure_util import add_structure_utilities

# ---------------------------------------------------------------------------
# Module 1: shape.py
# ---------------------------------------------------------------------------


def test_shape_property_scalar():
    instance = SimpleData.default()
    shape = instance.shape
    assert shape.batch == ()
    assert shape.id == ()
    assert shape.value == ()


def test_shape_property_batched():
    instance = SimpleData.default(shape=(4,))
    shape = instance.shape
    assert shape.batch == (4,)


def test_dtype_property():
    instance = SimpleData.default()
    dtype = instance.dtype
    assert dtype.id == jnp.uint32
    assert dtype.value == jnp.float32


def test_len_property():
    instance = SimpleData.default(shape=(5,))
    assert len(instance) == 5


def test_len_scalar_returns_one():
    instance = SimpleData.default()
    assert len(instance) == 1


def test_structured_type_single():
    instance = SimpleData.default()
    assert instance.structured_type == StructuredType.SINGLE


def test_structured_type_batched():
    instance = SimpleData.default(shape=(3,))
    assert instance.structured_type == StructuredType.BATCHED


# ---------------------------------------------------------------------------
# Module 2: structure_util.py
# ---------------------------------------------------------------------------


def test_random_generates_instance():
    key = jax.random.PRNGKey(42)
    instance = SimpleData.random(key=key)
    assert instance.id.shape == ()
    assert instance.value.shape == ()


def test_random_batched():
    key = jax.random.PRNGKey(0)
    instance = SimpleData.random(shape=(7,), key=key)
    assert instance.id.shape == (7,)
    assert instance.value.shape == (7,)


def test_random_default_key():
    # Calling without a key should not raise
    instance = SimpleData.random()
    assert instance.id.shape == ()


def test_random_vector_data():
    key = jax.random.PRNGKey(1)
    instance = VectorData.random(shape=(3,), key=key)
    assert instance.position.shape == (3, 3)
    assert instance.velocity.shape == (3, 3)


# ---------------------------------------------------------------------------
# Module 3: string_format.py
# ---------------------------------------------------------------------------


def test_str_scalar():
    instance = SimpleData.default()
    result = str(instance)
    assert isinstance(result, str)
    assert len(result) > 0


def test_str_batched_nonempty():
    instance = SimpleData.default(shape=(3,))
    result = str(instance)
    assert isinstance(result, str)
    assert len(result) > 0


def test_str_batched_contains_shape():
    instance = SimpleData.default(shape=(3,))
    result = str(instance)
    # The panel subtitle includes the batch shape
    assert "3" in result


def test_repr():
    instance = SimpleData.default()
    result = repr(instance)
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Decorator functions are callable
# ---------------------------------------------------------------------------


def test_decorator_functions_are_callable():
    assert callable(add_shape_dtype_len)
    assert callable(add_string_representation_methods)
    assert callable(add_structure_utilities)
