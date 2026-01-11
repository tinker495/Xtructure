"""Tests for xnp.stack functionality."""

import jax.numpy as jnp
import pytest

from tests.xnp.shared_data import SimpleData
from xtructure import numpy as xnp


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
