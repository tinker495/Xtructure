"""Tests for xnp.where and xnp.where_no_broadcast helpers."""

import jax
import jax.numpy as jnp
import pytest

from tests.xnp.shared_data import SimpleData
from xtructure import numpy as xnp


def test_where_with_dataclasses():
    """Test xnp.where with two dataclasses"""
    dc1 = SimpleData.default(shape=(3,))
    dc1 = dc1.replace(id=jnp.array([1, 2, 3]), value=jnp.array([1.0, 2.0, 3.0]))
    dc2 = SimpleData.default(shape=(3,))
    dc2 = dc2.replace(id=jnp.array([10, 20, 30]), value=jnp.array([10.0, 20.0, 30.0]))
    condition = jnp.array([True, False, True])

    result = xnp.where(condition, dc1, dc2)

    expected_id = jnp.array([1, 20, 3])
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

    expected_id = jnp.array([1, -1, 3])
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

    expected_id = jnp.array([1, 20])
    expected_value = jnp.array([1.0, 20.0])

    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_where_scalar_condition():
    """Test xnp.where with scalar boolean condition"""
    dc1 = SimpleData.default(shape=(3,))
    dc1 = dc1.replace(id=jnp.array([1, 2, 3]), value=jnp.array([1.0, 2.0, 3.0]))
    dc2 = SimpleData.default(shape=(3,))
    dc2 = dc2.replace(id=jnp.array([10, 20, 30]), value=jnp.array([10.0, 20.0, 30.0]))

    result_true = xnp.where(True, dc1, dc2)
    assert jnp.array_equal(result_true.id, dc1.id)
    assert jnp.array_equal(result_true.value, dc1.value)

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

    result_xnp = xnp.where(condition, dc1, dc2)
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

    result_xnp = xnp.where(condition, dc, fallback)
    result_manual = jax.tree_util.tree_map(
        lambda x: jnp.where(condition, x, fallback), dc
    )

    assert jnp.array_equal(result_xnp.id, result_manual.id)
    assert jnp.array_equal(result_xnp.value, result_manual.value)


def test_where_no_broadcast_basic():
    """where_no_broadcast succeeds when shapes and dtypes match exactly."""
    dc1 = SimpleData.default(shape=(3,))
    dc1 = dc1.replace(
        id=jnp.array([1, 2, 3], dtype=jnp.uint32), value=jnp.array([1.0, 2.0, 3.0])
    )
    dc2 = SimpleData.default(shape=(3,))
    dc2 = dc2.replace(
        id=jnp.array([10, 20, 30], dtype=jnp.uint32),
        value=jnp.array([10.0, 20.0, 30.0]),
    )
    condition = jnp.array([True, False, True])

    result = xnp.where_no_broadcast(condition, dc1, dc2)

    expected_id = jnp.array([1, 20, 3], dtype=jnp.uint32)
    expected_value = jnp.array([1.0, 20.0, 3.0])

    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_where_no_broadcast_rejects_type_mismatch():
    """where_no_broadcast rejects dataclass/scalar combinations to avoid broadcasting."""
    dc = SimpleData.default(shape=(2,))
    dc = dc.replace(id=jnp.array([1, 2], dtype=jnp.uint32), value=jnp.array([1.0, 2.0]))
    condition = jnp.array([True, False])

    with pytest.raises(TypeError):
        xnp.where_no_broadcast(condition, dc, 0)


def test_where_no_broadcast_rejects_shape_mismatch():
    """where_no_broadcast raises when mask shape differs from field shape."""
    dc1 = SimpleData.default(shape=(3,))
    dc1 = dc1.replace(
        id=jnp.array([1, 2, 3], dtype=jnp.uint32), value=jnp.array([1.0, 2.0, 3.0])
    )
    dc2 = SimpleData.default(shape=(3,))
    dc2 = dc2.replace(
        id=jnp.array([10, 20, 30], dtype=jnp.uint32),
        value=jnp.array([10.0, 20.0, 30.0]),
    )
    condition = jnp.array([True, False])

    with pytest.raises(ValueError):
        xnp.where_no_broadcast(condition, dc1, dc2)
