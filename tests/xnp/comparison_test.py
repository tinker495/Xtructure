"""Tests for xnp comparison operators: equal, not_equal, isclose, allclose."""

import jax.numpy as jnp
from tests.xnp.shared_data import SimpleData
from xtructure import numpy as xnp


def test_equal_comparison():
    data1 = SimpleData(id=jnp.array([1, 2]), value=jnp.array([1.0, 2.0]))
    data2 = SimpleData(id=jnp.array([1, 3]), value=jnp.array([1.0, 3.0]))

    eq = xnp.equal(data1, data2)
    assert jnp.array_equal(eq.id, jnp.array([True, False]))
    assert jnp.array_equal(eq.value, jnp.array([True, False]))


def test_not_equal_comparison():
    data1 = SimpleData(id=jnp.array([1, 2]), value=jnp.array([1.0, 2.0]))
    data2 = SimpleData(id=jnp.array([1, 3]), value=jnp.array([1.0, 3.0]))

    neq = xnp.not_equal(data1, data2)
    assert jnp.array_equal(neq.id, jnp.array([False, True]))
    assert jnp.array_equal(neq.value, jnp.array([False, True]))


def test_isclose():
    data1 = SimpleData(id=jnp.array([1]), value=jnp.array([1.0]))
    data2 = SimpleData(id=jnp.array([1]), value=jnp.array([1.0000001]))

    close = xnp.isclose(data1, data2)
    assert jnp.all(close.value)


def test_allclose():
    data1 = SimpleData(id=jnp.array([1]), value=jnp.array([1.0]))
    data2 = SimpleData(id=jnp.array([1]), value=jnp.array([1.0000001]))

    assert xnp.allclose(data1, data2)

    data3 = SimpleData(id=jnp.array([1]), value=jnp.array([2.0]))
    assert not xnp.allclose(data1, data3)
