"""Tests for xnp stacking operators: vstack, hstack, dstack, column_stack, block."""

import jax.numpy as jnp

from tests.xnp.shared_data import SimpleData
from xtructure import numpy as xnp


def test_vstack():
    d1 = SimpleData(id=jnp.array([1]), value=jnp.array([1.0]))
    d2 = SimpleData(id=jnp.array([2]), value=jnp.array([2.0]))

    v = xnp.vstack([d1, d2])
    assert v.id.shape == (2, 1)


def test_hstack():
    d1 = SimpleData(id=jnp.array([1]), value=jnp.array([1.0]))
    d2 = SimpleData(id=jnp.array([2]), value=jnp.array([2.0]))

    h = xnp.hstack([d1, d2])
    assert h.id.shape == (2,)
    assert jnp.array_equal(h.id, jnp.array([1, 2]))


def test_dstack():
    d1 = SimpleData(id=jnp.array([[1]]), value=jnp.array([[1.0]]))
    d2 = SimpleData(id=jnp.array([[2]]), value=jnp.array([[2.0]]))

    ds = xnp.dstack([d1, d2])
    assert ds.id.shape == (1, 1, 2)


def test_column_stack():
    d1 = SimpleData(id=jnp.array([1, 2]), value=jnp.array([1.0, 2.0]))
    d2 = SimpleData(id=jnp.array([3, 4]), value=jnp.array([3.0, 4.0]))

    cs = xnp.column_stack([d1, d2])
    assert cs.id.shape == (2, 2)


def test_block_horizontal():
    d1 = SimpleData(id=jnp.array([1]), value=jnp.array([1.0]))
    d2 = SimpleData(id=jnp.array([2]), value=jnp.array([2.0]))

    b_h = xnp.block([d1, d2])
    assert b_h.id.shape == (2,)


def test_block_vertical():
    d1 = SimpleData(id=jnp.array([1]), value=jnp.array([1.0]))
    d2 = SimpleData(id=jnp.array([2]), value=jnp.array([2.0]))

    b_v = xnp.block([[d1], [d2]])
    assert b_v.id.shape == (2, 1)
