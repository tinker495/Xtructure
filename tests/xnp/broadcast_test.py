"""Tests for xnp shape/broadcast operators: moveaxis, broadcast_to, broadcast_arrays, atleast_nd."""

import jax.numpy as jnp
from tests.xnp.shared_data import SimpleData
from xtructure import numpy as xnp


def test_moveaxis():
    data = SimpleData.default(shape=(2, 3, 4))
    moved = xnp.moveaxis(data, 2, 0)
    assert moved.shape.batch == (4, 2, 3)
    assert moved.id.shape == (4, 2, 3)


def test_broadcast_to():
    data = SimpleData(id=jnp.array([1]), value=jnp.array([10.0]))

    broadcasted = xnp.broadcast_to(data, (3, 2))
    assert broadcasted.shape.batch == (3, 2)
    assert broadcasted.id.shape == (3, 2)
    assert broadcasted.value.shape == (3, 2)


def test_broadcast_arrays():
    d1 = SimpleData(id=jnp.array([1]), value=jnp.array([10.0]))
    d2 = SimpleData(id=jnp.zeros((3, 1), dtype=int), value=jnp.zeros((3, 1)))

    b1, b2 = xnp.broadcast_arrays(d1, d2)
    assert b1.shape.batch == (3, 1)
    assert b2.shape.batch == (3, 1)


def test_atleast_1d():
    d = SimpleData(id=jnp.array(1), value=jnp.array(1.0))
    d1 = xnp.atleast_1d(d)
    assert d1.id.ndim >= 1


def test_atleast_2d():
    d = SimpleData(id=jnp.array(1), value=jnp.array(1.0))
    d2 = xnp.atleast_2d(d)
    assert d2.id.ndim >= 2


def test_atleast_3d():
    d = SimpleData(id=jnp.array(1), value=jnp.array(1.0))
    d3 = xnp.atleast_3d(d)
    assert d3.id.ndim >= 3
