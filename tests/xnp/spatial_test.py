"""Tests for xnp spatial operators: roll, flip, rot90."""

import jax.numpy as jnp

from tests.xnp.shared_data import SimpleData
from xtructure import numpy as xnp


def test_roll():
    d = SimpleData(id=jnp.array([1, 2, 3]), value=jnp.array([1.0, 2.0, 3.0]))

    r = xnp.roll(d, 1)
    assert jnp.array_equal(r.id, jnp.array([3, 1, 2]))


def test_flip():
    d = SimpleData(id=jnp.array([1, 2, 3]), value=jnp.array([1.0, 2.0, 3.0]))

    f = xnp.flip(d)
    assert jnp.array_equal(f.id, jnp.array([3, 2, 1]))


def test_rot90():
    d = SimpleData.default(shape=(2, 2))
    d = d.replace(id=jnp.array([[0, 1], [2, 3]]))

    rot = xnp.rot90(d)
    assert rot.id[0, 0] == 1
    assert rot.id[0, 1] == 3
