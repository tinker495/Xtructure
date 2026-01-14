"""Tests for new xtructure dataclass instance methods."""

import jax.numpy as jnp
from tests.xnp.shared_data import SimpleData


def test_method_swapaxes():
    d = SimpleData.default(shape=(2, 3))
    d = d.replace(id=jnp.arange(6, dtype=jnp.uint32).reshape(2, 3))
    swapped = d.swapaxes(0, 1)
    assert swapped.shape.batch == (3, 2)
    assert jnp.array_equal(swapped.id, jnp.swapaxes(d.id, 0, 1))


def test_method_moveaxis():
    d = SimpleData.default(shape=(2, 3, 4))
    moved = d.moveaxis(2, 0)
    assert moved.shape.batch == (4, 2, 3)


def test_method_squeeze():
    d = SimpleData.default(shape=(1, 3))
    squeezed = d.squeeze(axis=0)
    assert squeezed.shape.batch == (3,)


def test_method_expand_dims():
    d = SimpleData.default(shape=(3,))
    expanded = d.expand_dims(axis=0)
    assert expanded.shape.batch == (1, 3)
    

def test_method_roll():
    d = SimpleData(id=jnp.array([1, 2, 3]), value=jnp.array([1., 2., 3.]))
    rolled = d.roll(1)
    assert jnp.array_equal(rolled.id, jnp.array([3, 1, 2]))


def test_method_flip():
    d = SimpleData(id=jnp.array([1, 2, 3]), value=jnp.array([1., 2., 3.]))
    flipped = d.flip()
    assert jnp.array_equal(flipped.id, jnp.array([3, 2, 1]))


def test_method_rot90():
    d = SimpleData.default(shape=(2, 2))
    d = d.replace(id=jnp.array([[0, 1], [2, 3]]))
    rotated = d.rot90()
    assert rotated.id[0, 0] == 1


def test_method_broadcast_to():
    d = SimpleData(id=jnp.array([1]), value=jnp.array([1.0]))
    broadcasted = d.broadcast_to((3,))
    assert broadcasted.shape.batch == (3,)
    assert jnp.array_equal(broadcasted.id, jnp.array([1, 1, 1]))


def test_method_astype():
    d = SimpleData(id=jnp.array([1]), value=jnp.array([1.0]))
    d_float = d.astype(jnp.float32)
    assert d_float.id.dtype == jnp.float32
