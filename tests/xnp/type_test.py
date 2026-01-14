"""Tests for xnp type operators: astype, result_type, can_cast."""

import jax.numpy as jnp
from tests.xnp.shared_data import SimpleData
from xtructure import numpy as xnp


def test_astype():
    d = SimpleData(id=jnp.array([1]), value=jnp.array([1.0]))

    d_float = xnp.astype(d, jnp.float32)
    assert d_float.id.dtype == jnp.float32


def test_result_type():
    d = SimpleData(id=jnp.array([1]), value=jnp.array([1.0]))
    d_int = SimpleData(id=jnp.array([1]), value=jnp.array([1]))

    res = xnp.result_type(d, d_int)
    assert res.value == jnp.float32 or res.value == jnp.float64


def test_can_cast():
    d = SimpleData(id=jnp.array([1]), value=jnp.array([1.0]))
    d_int = SimpleData(id=jnp.array([1]), value=jnp.array([1]))

    # int32 -> float64 is safe
    assert xnp.can_cast(d_int, jnp.float64, "safe")
    # float to int is unsafe
    assert not xnp.can_cast(d, jnp.int32, "safe")
