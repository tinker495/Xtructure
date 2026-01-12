import chex
import jax.numpy as jnp

from xtructure.core.xtructure_decorators.aggregate_bitpack.bitpack import (
    from_uint8,
    to_uint8,
)


def test_bitpack_12bit_roundtrip_small():
    x = jnp.array([0, 1, 2, 4095, 17, 1234], dtype=jnp.uint32)
    packed = to_uint8(x, active_bits=12)
    out = from_uint8(packed, target_shape=x.shape, active_bits=12)
    assert out.dtype == jnp.uint32
    chex.assert_trees_all_equal(out, x)


def test_bitpack_16bit_roundtrip_small():
    x = jnp.array([0, 1, 2, 65535, 17, 1234], dtype=jnp.uint32)
    packed = to_uint8(x, active_bits=16)
    out = from_uint8(packed, target_shape=x.shape, active_bits=16)
    assert out.dtype == jnp.uint32
    chex.assert_trees_all_equal(out, x)
