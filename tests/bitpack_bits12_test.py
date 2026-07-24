import chex
import jax.numpy as jnp
import pytest

from xtructure.core.layout.bitpack import from_uint8, to_uint8


@pytest.mark.parametrize("active_bits", range(1, 33))
def test_bitpack_roundtrip_all_supported_widths(active_bits):
    max_value = (1 << active_bits) - 1
    values = jnp.array(
        [
            0,
            1,
            max_value >> 2,
            max_value >> 1,
            max_value - 1,
            max_value,
            2 & max_value,
            3 & max_value,
            max_value // 3,
            (2 * max_value) // 3,
        ],
        dtype=jnp.uint32,
    )

    packed = to_uint8(values, active_bits=active_bits)
    unpacked = from_uint8(packed, target_shape=values.shape, active_bits=active_bits).astype(
        jnp.uint32
    )

    chex.assert_trees_all_equal(unpacked, values)


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


def test_io_bitpack_rejects_unknown_dtype_kind():
    with pytest.raises(TypeError, match="DType Kind"):
        to_uint8(jnp.asarray([1 + 2j], dtype=jnp.complex64), active_bits=1)
