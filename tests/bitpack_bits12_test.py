import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xtructure.core.layout.bitpack import from_uint8, to_uint8


def _reference_pack(values, active_bits):
    values = np.asarray(values, dtype=np.uint32).reshape(-1)
    values_per_block = int(np.lcm(active_bits, 8)) // active_bits
    values = np.pad(values, (0, (-values.size) % values_per_block))
    shifts = np.arange(active_bits, dtype=np.uint32)
    bits = ((values[:, None] >> shifts) & np.uint32(1)).astype(np.uint8)
    return np.packbits(bits.reshape(-1), bitorder="little")


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

    np.testing.assert_array_equal(np.asarray(packed), _reference_pack(values, active_bits))
    chex.assert_trees_all_equal(unpacked, values)


@pytest.mark.parametrize("active_bits", [5, 27, 29, 30, 31])
def test_wide_bitpack_ignores_inactive_high_bits(active_bits):
    max_value = (1 << active_bits) - 1
    values = jnp.array([0xFFFFFFFF, max_value, 1], dtype=jnp.uint32)

    packed = to_uint8(values, active_bits=active_bits)
    unpacked = from_uint8(packed, target_shape=values.shape, active_bits=active_bits)

    np.testing.assert_array_equal(np.asarray(packed), _reference_pack(values, active_bits))
    np.testing.assert_array_equal(
        np.asarray(unpacked, dtype=np.uint32),
        np.asarray(values & jnp.uint32(max_value)),
    )


@pytest.mark.parametrize("active_bits", [2, 3, 4, 6, 12])
@pytest.mark.parametrize("negative", [False, True], ids=["overflow", "negative"])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_narrow_bitpack_ignores_inactive_high_bits(active_bits, negative, compiled):
    # Author: tinker495.
    # Narrow packing paths must mask inactive high bits so they cannot bleed into
    # the next value's slot; an overflowing or negative value followed by zero
    # must retain only its declared-width low bits under JIT and with block padding.
    max_value = (1 << active_bits) - 1
    values = jnp.array(
        [-1, 0, -2, 0] if negative else [max_value + 1, 0, max_value, 0],
        dtype=jnp.int32,
    )
    packed = (
        jax.jit(lambda x: to_uint8(x, active_bits=active_bits))(values)
        if compiled
        else to_uint8(values, active_bits=active_bits)
    )
    unpacked = from_uint8(packed, target_shape=values.shape, active_bits=active_bits)

    np.testing.assert_array_equal(np.asarray(packed), _reference_pack(values, active_bits))
    np.testing.assert_array_equal(
        np.asarray(unpacked, dtype=np.uint32),
        np.asarray(values, dtype=np.uint32) & np.uint32(max_value),
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


def test_io_bitpack_rejects_unknown_dtype_kind():
    with pytest.raises(TypeError, match="DType Kind"):
        to_uint8(jnp.asarray([1 + 2j], dtype=jnp.complex64), active_bits=1)
