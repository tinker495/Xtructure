import chex
import jax
import jax.numpy as jnp
import numpy as np

from xtructure import FieldDescriptor, xtructure_dataclass
from xtructure.core.layout.bitpack import packed_num_bytes


@xtructure_dataclass(validate=True)
class PackedState:
    # Store packed bytes, expose faces_unpacked and set_unpacked(faces=...)
    faces: FieldDescriptor.packed_tensor(
        unpacked_dtype=jnp.uint8,
        shape=(6, 9),
        packed_bits=3,
    )


@xtructure_dataclass(validate=True)
class PackedBoolState:
    flags: FieldDescriptor.packed_tensor(
        unpacked_dtype=jnp.bool_,
        shape=(17,),
        packed_bits=1,
    )


@xtructure_dataclass(validate=True)
class WidePackedState:
    values: FieldDescriptor.packed_tensor(
        unpacked_dtype=jnp.uint32,
        shape=(5,),
        packed_bits=27,
    )


def test_packed_tensor_logical_descriptor_storage_layout_shape():
    num_values = 6 * 9
    expected = packed_num_bytes(num_values, 3)
    descriptor = FieldDescriptor.packed_tensor(shape=(6, 9), packed_bits=3)

    assert descriptor.intrinsic_shape == (6, 9)
    assert PackedState.default_shape.faces == (expected,)


def test_packed_field_roundtrip_single():
    raw = jnp.arange(6 * 9, dtype=jnp.uint8).reshape((6, 9)) & jnp.uint8(7)
    state2 = PackedState.from_unpacked(faces=raw)

    assert state2.faces.dtype == jnp.uint8
    assert state2.faces.shape == (packed_num_bytes(6 * 9, 3),)
    chex.assert_trees_all_equal(state2.faces_unpacked, raw)


def test_packed_field_roundtrip_batched():
    raw = jnp.arange(2 * 6 * 9, dtype=jnp.uint8).reshape((2, 6, 9)) & jnp.uint8(7)
    state2 = PackedState.from_unpacked(shape=(2,), faces=raw)

    assert state2.faces.shape == (2, packed_num_bytes(6 * 9, 3))
    chex.assert_trees_all_equal(state2.faces_unpacked, raw)


def test_packed_bool_roundtrip():
    raw = jnp.array(np.random.RandomState(0).randint(0, 2, size=(17,)), dtype=jnp.bool_)
    state2 = PackedBoolState.from_unpacked(flags=raw)
    assert state2.flags.dtype == jnp.uint8
    chex.assert_trees_all_equal(state2.flags_unpacked, raw)


def test_packed_field_unpacks_on_host_without_device_transfer():
    raw = jnp.array(
        [0, 1, (1 << 26) - 1, (1 << 27) - 2, (1 << 27) - 1],
        dtype=jnp.uint32,
    )
    state = WidePackedState.from_unpacked(values=raw)
    host_state = jax.device_get(state)

    with jax.transfer_guard("disallow"):
        host_values = host_state.values_unpacked

    assert isinstance(host_values, np.ndarray)
    np.testing.assert_array_equal(host_values, np.asarray(raw))
