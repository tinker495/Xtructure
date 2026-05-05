import chex
import jax.numpy as jnp
import numpy as np
import pytest

from xtructure import FieldDescriptor, xtructure_dataclass
from xtructure.io.bitpack import packed_num_bytes


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
class MixedPackedState:
    """One packed field plus one plain field — exercises `from_unpacked` partial path."""

    faces: FieldDescriptor.packed_tensor(
        unpacked_dtype=jnp.uint8,
        shape=(6, 9),
        packed_bits=3,
    )
    counter: FieldDescriptor.scalar(dtype=jnp.int32)


def test_packed_tensor_descriptor_shape():
    num_values = 6 * 9
    expected = packed_num_bytes(num_values, 3)
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


def test_from_unpacked_partial_path_uses_default_for_missing_fields():
    """Providing only some fields exercises the partial path (default + replace)."""
    raw = jnp.arange(6 * 9, dtype=jnp.uint8).reshape((6, 9)) & jnp.uint8(7)
    state = MixedPackedState.from_unpacked(faces=raw)

    # Missing `counter` should fall back to the dataclass default.
    chex.assert_trees_all_equal(state.faces_unpacked, raw)
    assert state.counter == MixedPackedState.default().counter


def test_from_unpacked_full_path_packs_all_fields():
    """Providing all fields exercises the direct-build path with no default allocation."""
    raw = jnp.arange(6 * 9, dtype=jnp.uint8).reshape((6, 9)) & jnp.uint8(7)
    state = MixedPackedState.from_unpacked(faces=raw, counter=jnp.int32(42))

    chex.assert_trees_all_equal(state.faces_unpacked, raw)
    assert int(state.counter) == 42


def test_from_unpacked_unknown_field_raises_key_error():
    """Unknown kwargs must raise KeyError, not silently fall through to replace."""
    raw = jnp.arange(6 * 9, dtype=jnp.uint8).reshape((6, 9)) & jnp.uint8(7)
    with pytest.raises(KeyError, match="bogus"):
        MixedPackedState.from_unpacked(faces=raw, bogus=jnp.int32(1))
