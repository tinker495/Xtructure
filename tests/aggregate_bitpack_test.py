import chex
import jax.numpy as jnp

from xtructure import FieldDescriptor, xtructure_dataclass


@xtructure_dataclass(validate=True, aggregate_bitpack=True)
class AggState:
    # 1-bit flags
    flags: FieldDescriptor.tensor(dtype=jnp.bool_, shape=(17,), bits=1, fill_value=False)
    # 3-bit values in [0,7]
    faces: FieldDescriptor.tensor(dtype=jnp.uint8, shape=(6, 9), bits=3, fill_value=0)
    # 12-bit values -> should unpack to uint32 by default
    codes: FieldDescriptor.tensor(dtype=jnp.uint16, shape=(5,), bits=12, fill_value=0)


def test_aggregate_pack_roundtrip_single():
    s = AggState.default()
    s = s.replace(
        flags=jnp.array([True] * 17, dtype=jnp.bool_),
        faces=(jnp.arange(54, dtype=jnp.uint8).reshape((6, 9)) & jnp.uint8(7)),
        codes=jnp.array([1, 2, 4095, 17, 1234], dtype=jnp.uint16),
    )

    p = s.packed
    assert p.words.dtype == jnp.uint32
    assert p.tail.dtype == jnp.uint8

    u = p.unpacked
    chex.assert_trees_all_equal(u.flags, s.flags)
    chex.assert_trees_all_equal(u.faces, s.faces.astype(jnp.uint8))
    # codes should come back as uint32 by default for bits>8
    assert u.codes.dtype == jnp.uint32
    chex.assert_trees_all_equal(u.codes, s.codes.astype(jnp.uint32))

    # Reconstruction to original declared dtypes should pass validation.
    o = p.as_original()
    assert o.codes.dtype == jnp.uint16
    chex.assert_trees_all_equal(o, s)


def test_aggregate_pack_roundtrip_batched():
    s = AggState.default(shape=(2,))
    s = s.replace(
        flags=jnp.array([[True] * 17, [False] * 17], dtype=jnp.bool_),
        faces=jnp.stack(
            [
                (jnp.arange(54, dtype=jnp.uint8).reshape((6, 9)) & jnp.uint8(7)),
                (jnp.arange(54, dtype=jnp.uint8).reshape((6, 9)) & jnp.uint8(7)),
            ],
            axis=0,
        ),
        codes=jnp.array([[1, 2, 3, 4, 5], [4095, 0, 17, 32, 999]], dtype=jnp.uint16),
    )

    p = s.packed
    u = p.unpacked
    chex.assert_trees_all_equal(u.flags, s.flags)
    chex.assert_trees_all_equal(u.faces, s.faces.astype(jnp.uint8))
    chex.assert_trees_all_equal(u.codes, s.codes.astype(jnp.uint32))

    o = p.as_original()
    chex.assert_trees_all_equal(o, s)


@xtructure_dataclass(validate=True)
class AggStateAuto:
    # Same structure as AggState, but rely on auto aggregate activation (all fields have bits, no nesting)
    flags: FieldDescriptor.tensor(dtype=jnp.bool_, shape=(17,), bits=1, fill_value=False)
    faces: FieldDescriptor.tensor(dtype=jnp.uint8, shape=(6, 9), bits=3, fill_value=0)
    codes: FieldDescriptor.tensor(dtype=jnp.uint16, shape=(5,), bits=12, fill_value=0)


def test_aggregate_auto_enabled_when_all_fields_have_bits():
    s = AggStateAuto.default()
    p = s.packed
    u = p.unpacked
    assert hasattr(AggStateAuto, "Packed")
    assert p.words.dtype == jnp.uint32
    assert u.codes.dtype == jnp.uint32


def test_aggregate_unpack_dtype_policy_and_schema():
    s = AggState.default()
    p = s.packed
    u_default = p.unpack(dtype_policy="default")
    o_declared = p.unpack(dtype_policy="declared")
    assert u_default.__class__.__name__.endswith("Unpacked")
    assert o_declared.__class__.__name__ == "AggState"
    schema = AggState.bitpack_schema()
    assert schema["mode"] == "aggregate"
    assert (
        schema["storage_bytes"]
        == AggState.Packed.default_shape.words[0] * 4 + AggState.Packed.default_shape.tail[0]
    )


def test_aggregate_unpack_field_partial():
    s = AggState.default(shape=(2,))
    s = s.replace(
        flags=jnp.array([[True] * 17, [False] * 17], dtype=jnp.bool_),
        faces=jnp.stack(
            [
                (jnp.arange(54, dtype=jnp.uint8).reshape((6, 9)) & jnp.uint8(7)),
                (jnp.arange(54, dtype=jnp.uint8).reshape((6, 9)) & jnp.uint8(7)),
            ],
            axis=0,
        ),
        codes=jnp.array([[1, 2, 3, 4, 5], [4095, 0, 17, 32, 999]], dtype=jnp.uint16),
    )
    p = s.packed

    # Decode only one field (full), without materializing all fields.
    faces = p.unpack_field("faces")
    chex.assert_trees_all_equal(faces, s.faces.astype(jnp.uint8))

    # Decode a subset of flattened indices for one field.
    idxs = jnp.array([0, 2, 4], dtype=jnp.int32)
    codes_subset = p.unpack_field("codes", indices=idxs)
    chex.assert_trees_all_equal(codes_subset, s.codes.astype(jnp.uint32)[:, [0, 2, 4]])

    # Works on batch-sliced packed objects too.
    p0 = p[0]
    faces0 = p0.unpack_field("faces")
    chex.assert_trees_all_equal(faces0, s.faces.astype(jnp.uint8)[0])


@xtructure_dataclass(validate=True)
class InnerState:
    # 4-bit values
    a: FieldDescriptor.tensor(dtype=jnp.uint8, shape=(6,), bits=4, fill_value=0)
    # 12-bit values
    codes: FieldDescriptor.tensor(dtype=jnp.uint16, shape=(5,), bits=12, fill_value=0)


@xtructure_dataclass(validate=True, aggregate_bitpack=True)
class OuterState:
    inner: FieldDescriptor.scalar(dtype=InnerState)
    flags: FieldDescriptor.tensor(dtype=jnp.bool_, shape=(7,), bits=1, fill_value=False)


def test_aggregate_nested_roundtrip_and_partial():
    s = OuterState.default(shape=(2,))
    s = s.replace(
        inner=InnerState(
            a=jnp.array([[1, 2, 3, 4, 5, 6], [7, 0, 1, 2, 3, 4]], dtype=jnp.uint8),
            codes=jnp.array([[1, 2, 3, 4, 5], [4095, 0, 17, 32, 999]], dtype=jnp.uint16),
        ),
        flags=jnp.array([[True] * 7, [False] * 7], dtype=jnp.bool_),
    )

    p = s.packed
    u = p.unpacked

    # Nested view should preserve structure.
    assert hasattr(u, "inner")
    chex.assert_trees_all_equal(u.flags, s.flags)
    chex.assert_trees_all_equal(u.inner.a, s.inner.a.astype(jnp.uint8))
    chex.assert_trees_all_equal(u.inner.codes, s.inner.codes.astype(jnp.uint32))

    # Declared reconstruction should match exactly (and pass validation).
    o = p.unpack(dtype_policy="declared")
    chex.assert_trees_all_equal(o, s)

    # Partial decode for nested leaf via dotted path.
    codes_subset = p.unpack_field("inner.codes", indices=[0, 2, 4])
    chex.assert_trees_all_equal(codes_subset, s.inner.codes.astype(jnp.uint32)[:, [0, 2, 4]])
