import jax.numpy as jnp
import pytest

from xtructure import FieldDescriptor, clone_field_descriptor, descriptor_metadata
from xtructure.core.dtype_facts import (
    DTypeKind,
    default_fill_value_for_dtype,
    dtype_kind,
)
from xtructure.core.layout import get_type_layout
from xtructure.io.bitpack import packed_num_bytes


def test_field_descriptor_tensor_factory():
    fd = FieldDescriptor.tensor(dtype=jnp.float32, shape=(3, 4), fill_value=1.0)
    assert fd.dtype == jnp.float32
    assert fd.intrinsic_shape == (3, 4)
    assert fd.fill_value == 1.0
    assert fd.validator is None


def test_field_descriptor_normalizes_int_intrinsic_shape():
    fd = FieldDescriptor(dtype=jnp.float32, intrinsic_shape=3)
    assert fd.intrinsic_shape == (3,)


def test_packed_tensor_normalizes_int_shape():
    fd = FieldDescriptor.packed_tensor(shape=3, packed_bits=1)
    assert fd.intrinsic_shape == (packed_num_bytes(3, 1),)
    assert fd.unpacked_intrinsic_shape == (3,)


def test_packed_tensor_accepts_equivalent_normalized_shape_aliases():
    fd = FieldDescriptor.packed_tensor(shape=3, unpacked_shape=(3,), packed_bits=1)
    assert fd.unpacked_intrinsic_shape == (3,)


def test_packed_tensor_rejects_mismatched_normalized_shape_aliases():
    with pytest.raises(ValueError):
        FieldDescriptor.packed_tensor(shape=3, unpacked_shape=(4,), packed_bits=1)


@pytest.mark.parametrize(
    ("dtype", "expected_kind", "expected_fill"),
    [
        (jnp.bool_, DTypeKind.BOOL, False),
        (jnp.uint8, DTypeKind.UINT, jnp.iinfo(jnp.uint8).max),
        (jnp.int32, DTypeKind.INT, 0),
        (jnp.float32, DTypeKind.FLOAT, jnp.inf),
    ],
)
def test_dtype_kind_classifier_owns_default_fill_values(dtype, expected_kind, expected_fill):
    assert dtype_kind(dtype) is expected_kind
    assert default_fill_value_for_dtype(dtype) == expected_fill


def test_dtype_kind_rejects_unsupported_primitive_dtype():
    with pytest.raises(TypeError, match="DType Kind"):
        dtype_kind(jnp.complex64)


def test_implicit_field_descriptor_fill_value_comes_from_field_layout():
    descriptor = FieldDescriptor.scalar(dtype=jnp.uint8)

    assert descriptor.fill_value is None

    from xtructure import xtructure_dataclass

    @xtructure_dataclass(bitpack="off")
    class LayoutAuthoritativeFill:
        value: descriptor

    layout = get_type_layout(LayoutAuthoritativeFill)

    assert layout.field_for("value").fill_value == jnp.iinfo(jnp.uint8).max
    assert LayoutAuthoritativeFill.default().value == jnp.iinfo(jnp.uint8).max


def test_packed_tensor_uses_packed_data_kind_default_unpack_dtype():
    assert FieldDescriptor.packed_tensor(shape=(3,), packed_bits=1).unpacked_dtype is jnp.bool_
    assert FieldDescriptor.packed_tensor(shape=(3,), packed_bits=8).unpacked_dtype is jnp.uint8
    assert FieldDescriptor.packed_tensor(shape=(3,), packed_bits=12).unpacked_dtype is jnp.uint32


def test_field_descriptor_scalar_factory():
    fd = FieldDescriptor.scalar(dtype=jnp.int32, default=10)
    assert fd.dtype == jnp.int32
    assert fd.intrinsic_shape == ()
    assert fd.fill_value == 10
    assert fd.validator is None


def test_field_descriptor_with_validator():
    def my_validator(x):
        if x < 0:
            raise ValueError("Negative")

    fd = FieldDescriptor.scalar(dtype=jnp.int32, validator=my_validator)
    assert fd.validator is my_validator

    meta = descriptor_metadata(fd)
    assert meta["validator"] is my_validator


def test_clone_field_descriptor_preserves_validator():
    def val(x):
        pass

    original = FieldDescriptor(jnp.int32, validator=val)
    cloned = clone_field_descriptor(original, dtype=jnp.float32)

    assert cloned.dtype == jnp.float32
    assert cloned.validator is val


def test_clone_field_descriptor_overrides_validator():
    def val1(x):
        pass

    def val2(x):
        pass

    original = FieldDescriptor(jnp.int32, validator=val1)
    cloned = clone_field_descriptor(original, validator=val2)

    assert cloned.validator is val2
