import jax.numpy as jnp

from xtructure import FieldDescriptor, clone_field_descriptor, descriptor_metadata


def test_field_descriptor_tensor_factory():
    fd = FieldDescriptor.tensor(dtype=jnp.float32, shape=(3, 4), fill_value=1.0)
    assert fd.dtype == jnp.float32
    assert fd.intrinsic_shape == (3, 4)
    assert fd.fill_value == 1.0
    assert fd.validator is None


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
