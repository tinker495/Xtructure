import jax.numpy as jnp

from xtructure import (
    FieldDescriptor,
    broadcast_intrinsic_shape,
    clone_field_descriptor,
    descriptor_metadata,
    with_intrinsic_shape,
)


def test_clone_field_descriptor_overrides_dtype_and_fill():
    original = FieldDescriptor(jnp.float32, (2,))
    cloned = clone_field_descriptor(original, dtype=jnp.int32, fill_value=5)
    assert cloned.dtype == jnp.int32
    assert cloned.intrinsic_shape == (2,)
    assert cloned.fill_value == 5


def test_with_intrinsic_shape_updates_shape_only():
    original = FieldDescriptor(jnp.float32, (2,))
    updated = with_intrinsic_shape(original, (4, 4))
    assert updated.dtype == original.dtype
    assert updated.intrinsic_shape == (4, 4)


def test_broadcast_intrinsic_shape_prepends_batch_dims():
    original = FieldDescriptor(jnp.float32, (3,))
    broadcasted = broadcast_intrinsic_shape(original, (8, 2))
    assert broadcasted.intrinsic_shape == (8, 2, 3)


def test_descriptor_metadata_exposes_core_attributes():
    descriptor = FieldDescriptor(jnp.float32, (1, 2), fill_value=0.0)
    meta = descriptor_metadata(descriptor)
    assert meta["dtype"] == descriptor.dtype
    assert meta["intrinsic_shape"] == descriptor.intrinsic_shape
    assert meta["fill_value"] == descriptor.fill_value
    assert meta["fill_value_factory"] == descriptor.fill_value_factory
