import jax.numpy as jnp

from xtructure import FieldDescriptor, clone_field_descriptor


def test_clone_field_descriptor_overrides_dtype_and_fill():
    original = FieldDescriptor(jnp.float32, (2,))
    cloned = clone_field_descriptor(original, dtype=jnp.int32, fill_value=5)
    assert cloned.dtype == jnp.int32
    assert cloned.intrinsic_shape == (2,)
    assert cloned.fill_value == 5
