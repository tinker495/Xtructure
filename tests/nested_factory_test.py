import jax.numpy as jnp

from xtructure import FieldDescriptor, Xtructurable


class Inner(Xtructurable):
    val: FieldDescriptor.scalar(dtype=jnp.int32)


class OuterScalar(Xtructurable):
    inner: FieldDescriptor.scalar(dtype=Inner)


class OuterTensor(Xtructurable):
    inner_array: FieldDescriptor.tensor(dtype=Inner, shape=(2,))


def test_nested_scalar_api():
    obj = OuterScalar.default()
    assert obj.inner.val.shape == ()
    assert isinstance(obj.inner, Inner)


def test_nested_tensor_api():
    obj = OuterTensor.default()
    assert obj.inner_array.val.shape == (2,)
    assert isinstance(obj.inner_array, Inner)
    # Check if it behaves like a batched xtructure
    assert obj.inner_array.shape.batch == (2,)
