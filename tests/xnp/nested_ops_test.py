from typing import Annotated

import jax.numpy as jnp

from xtructure.core import xtructure_dataclass
from xtructure.core.field_descriptors import FieldDescriptor


@xtructure_dataclass
class Inner:
    val: Annotated[jnp.ndarray, FieldDescriptor.tensor(dtype=jnp.int32, shape=())]

    @classmethod
    def default(cls, shape=()):
        return cls(val=jnp.zeros(shape, dtype=jnp.int32))


@xtructure_dataclass
class Outer:
    inner: Annotated[Inner, FieldDescriptor.scalar(dtype=Inner)]
    extra: Annotated[jnp.ndarray, FieldDescriptor.tensor(dtype=jnp.int32, shape=())]

    @classmethod
    def default(cls, shape=()):
        return cls(inner=Inner.default(shape), extra=jnp.zeros(shape, dtype=jnp.int32))


def test_nested_reshape():
    # Batch shape: (2, 3)
    # Inner val intrinsic: ()
    # Extra intrinsic: ()
    outer = Outer(
        inner=Inner(val=jnp.arange(6).reshape(2, 3)), extra=jnp.arange(6).reshape(2, 3) + 10
    )

    # Verify initial shape
    assert outer.shape.batch == (2, 3)
    assert outer.inner.shape.batch == (2, 3)

    # Reshape to (6,)
    reshaped = outer.reshape(6)
    assert reshaped.shape.batch == (6,)
    assert reshaped.inner.shape.batch == (6,)
    assert jnp.array_equal(reshaped.inner.val, jnp.arange(6))

    # Reshape to (3, 2)
    reshaped_2 = outer.reshape(3, 2)
    assert reshaped_2.shape.batch == (3, 2)
    assert jnp.array_equal(reshaped_2.inner.val, jnp.arange(6).reshape(3, 2))


def test_nested_transpose():
    # Batch shape: (2, 3)
    val = jnp.arange(6).reshape(2, 3)
    outer = Outer(inner=Inner(val=val), extra=val + 10)

    transposed = outer.transpose()
    assert transposed.shape.batch == (3, 2)
    assert jnp.array_equal(transposed.inner.val, val.T)

    # Test with method factory generated method
    if hasattr(outer, "transpose"):
        method_transposed = outer.transpose()
        assert method_transposed.shape.batch == (3, 2)
        assert jnp.array_equal(method_transposed.inner.val, val.T)


def test_nested_swapaxes():
    # Batch shape: (2, 3, 4)
    shape = (2, 3, 4)
    size = 24
    val = jnp.arange(size).reshape(shape)
    outer = Outer.default(shape)
    # Manually set data to recognize patterns
    outer = Outer(inner=Inner(val=val), extra=val + 100)

    swapped = outer.swapaxes(0, 2)  # (4, 3, 2)
    assert swapped.shape.batch == (4, 3, 2)

    expected = jnp.swapaxes(val, 0, 2)
    assert jnp.array_equal(swapped.inner.val, expected)


def test_nested_flatten():
    # Batch shape: (2, 3)
    outer = Outer(
        inner=Inner(val=jnp.arange(6).reshape(2, 3)), extra=jnp.arange(6).reshape(2, 3) + 10
    )

    flattened = outer.flatten()
    assert flattened.shape.batch == (6,)
    assert jnp.array_equal(flattened.inner.val, jnp.arange(6))


def test_nested_with_different_intrinsic_shapes():
    @xtructure_dataclass
    class DeepInner:
        feat: Annotated[jnp.ndarray, FieldDescriptor.tensor(dtype=jnp.float32, shape=(10,))]

        @classmethod
        def default(cls, shape=()):
            return cls(feat=jnp.zeros(shape + (10,), dtype=jnp.float32))

    # Batch (2, 3)
    # DeepInner feat: (2, 3, 10)
    deep = DeepInner.default((2, 3))

    assert deep.shape.batch == (2, 3)

    # Reshape keys only affects batch dims
    reshaped = deep.reshape(6)
    assert reshaped.shape.batch == (6,)
    assert reshaped.feat.shape == (6, 10)  # intrinsic preserved

    # Transpose
    transposed = deep.transpose()
    assert transposed.shape.batch == (3, 2)
    assert transposed.feat.shape == (3, 2, 10)
