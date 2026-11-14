"""Tests for xnp.pad behavior across padding scenarios."""

import jax.numpy as jnp
import pytest

from tests.xnp.shared_data import SimpleData, VectorData
from xtructure import FieldDescriptor, Xtructurable
from xtructure import numpy as xnp


def test_pad_single_to_batched():
    """Test padding a SINGLE dataclass to create a batched version."""
    data = SimpleData.default()
    data = data.replace(id=jnp.array(42, dtype=jnp.uint32), value=jnp.array(3.14))

    result = xnp.pad(data, (0, 4))

    assert result.structured_type.name == "BATCHED"
    assert result.shape == (5,)
    expected_id = jnp.array([42, 4294967295, 4294967295, 4294967295, 4294967295], dtype=jnp.uint32)
    expected_value = jnp.array([3.14, jnp.inf, jnp.inf, jnp.inf, jnp.inf], dtype=jnp.float32)
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.allclose(result.value, expected_value)


def test_pad_batched_axis_0():
    """Test padding a BATCHED dataclass along axis 0."""
    data = SimpleData.default((3,))
    data = data.replace(id=jnp.array([1, 2, 3], dtype=jnp.uint32), value=jnp.array([1.0, 2.0, 3.0]))

    result = xnp.pad(data, (0, 2))

    assert result.structured_type.name == "BATCHED"
    assert result.shape == (5,)
    expected_id = jnp.array([1, 2, 3, 4294967295, 4294967295], dtype=jnp.uint32)
    expected_value = jnp.array([1.0, 2.0, 3.0, jnp.inf, jnp.inf], dtype=jnp.float32)
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.array_equal(result.value, expected_value)


def test_pad_single_before_after_inserts_correctly():
    """SINGLE padding with (before, after) inserts at the correct offset."""
    data = SimpleData.default()
    data = data.replace(
        id=jnp.array(42, dtype=jnp.uint32), value=jnp.array(3.14, dtype=jnp.float32)
    )

    result = xnp.pad(data, (2, 1))

    assert result.structured_type.name == "BATCHED"
    assert result.shape == (4,)
    expected_id = jnp.array([4294967295, 4294967295, 42, 4294967295], dtype=jnp.uint32)
    expected_value = jnp.array([jnp.inf, jnp.inf, 3.14, jnp.inf], dtype=jnp.float32)
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.allclose(result.value, expected_value)


def test_pad_batched_before_after_inserts_correctly():
    """BATCHED padding with (before, after) inserts at the correct offset."""
    data = SimpleData.default((3,))
    data = data.replace(id=jnp.array([1, 2, 3], dtype=jnp.uint32), value=jnp.array([1.0, 2.0, 3.0]))

    result = xnp.pad(data, (2, 1))

    assert result.structured_type.name == "BATCHED"
    assert result.shape == (6,)
    expected_id = jnp.array([4294967295, 4294967295, 1, 2, 3, 4294967295], dtype=jnp.uint32)
    expected_value = jnp.array([jnp.inf, jnp.inf, 1.0, 2.0, 3.0, jnp.inf], dtype=jnp.float32)
    assert jnp.array_equal(result.id, expected_id)
    assert jnp.allclose(result.value, expected_value)


def test_pad_uses_existing_padding_as_batch():
    """Test that pad function uses the existing padding_as_batch method when appropriate."""
    data = SimpleData.default((2,))
    data = data.replace(id=jnp.array([1, 2], dtype=jnp.uint32), value=jnp.array([1.0, 2.0]))

    result_xnp = xnp.pad(data, (0, 2))
    result_builtin = data.padding_as_batch((4,))

    assert jnp.array_equal(result_xnp.id, result_builtin.id)
    assert jnp.array_equal(result_xnp.value, result_builtin.value)
    assert result_xnp.shape == result_builtin.shape


def test_pad_batched_with_constant_values():
    """Test padding with custom constant values."""
    data = SimpleData.default((2,))
    data = data.replace(id=jnp.array([1, 2], dtype=jnp.uint32), value=jnp.array([1.0, 2.0]))

    result = xnp.pad(data, (0, 2), constant_values=99)

    assert result.shape == (4,)
    assert jnp.array_equal(result.id, jnp.array([1, 2, 99, 99], dtype=jnp.uint32))
    assert jnp.array_equal(result.value, jnp.array([1.0, 2.0, 99.0, 99.0], dtype=jnp.float32))


def test_pad_batched_target_shape():
    """Test padding to a target batch shape."""
    data = SimpleData.default((2, 3))

    result = xnp.pad(data, [(0, 2), (0, 2)])

    assert result.shape == (4, 5)


def test_pad_tensor_fields_multidim_batch_default_constant():
    """Default constant padding should work for tensor fields with batch_ndim > 1."""
    data = VectorData.default((2, 3))
    position = jnp.arange(18, dtype=jnp.float32).reshape(2, 3, 3)
    velocity = position + 0.5
    data = data.replace(position=position, velocity=velocity)

    result = xnp.pad(data, [(1, 0), (0, 2)])

    assert result.shape == (3, 5)
    assert jnp.array_equal(result.position[1:3, 0:3], position)
    assert jnp.array_equal(result.velocity[1:3, 0:3], velocity)
    assert jnp.all(jnp.isinf(result.position[0]))
    assert jnp.all(jnp.isinf(result.velocity[0]))
    assert jnp.all(jnp.isinf(result.position[:, 3:]))
    assert jnp.all(jnp.isinf(result.velocity[:, 3:]))


def test_pad_no_change_needed():
    """Test that padding with zero padding returns the same instance."""
    data = SimpleData.default((3,))
    result = xnp.pad(data, (0, 0))
    assert result is data


def test_pad_invalid_pad_width():
    """Test that invalid pad_width raises ValueError."""
    data = SimpleData.default((3,))

    with pytest.raises(
        ValueError, match="pad_width must be int, sequence of int, or sequence of pairs"
    ):
        xnp.pad(data, "invalid")


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_default_pad_validates_completed_arrays(batched, nested):
    # Author: tinker495. A zero-filled temporary object violates this validator,
    # while the completed padding preserves the positive sum. Both batch modes
    # and nested fields must reach validation only after inserting the input.
    def positive_sum(value):
        if jnp.sum(value) <= 0:
            raise ValueError("sum must be positive")

    class PositiveSum(Xtructurable):
        value: FieldDescriptor.scalar(dtype=jnp.int32, default=0, validator=positive_sum)

    class Nested(Xtructurable):
        child: FieldDescriptor.scalar(dtype=PositiveSum)

    data = PositiveSum(value=jnp.array([1] if batched else 1, dtype=jnp.int32))
    result = xnp.pad(Nested(child=data) if nested else data, 1)

    assert result.shape == (3,)
    assert jnp.array_equal(
        result.child.value if nested else result.value,
        jnp.array([0, 1, 0], dtype=jnp.int32),
    )


@pytest.mark.parametrize("batched", [False, True])
def test_default_pad_rejects_invalid_completed_arrays(batched):
    # Author: tinker495. Avoiding validation of temporary fill arrays must not
    # allow a completed padded result that violates the declared validator.
    def all_positive(value):
        if jnp.any(value <= 0):
            raise ValueError("all values must be positive")

    class AllPositive(Xtructurable):
        value: FieldDescriptor.scalar(dtype=jnp.int32, default=0, validator=all_positive)

    data = AllPositive(value=jnp.array([1] if batched else 1, dtype=jnp.int32))
    with pytest.raises(ValueError, match="all values must be positive"):
        xnp.pad(data, 1)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_default_pad_preserves_initialized_arrays(batched, nested):
    # Author: tinker495. Padding must preserve initialized values and declared
    # fill values without rerunning either the leaf or parent initialization.
    hook_calls = []

    class Incremented(Xtructurable):
        value: FieldDescriptor.scalar(dtype=jnp.int32, default=0)

        def __post_init__(self):
            hook_calls.append("leaf")
            self.value = self.value + 1

    class Nested(Xtructurable):
        child: FieldDescriptor.scalar(dtype=Incremented)

        def __post_init__(self):
            hook_calls.append("parent")

    child = Incremented(value=jnp.array([1] if batched else 1, dtype=jnp.int32))
    data = Nested(child=child) if nested else child
    result = xnp.pad(data, 1)

    assert result.shape == (3,)
    assert jnp.array_equal(
        result.child.value if nested else result.value,
        jnp.array([0, 2, 0], dtype=jnp.int32),
    )
    assert hook_calls == (["leaf", "parent"] if nested else ["leaf"])
    assert jnp.array_equal(child.value, jnp.array([2] if batched else 2, dtype=jnp.int32))


@pytest.mark.parametrize("batched", [False, True])
def test_default_pad_validates_without_initialization_rewriting_result(batched):
    # Author: tinker495. A hook must not turn invalid padding zeros into positive
    # values before validation; validate the completed arrays as produced.
    hook_calls = []

    def positive(value):
        if jnp.any(value <= 0):
            raise ValueError("values must stay positive")

    class Incremented(Xtructurable):
        value: FieldDescriptor.scalar(dtype=jnp.int32, default=0, validator=positive)

        def __post_init__(self):
            hook_calls.append("leaf")
            self.value = self.value + 1

    data = Incremented(value=jnp.array([1] if batched else 1, dtype=jnp.int32))
    with pytest.raises(ValueError, match="values must stay positive"):
        xnp.pad(data, 1)
    assert hook_calls == ["leaf"]
