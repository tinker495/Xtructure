import jax.numpy as jnp
import pytest

from xtructure import FieldDescriptor, xtructure_dataclass


def test_check_invariants_manual_call():
    @xtructure_dataclass
    class MyData:
        x: FieldDescriptor.scalar(dtype=jnp.int32)

    # Correct type
    obj = MyData(x=jnp.array(10, dtype=jnp.int32))
    obj.check_invariants()  # Should not raise

    # Incorrect type (float instead of int)
    obj_bad = MyData(x=jnp.array(10.5, dtype=jnp.float32))
    with pytest.raises(TypeError, match="expected dtype"):
        obj_bad.check_invariants()


def test_check_invariants_auto_validation_enabled():
    @xtructure_dataclass(validate=True)
    class MyDataAuto:
        x: FieldDescriptor.tensor(dtype=jnp.float32, shape=(2,))

    # Correct shape/dtype
    MyDataAuto(x=jnp.array([1.0, 2.0], dtype=jnp.float32))

    # Incorrect shape
    with pytest.raises(ValueError, match="expected trailing shape"):
        MyDataAuto(x=jnp.array([1.0], dtype=jnp.float32))


def test_check_invariants_custom_validator():
    def non_negative(x):
        if jnp.any(x < 0):
            raise ValueError("Must be non-negative")

    @xtructure_dataclass(validate=True)
    class ValidatedData:
        val: FieldDescriptor.scalar(dtype=jnp.int32, validator=non_negative)

    # Valid
    ValidatedData(val=jnp.array(5, dtype=jnp.int32))

    # Invalid via custom validator
    with pytest.raises(ValueError, match="Must be non-negative"):
        ValidatedData(val=jnp.array(-1, dtype=jnp.int32))


def test_check_invariants_custom_validator_manual():
    def non_negative(x):
        if jnp.any(x < 0):
            raise ValueError("Must be non-negative")

    @xtructure_dataclass(validate=False)
    class ValidatedDataManual:
        val: FieldDescriptor.scalar(dtype=jnp.int32, validator=non_negative)

    # validate=False, so init should pass even with invalid data
    obj = ValidatedDataManual(val=jnp.array(-1, dtype=jnp.int32))

    # Manual check should raise
    with pytest.raises(ValueError, match="Must be non-negative"):
        obj.check_invariants()
