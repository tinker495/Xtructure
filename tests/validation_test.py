import jax
import jax.numpy as jnp
import pytest

from xtructure import FieldDescriptor, Xtructurable
from xtructure import numpy as xnp


def test_check_invariants_manual_call():
    class MyData(Xtructurable):
        x: FieldDescriptor.scalar(dtype=jnp.int32)

    # Correct type
    obj = MyData(x=jnp.array(10, dtype=jnp.int32))
    obj.check_invariants()  # Should not raise

    # Incorrect type (float instead of int)
    # Author: tinker495. Structural maps may change dtype until data validation.
    obj_bad = jax.tree_util.tree_map(lambda _: jnp.array(10.5, dtype=jnp.float32), obj)
    with pytest.raises(TypeError, match="expected dtype"):
        obj_bad.check_invariants()


def test_check_invariants_auto_validation_enabled():
    class MyDataAuto(Xtructurable):
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

    class ValidatedData(Xtructurable):
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

    class ValidatedDataManual(Xtructurable):
        val: FieldDescriptor.scalar(dtype=jnp.int32, validator=non_negative)

    # Author: tinker495. Raw PyTree reconstruction defers the custom data check.
    obj = jax.tree_util.tree_map(
        lambda _: jnp.array(-1, dtype=jnp.int32),
        ValidatedDataManual(val=jnp.array(5, dtype=jnp.int32)),
    )
    with pytest.raises(ValueError, match="Must be non-negative"):
        obj.check_invariants()


def test_data_operations_validate_custom_constraints_before_returning():
    # Author: tinker495. A valid batch can yield an invalid slice or update; every
    # data result must satisfy the same validator as ordinary construction (#53).
    def nonnegative_total(value):
        if jnp.sum(value) < 0:
            raise ValueError("negative total")

    class Values(Xtructurable, bitpack="off"):
        value: FieldDescriptor.scalar(dtype=jnp.int32, validator=nonnegative_total)

    data = Values(value=jnp.array([-1, 2], dtype=jnp.int32))
    with pytest.raises(ValueError, match="negative total"):
        data[0]
    with pytest.raises(ValueError, match="negative total"):
        data.replace(value=jnp.array([-3, 2], dtype=jnp.int32))
    with pytest.raises(ValueError, match="negative total"):
        data.at[1].set(-2)
    with pytest.raises(ValueError, match="negative total"):
        xnp.full_like(data, -1)
    assert int(data[1].value) == 2


def test_jit_incompatible_validator_is_not_silently_skipped():
    # Author: tinker495. Python value-dependent validation cannot run during JIT
    # tracing; surfacing that limitation is part of the always-validate contract.
    def nonnegative(value):
        if jnp.any(value < 0):
            raise ValueError("negative value")

    class Values(Xtructurable, bitpack="off"):
        value: FieldDescriptor.scalar(dtype=jnp.float32, validator=nonnegative)

    Values(value=jnp.array(1, dtype=jnp.float32))
    with pytest.raises(jax.errors.TracerBoolConversionError):
        jax.jit(lambda value: Values(value=value))(jnp.array(1, dtype=jnp.float32))
