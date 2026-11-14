import jax
import jax.numpy as jnp
import pytest

from tests.dataclass.fixtures import SimpleData, VectorData
from xtructure import FieldDescriptor, Xtructurable


class ValidatedScalarData(Xtructurable):
    value: FieldDescriptor.scalar(dtype=jnp.float32)


class ValidatedVectorData(Xtructurable):
    vector: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))


class ValidatedNestedData(Xtructurable):
    simple: FieldDescriptor.scalar(dtype=SimpleData)


class ValidatedWithPostInit(Xtructurable):
    value: FieldDescriptor.scalar(dtype=jnp.float32)

    def __post_init__(self):
        self.value = self.value + 1.0


def test_validate_dtype_mismatch():
    ValidatedScalarData(value=jnp.array(1.0, dtype=jnp.float32))
    with pytest.raises(TypeError):
        ValidatedScalarData(value=jnp.array(1.0, dtype=jnp.int32))


def test_validate_shape_mismatch():
    ValidatedVectorData(vector=jnp.ones((2, 3), dtype=jnp.float32))
    with pytest.raises(ValueError):
        ValidatedVectorData(vector=jnp.ones((2,), dtype=jnp.float32))


def test_validate_nested_type():
    ValidatedNestedData(simple=SimpleData.default())
    with pytest.raises(TypeError):
        ValidatedNestedData(simple=VectorData.default())


def test_validate_preserves_existing_post_init():
    data = ValidatedWithPostInit(value=jnp.array(1.0, dtype=jnp.float32))
    assert jnp.array_equal(data.value, jnp.array(2.0, dtype=jnp.float32))


def test_replace_validates_fields_changed_indirectly_by_post_init():
    # Author: tinker495. A hook can change a field absent from replace kwargs;
    # comparing only explicitly replaced shapes would reuse an invalid layout.
    class Hooked(Xtructurable, bitpack="off"):
        x: FieldDescriptor.scalar(dtype=jnp.float32)
        y: FieldDescriptor.scalar(dtype=jnp.float32)

        def __post_init__(self):
            if jnp.all(self.x == 9):
                self.y = jnp.zeros((4,), dtype=jnp.float32)

    data = Hooked(x=jnp.zeros((3,), dtype=jnp.float32), y=jnp.zeros((3,), dtype=jnp.float32))
    with pytest.raises(ValueError, match="field batch shapes disagree"):
        data.replace(x=jnp.full((3,), 9, dtype=jnp.float32))
    with pytest.raises(ValueError, match="field batch shapes disagree"):
        Hooked(x=jnp.full((3,), 9, dtype=jnp.float32), y=jnp.zeros((3,), dtype=jnp.float32))
    assert data.shape == (3,)


def test_validation_cache_bypasses_user_setattr():
    # Author: tinker495. Internal cache writes must not invoke a user setter that
    # forbids internal mutation after post-init, including during replace.
    class Restricted(Xtructurable, bitpack="off"):
        value: FieldDescriptor.scalar(dtype=jnp.float32)

        def __setattr__(self, name, value):
            if name == "_layout_cache":
                raise AttributeError("internal writes are forbidden")
            object.__setattr__(self, name, value)

    data = Restricted(value=jnp.array(1, dtype=jnp.float32))
    assert data.replace(value=jnp.array(2, dtype=jnp.float32)).shape == ()


@pytest.mark.parametrize(
    "operation, expected_value, expected_doubled",
    [("identity", 3, 6), ("roundtrip", 3, 6), ("increment", 4, 7)],
)
def test_pytree_restores_dynamic_attributes_without_post_init(
    operation, expected_value, expected_doubled
):
    # Author: tinker495. Hook-created attributes are independent PyTree leaves;
    # restoration must keep their supplied values without rerunning the hook or
    # restoring a layout cache computed for the original leaves.
    hook_calls = []

    class Derived(Xtructurable):
        value: FieldDescriptor.scalar(dtype=jnp.int32)

        def __post_init__(self):
            hook_calls.append(1)
            self.doubled = self.value * 2

    data = Derived(value=jnp.array(3, dtype=jnp.int32))
    assert data._layout_cache is not None
    if operation == "identity":
        restored = jax.tree_util.tree_map(lambda leaf: leaf, data)
    elif operation == "roundtrip":
        leaves, treedef = jax.tree_util.tree_flatten(data)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
    else:
        restored = jax.tree_util.tree_map(lambda leaf: leaf + 1, data)

    assert int(restored.value) == expected_value
    assert int(restored.doubled) == expected_doubled
    assert hook_calls == [1]
    assert restored._layout_cache is None
