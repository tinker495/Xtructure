import jax
import jax.numpy as jnp
import pytest

from xtructure.core.dataclass import base_dataclass


@base_dataclass(static_fields=("name",))
class WithStaticFields:
    x: jax.Array
    name: str


def test_dataclass_static_fields_roundtrip_and_jit():
    obj = WithStaticFields(x=jnp.arange(3, dtype=jnp.int32), name="abc")

    leaves, treedef = jax.tree_util.tree_flatten(obj)
    assert len(leaves) == 1
    assert (leaves[0] == obj.x).all()

    obj2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert obj2.name == "abc"
    assert (obj2.x == obj.x).all()

    @jax.jit
    def f(o: WithStaticFields):
        return o.replace(x=o.x + 1)

    out = f(obj)
    assert out.name == "abc"
    assert (out.x == obj.x + 1).all()


@pytest.mark.parametrize(
    "operation, expected_value, expected_doubled",
    [("identity", 3, 6), ("roundtrip", 3, 6), ("increment", 4, 7)],
)
def test_static_pytree_restores_dynamic_attributes(operation, expected_value, expected_doubled):
    # Author: tinker495. The static-metadata restoration path must retain both
    # dynamic array attributes and dynamic static metadata without retaining
    # transient caches, even when no post-init hook recreates the attributes.
    @base_dataclass(static_fields=("label",))
    class DynamicMetadata:
        value: jax.Array

    data = DynamicMetadata(value=jnp.array(3, dtype=jnp.int32))
    data.label = "stable"
    data.doubled = data.value * 2
    data._layout_cache = object()
    if operation == "identity":
        restored = jax.tree_util.tree_map(lambda leaf: leaf, data)
    elif operation == "roundtrip":
        leaves, treedef = jax.tree_util.tree_flatten(data)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
    else:
        restored = jax.tree_util.tree_map(lambda leaf: leaf + 1, data)

    assert int(restored.value) == expected_value
    assert int(restored.doubled) == expected_doubled
    assert restored.label == "stable"
    assert "_layout_cache" not in vars(restored)
