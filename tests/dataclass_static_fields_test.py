import jax
import jax.numpy as jnp

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
