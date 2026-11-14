import pickle

import jax
import jax.numpy as jnp
import pytest

from xtructure import FieldDescriptor, StructuredType, Xtructurable
from xtructure import numpy as xnp
from xtructure.core.layout import get_instance_layout, get_type_layout
from xtructure.core.layout.traversal import build_instance_from_leaf_values


class LayoutPrimitive(Xtructurable, bitpack="off"):
    id: FieldDescriptor.scalar(dtype=jnp.uint32)
    vector: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))


class LayoutInner(Xtructurable, bitpack="off"):
    val: FieldDescriptor.scalar(dtype=jnp.int32)


class LayoutNestedScalar(Xtructurable, bitpack="off"):
    inner: FieldDescriptor.scalar(dtype=LayoutInner)


class LayoutEmpty(Xtructurable):
    pass


class LayoutNestedEmptyAuto(Xtructurable):
    empty: FieldDescriptor.scalar(dtype=LayoutEmpty)


class LayoutNestedEmptyAndData(Xtructurable, bitpack="off"):
    empty: FieldDescriptor.scalar(dtype=LayoutEmpty)
    data: FieldDescriptor.scalar(dtype=LayoutInner)


class LayoutNestedDataPair(Xtructurable, bitpack="off"):
    left: FieldDescriptor.scalar(dtype=LayoutInner)
    right: FieldDescriptor.scalar(dtype=LayoutInner)


class LayoutAggregateEligible(Xtructurable):
    flags: FieldDescriptor.tensor(dtype=jnp.bool_, shape=(7,), bits=1, fill_value=False)
    codes: FieldDescriptor.tensor(dtype=jnp.uint16, shape=(3,), bits=12, fill_value=0)


class LayoutPackedField(Xtructurable):
    flags: FieldDescriptor.packed_tensor(
        unpacked_dtype=jnp.bool_,
        shape=(17,),
        packed_bits=1,
    )


class LayoutDTypeKindFacts(Xtructurable, bitpack="off"):
    flag: FieldDescriptor.scalar(dtype=jnp.bool_)
    code: FieldDescriptor.scalar(dtype=jnp.uint16)
    offset: FieldDescriptor.scalar(dtype=jnp.int16)
    weight: FieldDescriptor.scalar(dtype=jnp.float32)


def test_type_layout_adapter_field_plans_use_dtype_kind_facts():
    layout = get_type_layout(LayoutDTypeKindFacts)
    plans = {name: plan for name, plan in layout.adapter_field_plan_by_name}

    assert plans["flag"].random_kind == "bool"
    assert plans["code"].random_kind == "bits_int"
    assert plans["code"].random_bits_dtype == jnp.dtype(jnp.uint16)
    assert not plans["code"].random_view_as_signed
    assert plans["offset"].random_kind == "bits_int"
    assert plans["offset"].random_bits_dtype == jnp.dtype(jnp.uint16)
    assert plans["offset"].random_view_as_signed
    assert plans["weight"].random_kind == "float"

    assert layout.field_for("flag").fill_value is False
    assert layout.field_for("code").fill_value == jnp.iinfo(jnp.uint16).max
    assert layout.field_for("offset").fill_value == 0
    assert layout.field_for("weight").fill_value == jnp.inf


def test_type_layout_rejects_unsupported_dtype_kind_at_definition_time():
    with pytest.raises(TypeError, match="DType Kind"):

        class LayoutComplexUnsupported(Xtructurable, bitpack="off"):
            value: FieldDescriptor.scalar(dtype=jnp.complex64)


def test_build_instance_from_leaf_values_cast_declared_fails_with_context():
    with pytest.raises(TypeError, match="Failed to cast leaf 'id'.*LayoutPrimitive"):
        build_instance_from_leaf_values(
            LayoutPrimitive,
            {
                ("id",): object(),
                ("vector",): jnp.ones((3,), dtype=jnp.float32),
            },
            cast_declared=True,
        )


def test_type_layout_lookup_facts_are_tuple_backed_and_hashable():
    layout = get_type_layout(LayoutAggregateEligible)

    assert isinstance(layout.field_by_name, tuple)
    assert isinstance(layout.adapter_field_plan_by_name, tuple)
    assert isinstance(layout.leaf_by_path, tuple)
    assert isinstance(layout.packed_field_layout_by_name, tuple)
    assert isinstance(layout.aggregate_bitpack.view_fields_by_owner, tuple)

    assert layout.field_for("flags") is layout.fields[0]
    assert layout.adapter_field_plan_for("flags").name == "flags"
    assert layout.leaf_for(("flags",)).path == ("flags",)
    assert layout.aggregate_bitpack.view_fields_for(LayoutAggregateEligible)[0].name == "flags"

    with pytest.raises(KeyError):
        layout.field_for("other")
    with pytest.raises(KeyError):
        layout.adapter_field_plan_for("other")

    hash(layout)


def test_scalar_nested_zero_leaf_field_is_batch_neutral():
    outer = LayoutNestedEmptyAndData(
        empty=LayoutEmpty(),
        data=LayoutInner(val=jnp.arange(6, dtype=jnp.int32).reshape(2, 3)),
    )

    layout = get_instance_layout(outer)

    assert layout.structured_type == StructuredType.BATCHED
    assert layout.batch_shape == (2, 3)
    assert layout.field_shape_for("empty").batch == ()
    assert layout.fields[0].batch_shape == ()
    assert layout.fields[0].mismatch_reason is None


def test_data_bearing_nested_fields_still_require_one_batch_shape():
    # Author: tinker495. Individually valid nested fields must share one outer batch.
    with pytest.raises(ValueError, match="field batch shapes disagree"):
        LayoutNestedDataPair(
            left=LayoutInner.default(shape=(2,)),
            right=LayoutInner.default(shape=(3,)),
        )


def test_all_zero_leaf_composite_remains_single():
    outer = LayoutNestedEmptyAuto(empty=LayoutEmpty())

    layout = get_instance_layout(outer)

    assert layout.structured_type == StructuredType.SINGLE
    assert layout.batch_shape == ()
    assert layout.field_shape_for("empty").batch == ()


def test_layout_cache_populates_instance_and_survives_pytree_roundtrip():
    instance = LayoutPrimitive.default(shape=(2,))

    assert instance._layout_cache.shape_tuple == get_instance_layout(instance).shape_tuple
    assert instance.layout is instance._layout_cache
    assert instance.dtype is instance._layout_cache.dtype_tuple
    assert instance.shape == (2,)
    assert instance.structured_type == StructuredType.BATCHED
    assert len(instance) == 2
    assert instance.ndim == 1

    leaves, treedef = jax.tree_util.tree_flatten(instance)
    assert len(leaves) == 2
    assert all(leaf is not instance._layout_cache for leaf in leaves)

    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    rebuilt.layout
    assert rebuilt._layout_cache.shape_tuple == instance._layout_cache.shape_tuple
    assert rebuilt._layout_cache.dtype_tuple == instance._layout_cache.dtype_tuple


def test_layout_cache_is_transient_across_pickle_roundtrip():
    instance = LayoutNestedScalar.default(shape=(2,))

    payload = pickle.dumps(instance)
    rebuilt = pickle.loads(payload)

    assert "_layout_cache" in rebuilt.__dict__
    assert rebuilt.shape == instance.shape
    assert rebuilt.dtype == instance.dtype
    assert rebuilt.shape == instance.shape


def test_instance_layout_accepts_python_primitive_values():
    # Author: tinker495. Primitive inputs are checked without implicit dtype casts.
    with pytest.raises(TypeError, match="expected dtype"):
        LayoutPrimitive(id=1, vector=[1.0, 2.0, 3.0])
    instance = LayoutPrimitive(id=jnp.uint32(1), vector=[1.0, 2.0, 3.0])

    assert instance.layout.field_shape_for("id") == ()
    assert instance.layout.field_shape_for("vector") == (3,)
    assert instance.structured_type == StructuredType.SINGLE
    assert instance.dtype.id == jnp.dtype(jnp.uint32)
    assert instance.dtype.vector == jnp.asarray([1.0, 2.0, 3.0]).dtype


def test_layout_cache_populates_nested_instances_before_outer_cache():
    outer = LayoutNestedScalar.default(shape=(3,))

    assert hasattr(outer.inner, "_layout_cache")
    assert outer.inner.layout is outer.inner._layout_cache
    assert outer._layout_cache.shape_tuple == get_instance_layout(outer).shape_tuple
    assert outer.layout.field_shape_for("inner") == outer.inner._layout_cache.shape_tuple.__class__(
        (), ()
    )


def test_layout_cache_refreshes_on_replace():
    single = LayoutPrimitive.default()

    replaced = single.replace(
        id=jnp.arange(3, dtype=jnp.uint32),
        vector=jnp.ones((3, 3), dtype=jnp.float32),
    )

    assert single._layout_cache.batch_shape == ()
    assert replaced._layout_cache.batch_shape == (3,)
    assert replaced.shape == (3,)


def test_layout_cache_replace_reuses_packed_storage_shape():
    state = LayoutPackedField.default(shape=(2,))

    replaced = state.replace(flags=jnp.zeros_like(state.flags))

    assert replaced._layout_cache is state._layout_cache
    assert replaced.layout.field_shape_for("flags") == (3,)
    assert replaced.shape == (2,)


def test_pytree_reconstruction_preserves_values_without_replaying_post_init():
    # Author: tinker495. Structural JAX reconstruction must not repeat data
    # normalization, even when a tree_map changes values or supplies metadata.
    class Hooked(Xtructurable, bitpack="off"):
        value: FieldDescriptor.scalar(dtype=jnp.float32)

        def __post_init__(self):
            self.value = self.value + 1

    data = Hooked(value=jnp.array(1, dtype=jnp.float32))
    leaves, structure = jax.tree_util.tree_flatten(data)
    rebuilt = jax.tree_util.tree_unflatten(structure, leaves)
    assert float(rebuilt.value) == 2
    assert rebuilt.shape == ()
    mapped = jax.tree_util.tree_map(lambda value: value + 2, data)
    assert float(mapped.value) == 4
    assert mapped.shape == ()
    metadata = jax.tree_util.tree_map(lambda _: object(), data)
    assert type(metadata.value) is object
    with pytest.raises(TypeError):
        _ = metadata.shape


@pytest.mark.parametrize(
    "operation",
    [
        "shape",
        "dtype",
        "layout",
        "structured_type",
        "len",
        "str",
        "bytes",
        "hash",
        "equality",
        "index",
        "replace",
        "save",
        "concat",
        "take",
        "tile",
        "unique",
        "at",
    ],
)
def test_data_api_rejects_invalid_structurally_reconstructed_dtype(operation, tmp_path):
    # Author: tinker495. Raw tree_map is structural, but no data API may consume
    # the resulting wrong-dtype object, even if an operation could repair it.
    valid = LayoutInner(val=jnp.array([1, 2], dtype=jnp.int32))
    invalid = jax.tree_util.tree_map(lambda value: value.astype(jnp.float32), valid)
    operations = {
        "shape": lambda: invalid.shape,
        "dtype": lambda: invalid.dtype,
        "layout": lambda: invalid.layout,
        "structured_type": lambda: invalid.structured_type,
        "len": lambda: len(invalid),
        "str": lambda: str(invalid),
        "bytes": lambda: invalid.bytes,
        "hash": lambda: invalid.hash(),
        "equality": lambda: invalid == valid,
        "index": lambda: invalid[0],
        "replace": lambda: invalid.replace(val=valid.val),
        "save": lambda: invalid.save(str(tmp_path / "invalid.npz")),
        "concat": lambda: xnp.concat([invalid]),
        "take": lambda: xnp.take(invalid, jnp.array([0]), axis=0),
        "tile": lambda: xnp.tile(invalid, 1),
        "unique": lambda: xnp.unique_mask(invalid, batch_len=0),
        "at": lambda: invalid.at[0].set(1),
    }
    with pytest.raises(TypeError, match="expected dtype"):
        operations[operation]()


def test_jax_transformations_keep_structural_metadata_separate_from_data():
    # Author: tinker495. Mixed-type data must retain normal JIT/vmap behavior;
    # eval_shape and axis specifications reconstruct metadata without validation.
    data = LayoutPrimitive.default(shape=(2,))
    compiled = jax.jit(lambda value: value.reshape((1, 2)))(data)
    assert compiled.shape == (1, 2)
    assert compiled.id.dtype == jnp.uint32
    axes = jax.tree_util.tree_map(lambda _: 0, data)
    mapped = jax.vmap(lambda value: value, in_axes=(axes,))(data)
    assert mapped.shape == (2,)
    metadata = jax.eval_shape(lambda value: value, data)
    assert isinstance(metadata.vector, jax.ShapeDtypeStruct)
    assert metadata.vector.shape == (2, 3)


def test_float_gradient_reconstructs_valid_data():
    # Author: tinker495. Differentiation uses structural reconstruction too;
    # float-only gradients retain the declared shape and dtype when consumed.
    class Vector(Xtructurable, bitpack="off"):
        value: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))

    gradient = jax.grad(lambda vector: jnp.sum(vector.value**2))(
        Vector(value=jnp.array([1, 2, 3], dtype=jnp.float32))
    )
    assert gradient.shape == ()
    assert jnp.array_equal(gradient.value, jnp.array([2, 4, 6], dtype=jnp.float32))


def test_pickle_restore_rejects_invalid_stored_fields():
    # Author: tinker495. Pickle restoration is a data boundary, unlike JAX
    # reconstruction. A corrupted stored field must fail before returning data.
    data = LayoutInner(val=jnp.array(1, dtype=jnp.int32))
    object.__setattr__(data, "val", jnp.array(1, dtype=jnp.float32))
    with pytest.raises(TypeError, match="expected dtype"):
        pickle.loads(pickle.dumps(data))
