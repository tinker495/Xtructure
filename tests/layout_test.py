import pickle

import jax
import jax.numpy as jnp
import pytest

from xtructure import FieldDescriptor, StructuredType, xtructure_dataclass
from xtructure.core.layout import get_instance_layout, get_type_layout
from xtructure.core.layout.traversal import build_instance_from_leaf_values


@xtructure_dataclass(bitpack="off")
class LayoutPrimitive:
    id: FieldDescriptor.scalar(dtype=jnp.uint32)
    vector: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))


@xtructure_dataclass(bitpack="off")
class LayoutInner:
    val: FieldDescriptor.scalar(dtype=jnp.int32)


@xtructure_dataclass(bitpack="off")
class LayoutNestedScalar:
    inner: FieldDescriptor.scalar(dtype=LayoutInner)


@xtructure_dataclass
class LayoutEmpty:
    pass


@xtructure_dataclass
class LayoutNestedEmptyAuto:
    empty: FieldDescriptor.scalar(dtype=LayoutEmpty)


@xtructure_dataclass(bitpack="off")
class LayoutNestedEmptyAndData:
    empty: FieldDescriptor.scalar(dtype=LayoutEmpty)
    data: FieldDescriptor.scalar(dtype=LayoutInner)


@xtructure_dataclass(bitpack="off")
class LayoutNestedDataPair:
    left: FieldDescriptor.scalar(dtype=LayoutInner)
    right: FieldDescriptor.scalar(dtype=LayoutInner)


@xtructure_dataclass(validate=True)
class LayoutAggregateEligible:
    flags: FieldDescriptor.tensor(dtype=jnp.bool_, shape=(7,), bits=1, fill_value=False)
    codes: FieldDescriptor.tensor(dtype=jnp.uint16, shape=(3,), bits=12, fill_value=0)


@xtructure_dataclass(validate=True)
class LayoutPackedField:
    flags: FieldDescriptor.packed_tensor(
        unpacked_dtype=jnp.bool_,
        shape=(17,),
        packed_bits=1,
    )


@xtructure_dataclass(bitpack="off")
class LayoutDTypeKindFacts:
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

        @xtructure_dataclass(bitpack="off")
        class LayoutComplexUnsupported:
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
    outer = LayoutNestedDataPair(
        left=LayoutInner.default(shape=(2,)),
        right=LayoutInner.default(shape=(3,)),
    )

    layout = get_instance_layout(outer)

    assert layout.structured_type == StructuredType.UNSTRUCTURED
    assert layout.batch_shape == -1
    assert "field batch shapes disagree" in layout.mismatch_reason


def test_all_zero_leaf_composite_remains_single():
    outer = LayoutNestedEmptyAuto(empty=LayoutEmpty())

    layout = get_instance_layout(outer)

    assert layout.structured_type == StructuredType.SINGLE
    assert layout.batch_shape == ()
    assert layout.field_shape_for("empty").batch == ()


def test_layout_cache_populates_instance_and_survives_pytree_roundtrip():
    instance = LayoutPrimitive.default(shape=(2,))

    assert instance._layout_cache.shape_tuple == get_instance_layout(instance).shape_tuple
    assert instance.shape is instance._layout_cache.shape_tuple
    assert instance.dtype is instance._layout_cache.dtype_tuple
    assert instance.batch_shape == (2,)
    assert instance.structured_type == StructuredType.BATCHED
    assert len(instance) == 2
    assert instance.ndim == 1

    leaves, treedef = jax.tree_util.tree_flatten(instance)
    assert len(leaves) == 2
    assert all(leaf is not instance._layout_cache for leaf in leaves)

    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt._layout_cache.shape_tuple == instance._layout_cache.shape_tuple
    assert rebuilt._layout_cache.dtype_tuple == instance._layout_cache.dtype_tuple


def test_layout_cache_is_transient_across_pickle_roundtrip():
    instance = LayoutNestedScalar.default(shape=(2,))

    payload = pickle.dumps(instance)
    rebuilt = pickle.loads(payload)

    assert "_layout_cache" in rebuilt.__dict__
    assert rebuilt.shape == instance.shape
    assert rebuilt.dtype == instance.dtype
    assert rebuilt.batch_shape == instance.batch_shape


def test_instance_layout_accepts_python_primitive_values():
    instance = LayoutPrimitive(id=1, vector=[1.0, 2.0, 3.0])

    assert instance.shape.id == ()
    assert instance.shape.vector == (3,)
    assert instance.structured_type == StructuredType.SINGLE
    assert instance.dtype.id == jnp.asarray(1).dtype
    assert instance.dtype.vector == jnp.asarray([1.0, 2.0, 3.0]).dtype


def test_layout_cache_populates_nested_instances_before_outer_cache():
    outer = LayoutNestedScalar.default(shape=(3,))

    assert hasattr(outer.inner, "_layout_cache")
    assert outer.inner._layout_cache.shape_tuple == outer.inner.shape
    assert outer._layout_cache.shape_tuple == get_instance_layout(outer).shape_tuple
    assert outer.shape.inner == outer.inner._layout_cache.shape_tuple.__class__((), ())


def test_layout_cache_refreshes_on_replace():
    single = LayoutPrimitive.default()

    replaced = single.replace(
        id=jnp.arange(3, dtype=jnp.uint32),
        vector=jnp.ones((3, 3), dtype=jnp.float32),
    )

    assert single._layout_cache.batch_shape == ()
    assert replaced._layout_cache.batch_shape == (3,)
    assert replaced.shape.batch == (3,)


def test_layout_cache_replace_reuses_packed_storage_shape():
    state = LayoutPackedField.default(shape=(2,))

    replaced = state.replace(flags=jnp.zeros_like(state.flags))

    assert replaced._layout_cache is state._layout_cache
    assert replaced.shape.flags == (3,)
    assert replaced.shape.batch == (2,)
