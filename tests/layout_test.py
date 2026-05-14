import pickle

import jax
import jax.numpy as jnp
import pytest

from xtructure import FieldDescriptor, StructuredType, xtructure_dataclass
from xtructure.core.layout import get_instance_layout, get_type_layout
from xtructure.core.layout.traversal import (
    build_instance_from_leaf_values,
    get_path_value,
    iter_leaf_values,
)
from xtructure.core.layout.types import AggregateBitpackReason


@xtructure_dataclass(bitpack="off")
class LayoutPrimitive:
    id: FieldDescriptor.scalar(dtype=jnp.uint32)
    vector: FieldDescriptor.tensor(dtype=jnp.float32, shape=(3,))


@xtructure_dataclass(bitpack="off")
class LayoutInner:
    val: FieldDescriptor.scalar(dtype=jnp.int32)


@xtructure_dataclass(bitpack="off")
class LayoutInnerEquivalent:
    val: FieldDescriptor.scalar(dtype=jnp.int32)


@xtructure_dataclass(bitpack="off")
class LayoutNestedScalar:
    inner: FieldDescriptor.scalar(dtype=LayoutInner)


@xtructure_dataclass(bitpack="off")
class LayoutInnerView:
    val: FieldDescriptor.scalar(dtype=jnp.int32)


@xtructure_dataclass(bitpack="off")
class LayoutNestedScalarView:
    inner: FieldDescriptor.scalar(dtype=LayoutInnerView)


@xtructure_dataclass(bitpack="off")
class LayoutNestedTensor:
    inner_array: FieldDescriptor.tensor(dtype=LayoutInner, shape=(2,))


@xtructure_dataclass(bitpack="off")
class LayoutInnerPair:
    a: FieldDescriptor.scalar(dtype=jnp.int32)
    b: FieldDescriptor.tensor(dtype=jnp.int32, shape=(2,))


@xtructure_dataclass(bitpack="off")
class LayoutNestedUnstructuredTensor:
    inner_array: FieldDescriptor.tensor(dtype=LayoutInnerPair, shape=(2,))


@xtructure_dataclass
class LayoutEmpty:
    pass


@xtructure_dataclass
class LayoutNestedEmptyAuto:
    empty: FieldDescriptor.scalar(dtype=LayoutEmpty)


@xtructure_dataclass(bitpack="off")
class LayoutIntIntrinsicShape:
    value: FieldDescriptor(dtype=jnp.int32, intrinsic_shape=3)


@xtructure_dataclass(validate=True)
class LayoutAggregateEligible:
    flags: FieldDescriptor.tensor(dtype=jnp.bool_, shape=(7,), bits=1, fill_value=False)
    codes: FieldDescriptor.tensor(dtype=jnp.uint16, shape=(3,), bits=12, fill_value=0)


@xtructure_dataclass(bitpack="off")
class LayoutAggregateIneligible:
    id: FieldDescriptor.scalar(dtype=jnp.uint32)
    value: FieldDescriptor.scalar(dtype=jnp.float32)


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


def test_type_layout_flat_primitive_fields():
    layout = get_type_layout(LayoutPrimitive)

    assert layout.field_names == ("id", "vector")
    assert [leaf.path for leaf in layout.leaves] == [("id",), ("vector",)]
    assert layout.field_for("vector").intrinsic_shape == (3,)
    assert layout.default_shape.vector == (3,)
    assert layout.default_dtype.id == jnp.uint32
    assert layout.default_dtype.__class__.__name__ == "default_dtype"

    assert [plan.name for plan in layout.adapter_field_plans] == ["id", "vector"]
    id_plan = layout.adapter_field_plan_for("id")
    vector_plan = layout.adapter_field_plan_for("vector")
    assert id_plan.field_kind == "primitive"
    assert id_plan.path == ("id",)
    assert id_plan.dotted_path == "id"
    assert jnp.dtype(id_plan.declared_dtype) == jnp.dtype(jnp.uint32)
    assert id_plan.intrinsic_shape == ()
    assert id_plan.random_kind == "bits_int"
    assert id_plan.is_primitive_jax_dtype
    assert vector_plan.random_kind == "float"
    assert vector_plan.intrinsic_shape == (3,)


def test_type_layout_normalizes_direct_int_intrinsic_shape():
    layout = get_type_layout(LayoutIntIntrinsicShape)

    assert layout.field_for("value").intrinsic_shape == (3,)
    assert LayoutIntIntrinsicShape.default_shape.value == (3,)
    assert LayoutIntIntrinsicShape.default().value.shape == (3,)


def test_type_layout_nested_scalar_and_tensor_paths():
    scalar_layout = get_type_layout(LayoutNestedScalar)
    tensor_layout = get_type_layout(LayoutNestedTensor)

    assert [leaf.path for leaf in scalar_layout.leaves] == [("inner", "val")]
    assert scalar_layout.leaves[0].intrinsic_shape == ()

    assert [leaf.path for leaf in tensor_layout.leaves] == [("inner_array", "val")]
    assert tensor_layout.leaves[0].intrinsic_shape == (2,)
    assert tensor_layout.field_for("inner_array").is_nested


def test_type_layout_bitpack_eligibility_and_packed_fields():
    eligible = get_type_layout(LayoutAggregateEligible)
    ineligible = get_type_layout(LayoutAggregateIneligible)
    packed = get_type_layout(LayoutPackedField)

    assert eligible.aggregate_bitpack.eligible
    assert eligible.aggregate_bitpack.total_bits == 7 + 3 * 12
    assert [leaf.path for leaf in eligible.aggregate_bitpack.leaves] == [
        ("flags",),
        ("codes",),
    ]
    assert eligible.leaf_for(("flags",)).io_pack_bits == 1
    assert eligible.leaf_for(("codes",)).io_pack_bits == 12

    assert not ineligible.aggregate_bitpack.eligible
    assert "bits" in ineligible.aggregate_bitpack.reason
    assert ineligible.aggregate_bitpack.reason_kind is AggregateBitpackReason.MISSING_BITS

    assert [field.name for field in packed.packed_fields] == ["flags"]
    assert packed.packed_fields[0].packed_bits == 1
    assert packed.packed_fields[0].unpacked_intrinsic_shape == (17,)
    assert packed.leaf_for(("flags",)).io_pack_bits is None

    packed_layout = packed.packed_field_layouts[0]
    assert packed_layout.name == "flags"
    assert packed_layout.path == ("flags",)
    assert packed_layout.packed_bits == 1
    assert packed_layout.unpacked_intrinsic_shape == (17,)
    assert packed_layout.value_count == 17
    assert packed_layout.packed_byte_count == packed.default_shape.flags[0]
    assert packed_layout.io_pack_bits is None


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


def test_type_layout_namedtuple_class_names_preserved():
    """Public namedtuple class names are part of the API contract for shape/dtype."""
    layout = get_type_layout(LayoutPrimitive)

    assert layout.shape_tuple_cls.__name__ == "shape"
    assert layout.dtype_tuple_cls.__name__ == "dtype"
    assert layout.default_shape.__class__.__name__ == "default_shape"
    assert layout.default_dtype.__class__.__name__ == "default_dtype"


def test_aggregate_bitpack_reason_kind_scalar_nested():
    """A nested xtructure field with non-scalar intrinsic shape is SCALAR_NESTED."""
    layout = get_type_layout(LayoutNestedTensor)

    assert not layout.aggregate_bitpack.eligible
    assert layout.aggregate_bitpack.reason_kind is AggregateBitpackReason.SCALAR_NESTED
    assert "scalar nested" in layout.aggregate_bitpack.reason


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


def test_aggregate_zero_leaf_layout_does_not_auto_recurse():
    layout = get_type_layout(LayoutNestedEmptyAuto)

    assert not layout.aggregate_bitpack.eligible
    assert "primitive leaf" in layout.aggregate_bitpack.reason
    assert layout.aggregate_bitpack.reason_kind is AggregateBitpackReason.NO_LEAVES
    assert not hasattr(LayoutNestedEmptyAuto, "Packed")


def test_forced_aggregate_zero_leaf_layout_is_rejected():
    with pytest.raises(ValueError, match="primitive leaf"):

        @xtructure_dataclass(bitpack="aggregate")
        class LayoutForcedEmptyAggregate:
            empty: FieldDescriptor.scalar(dtype=LayoutEmpty)


def test_instance_layout_single_batched_and_unstructured():
    single = LayoutPrimitive.default()
    single_layout = get_instance_layout(single)
    assert hash(single_layout)
    assert single_layout.cls is LayoutPrimitive
    assert single_layout.field_shape_for("id") == ()
    assert single_layout.field_shape_for("vector") == (3,)
    assert single_layout.structured_type == StructuredType.SINGLE
    assert single_layout.batch_shape == ()
    assert single_layout.shape_tuple == single.shape

    batched = LayoutPrimitive.default(shape=(2, 4))
    batched_layout = get_instance_layout(batched)
    assert batched_layout.structured_type == StructuredType.BATCHED
    assert batched_layout.batch_shape == (2, 4)
    assert batched_layout.shape_tuple == batched.shape

    unstructured = LayoutPrimitive(
        id=jnp.array(1, dtype=jnp.uint32),
        vector=jnp.ones((2,), dtype=jnp.float32),
    )
    unstructured_layout = get_instance_layout(unstructured)
    assert unstructured_layout.structured_type == StructuredType.UNSTRUCTURED
    assert unstructured_layout.batch_shape == -1
    assert unstructured_layout.mismatch_reason is not None
    assert unstructured.shape.batch == -1


def test_instance_layout_nested_scalar_and_tensor_shapes():
    scalar = LayoutNestedScalar.default(shape=(5,))
    scalar_layout = get_instance_layout(scalar)
    assert scalar_layout.batch_shape == (5,)
    assert scalar_layout.shape_tuple == scalar.shape
    assert scalar_layout.shape_tuple.inner.batch == ()

    tensor = LayoutNestedTensor.default()
    tensor_layout = get_instance_layout(tensor)
    assert tensor_layout.batch_shape == ()
    assert tensor_layout.shape_tuple == tensor.shape
    assert tensor_layout.shape_tuple.inner_array.batch == (2,)


def test_instance_layout_accepts_compatible_nested_shape_tuple():
    equivalent_inner = LayoutInnerEquivalent.default(shape=(5,))
    outer = LayoutNestedScalar(inner=equivalent_inner)

    layout = get_instance_layout(outer)

    assert layout.structured_type == StructuredType.BATCHED
    assert layout.batch_shape == (5,)
    assert layout.shape_tuple.inner.batch == ()
    assert layout.mismatch_reason is None


def test_instance_layout_reports_nested_unstructured_tensor_field():
    inner = LayoutInnerPair(
        a=jnp.ones((3,), dtype=jnp.int32),
        b=jnp.ones((4, 2), dtype=jnp.int32),
    )
    outer = LayoutNestedUnstructuredTensor(inner_array=inner)

    layout = get_instance_layout(outer)

    assert layout.batch_shape == -1
    assert layout.mismatch_reason is not None
    assert "UNSTRUCTURED" in layout.mismatch_reason


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


def test_layout_traversal_iterates_and_reads_leaf_values():
    instance = LayoutNestedTensor.default()

    leaf_values = list(iter_leaf_values(instance))

    assert [leaf.path for leaf, _ in leaf_values] == [("inner_array", "val")]
    assert leaf_values[0][1] is instance.inner_array.val
    assert get_path_value(instance, ("inner_array", "val")) is instance.inner_array.val


def test_layout_traversal_rebuilds_original_and_type_mapped_views():
    leaf_values = {("inner", "val"): jnp.array(7, dtype=jnp.int32)}

    rebuilt = build_instance_from_leaf_values(LayoutNestedScalar, leaf_values)
    assert isinstance(rebuilt, LayoutNestedScalar)
    assert isinstance(rebuilt.inner, LayoutInner)
    assert int(rebuilt.inner.val) == 7

    rebuilt_view = build_instance_from_leaf_values(
        LayoutNestedScalar,
        leaf_values,
        type_map={
            LayoutNestedScalar: LayoutNestedScalarView,
            LayoutInner: LayoutInnerView,
        },
    )
    assert isinstance(rebuilt_view, LayoutNestedScalarView)
    assert isinstance(rebuilt_view.inner, LayoutInnerView)
    assert int(rebuilt_view.inner.val) == 7


def test_layout_traversal_handles_empty_dataclass():
    instance = LayoutEmpty.default()

    assert list(iter_leaf_values(instance)) == []

    rebuilt = build_instance_from_leaf_values(LayoutEmpty, {})
    assert isinstance(rebuilt, LayoutEmpty)
