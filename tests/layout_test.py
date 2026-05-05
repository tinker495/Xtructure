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


def test_type_layout_flat_primitive_fields():
    layout = get_type_layout(LayoutPrimitive)

    assert layout.field_names == ("id", "vector")
    assert [leaf.path for leaf in layout.leaves] == [("id",), ("vector",)]
    assert layout.field_by_name["vector"].intrinsic_shape == (3,)
    assert layout.default_shape.vector == (3,)
    assert layout.default_dtype.id == jnp.uint32
    assert layout.default_dtype.__class__.__name__ == "default_dtype"

    assert [plan.name for plan in layout.adapter_field_plans] == ["id", "vector"]
    id_plan = layout.adapter_field_plan_by_name["id"]
    vector_plan = layout.adapter_field_plan_by_name["vector"]
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

    assert layout.field_by_name["value"].intrinsic_shape == (3,)
    assert LayoutIntIntrinsicShape.default_shape.value == (3,)
    assert LayoutIntIntrinsicShape.default().value.shape == (3,)


def test_type_layout_nested_scalar_and_tensor_paths():
    scalar_layout = get_type_layout(LayoutNestedScalar)
    tensor_layout = get_type_layout(LayoutNestedTensor)

    assert [leaf.path for leaf in scalar_layout.leaves] == [("inner", "val")]
    assert scalar_layout.leaves[0].intrinsic_shape == ()

    assert [leaf.path for leaf in tensor_layout.leaves] == [("inner_array", "val")]
    assert tensor_layout.leaves[0].intrinsic_shape == (2,)
    assert tensor_layout.field_by_name["inner_array"].is_nested


def test_type_layout_bitpack_eligibility_and_packed_fields():
    eligible = get_type_layout(LayoutAggregateEligible)
    ineligible = get_type_layout(LayoutAggregateIneligible)
    packed = get_type_layout(LayoutPackedField)

    assert eligible.aggregate_bitpack.eligible
    assert eligible.aggregate_bitpack.total_bits == 7 + 3 * 12
    assert [leaf.path for leaf in eligible.aggregate_bitpack.leaves] == [("flags",), ("codes",)]
    assert eligible.leaf_by_path[("flags",)].io_pack_bits == 1
    assert eligible.leaf_by_path[("codes",)].io_pack_bits == 12

    assert not ineligible.aggregate_bitpack.eligible
    assert "bits" in ineligible.aggregate_bitpack.reason
    assert ineligible.aggregate_bitpack.reason_kind is AggregateBitpackReason.MISSING_BITS

    assert [field.name for field in packed.packed_fields] == ["flags"]
    assert packed.packed_fields[0].packed_bits == 1
    assert packed.packed_fields[0].unpacked_intrinsic_shape == (17,)
    assert packed.leaf_by_path[("flags",)].io_pack_bits is None

    packed_layout = packed.packed_field_layouts[0]
    assert packed_layout.name == "flags"
    assert packed_layout.path == ("flags",)
    assert packed_layout.packed_bits == 1
    assert packed_layout.unpacked_intrinsic_shape == (17,)
    assert packed_layout.value_count == 17
    assert packed_layout.packed_byte_count == packed.default_shape.flags[0]
    assert packed_layout.io_pack_bits is None


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


def test_type_layout_cached_mappings_are_read_only():
    layout = get_type_layout(LayoutAggregateEligible)

    with pytest.raises(TypeError):
        layout.field_by_name["other"] = layout.fields[0]
    with pytest.raises(TypeError):
        layout.adapter_field_plan_by_name["other"] = layout.adapter_field_plans[0]
    with pytest.raises(TypeError):
        layout.leaf_by_path[("other",)] = layout.leaves[0]
    with pytest.raises(TypeError):
        layout.aggregate_bitpack.view_fields_by_owner[object] = ()


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
