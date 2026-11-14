"""Prepare the per-class state and generated members of Xtructurable subclasses."""

from collections.abc import Callable
from dataclasses import dataclass
from inspect import get_annotations
from operator import attrgetter
from typing import Any

import jax.numpy as jnp

from xtructure.core.dataclass import base_dataclass
from xtructure.core.display import BatchedRenderer
from xtructure.core.field_descriptors import _FIELD_DESCRIPTOR_ATTR
from xtructure.core.layout.type_layout import get_type_layout
from xtructure.core.layout.types import TypeLayout

from .display.formatting import _custom_pretty_formatter
from .indexing import _compile_field_setter


@dataclass(frozen=True)
class _ClassState:
    type_layout: TypeLayout
    user_post_init: Callable[..., None] | None
    single_formatter: Callable[..., str]
    renderer: BatchedRenderer
    field_getter: Callable[[Any], Any] | None
    setter: Callable[..., Any]
    condition_setter: Callable[..., Any]


def install_xtructure_members(target_cls: type, *, bitpack: str = "auto") -> None:
    """Prepare layout facts, user hooks, compiled updates, and packed classes."""
    from xtructure.core.protocol import Xtructurable

    if not issubclass(target_cls, Xtructurable):
        raise TypeError("Class preparation requires an Xtructurable subclass.")
    if bitpack not in ("auto", "aggregate", "field", "off"):
        raise ValueError(
            f"bitpack must be one of 'auto', 'aggregate', 'field', 'off', got {bitpack!r}"
        )

    reserved = (
        set(vars(Xtructurable))
        | Xtructurable.__annotations__.keys()
        | {
            _FIELD_DESCRIPTOR_ATTR,
            "__dataclass_fields__",
            "__dataclass_params__",
            "__init__",
            "__repr__",
            "__getstate__",
            "__setstate__",
        }
    )
    conflicts = reserved.intersection(get_annotations(target_cls))
    if conflicts:
        raise TypeError(
            f"{target_cls.__name__} fields conflict with Xtructurable members: "
            + ", ".join(sorted(conflicts))
        )

    user_post_init = next(
        (
            base.__post_init__
            for base in target_cls.__mro__
            if base is not Xtructurable and "__post_init__" in vars(base)
        ),
        None,
    )
    single_formatter = next(
        (
            base.__str__
            for base in target_cls.__mro__
            if base not in (Xtructurable, object) and "__str__" in vars(base)
        ),
        _custom_pretty_formatter,
    )
    # These two extension hooks retain their wrapping contract.
    target_cls.__post_init__ = Xtructurable.__post_init__
    target_cls.__str__ = Xtructurable.str
    cls = target_cls
    base_dataclass(cls, eq=False, static_fields=("_layout_cache",))
    type_layout = get_type_layout(cls)
    for plan in type_layout.adapter_field_plans:
        if plan.field_kind == "nested":
            if not isinstance(plan.nested_type, type) or not issubclass(
                plan.nested_type, Xtructurable
            ):
                raise TypeError(
                    f"Error during method creation for '{cls.__name__}': "
                    f"Nested field '{plan.name}' must declare an Xtructurable subclass."
                )
        elif not (isinstance(plan.declared_dtype, jnp.dtype) or plan.is_primitive_jax_dtype):
            raise TypeError(
                f"Error during method creation for '{cls.__name__}': "
                f"Field '{plan.name}' uses FieldDescriptor with an unsupported "
                f".dtype attribute: '{plan.declared_dtype}' "
                f"(type: {type(plan.declared_dtype).__name__}). "
                "Expected a JAX primitive type/class (like jnp.int32 or "
                "jnp.dtype('int32')), or an Xtructurable subclass (like Parent)."
            )

    cls.default_shape = type_layout.default_shape
    cls.default_dtype = type_layout.default_dtype
    cls._xtructure_state = _ClassState(
        type_layout=type_layout,
        user_post_init=user_post_init,
        single_formatter=single_formatter,
        renderer=BatchedRenderer(single_formatter),
        field_getter=attrgetter(*type_layout.field_names) if type_layout.field_names else None,
        setter=_compile_field_setter(cls, type_layout.adapter_field_plans, mode="set"),
        condition_setter=_compile_field_setter(
            cls, type_layout.adapter_field_plans, mode="set_as_condition"
        ),
    )

    aggregate = bitpack == "aggregate" or (
        bool(type_layout.fields) and bitpack == "auto" and type_layout.aggregate_bitpack.eligible
    )
    generated_names = {"Packed", "packed", "bitpack_schema"} if aggregate else set()
    if bitpack != "off":
        generated_names.update(
            f"{field.name}_unpacked" for field in type_layout.packed_field_layouts
        )
    conflicts = generated_names.intersection(type_layout.field_names)
    if conflicts:
        raise TypeError(
            f"{cls.__name__} fields conflict with generated members: "
            + ", ".join(sorted(conflicts))
        )

    if aggregate:
        from .aggregate_bitpack import add_aggregate_bitpack

        add_aggregate_bitpack(cls)
    if bitpack != "off":
        from .bitpack_accessors import add_bitpack_accessors

        add_bitpack_accessors(cls)
