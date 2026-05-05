"""Logical unpacked view class builder for aggregate bitpack."""

from __future__ import annotations

from typing import Any

from xtructure.core.field_descriptors import FieldDescriptor
from xtructure.core.layout import get_type_layout

from .generated import GENERATED_UNPACKED_VIEW_ROLE, register_generated_class


def build_unpacked_view_cls(root_cls: type) -> tuple[type, dict[type, type]]:
    """Return the logical view type for the aggregate-packed class and a cache."""
    view_cache: dict[type, type] = {}
    root_aggregate = get_type_layout(root_cls).aggregate_bitpack
    view_fields_by_owner = root_aggregate.view_fields_by_owner

    def _build_view_type(orig: type) -> type:
        cached = view_cache.get(orig)
        if cached is not None:
            return cached

        view_name = f"{orig.__name__}Unpacked"
        View = type(view_name, (), {"__module__": orig.__module__})
        annotations: dict[str, Any] = {}
        for field in view_fields_by_owner.get(orig, ()):
            if field.is_nested:
                nested_view = _build_view_type(field.nested_type)  # type: ignore[arg-type]
                annotations[field.name] = FieldDescriptor.scalar(dtype=nested_view)
            else:
                annotations[field.name] = FieldDescriptor.tensor(
                    dtype=field.unpack_dtype,
                    shape=tuple(field.unpacked_shape),
                    fill_value=0,
                )
        View.__annotations__ = annotations

        # Delay import to avoid circular dependency during decorator import graph.
        from xtructure.core.xtructure_decorators import (
            xtructure_dataclass as _xtructure_dataclass,
        )

        View = _xtructure_dataclass(  # type: ignore[assignment]
            View,
            validate=False,
            aggregate_bitpack=False,
            bitpack="off",
        )
        register_generated_class(View, role=GENERATED_UNPACKED_VIEW_ROLE)
        view_cache[orig] = View
        return View

    return _build_view_type(root_cls), view_cache
