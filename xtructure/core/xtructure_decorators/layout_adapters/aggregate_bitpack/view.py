"""Logical unpacked view class builder for aggregate bitpack."""

from __future__ import annotations

from typing import Any

from xtructure.core.field_descriptors import FieldDescriptor
from xtructure.core.layout.type_layout import get_type_layout

from .generated import GENERATED_UNPACKED_VIEW_ROLE, make_generated_class


def build_unpacked_view_cls(
    root_cls: type,
) -> tuple[type, dict[type, type]]:
    """Return the logical view type for the aggregate-packed class and a cache."""
    view_cache: dict[type, type] = {}
    root_aggregate = get_type_layout(root_cls).aggregate_bitpack

    def _build_view_type(orig: type) -> type:
        cached = view_cache.get(orig)
        if cached is not None:
            return cached

        annotations: dict[str, Any] = {}
        for field in root_aggregate.view_fields_for(orig):
            if field.nested_type is not None:
                nested_view = _build_view_type(field.nested_type)
                annotations[field.name] = FieldDescriptor.scalar(dtype=nested_view)
            else:
                annotations[field.name] = FieldDescriptor.tensor(
                    dtype=field.unpack_dtype,
                    shape=tuple(field.unpacked_shape),
                    fill_value=0,
                )
        View = make_generated_class(
            f"{orig.__name__}Unpacked",
            orig.__module__,
            annotations,
            role=GENERATED_UNPACKED_VIEW_ROLE,
            bitpack="off",
        )
        view_cache[orig] = View
        return View

    return _build_view_type(root_cls), view_cache
