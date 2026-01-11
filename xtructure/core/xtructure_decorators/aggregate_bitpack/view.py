"""Logical unpacked view class builder for aggregate bitpack."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

from xtructure.core.field_descriptors import FieldDescriptor, get_field_descriptors
from xtructure.core.type_utils import is_xtructure_dataclass_type


def build_unpacked_view_cls(
    root_cls: type, *, default_unpack_dtype: Callable[[int], Any]
) -> tuple[type, dict[type, type]]:
    """Return the logical view type for the aggregate-packed class and a cache."""
    view_cache: dict[type, type] = {}

    def _build_view_type(orig: type) -> type:
        cached = view_cache.get(orig)
        if cached is not None:
            return cached

        view_name = f"{orig.__name__}Unpacked"
        View = type(view_name, (), {"__module__": orig.__module__})
        descriptors = get_field_descriptors(orig)
        annotations: dict[str, Any] = {}
        for field in dataclasses.fields(orig):
            name = field.name
            fd = descriptors.get(name)
            if fd is None:
                continue
            if is_xtructure_dataclass_type(fd.dtype):
                if tuple(fd.intrinsic_shape) not in ((),):
                    raise NotImplementedError(
                        f"aggregate_bitpack currently supports only scalar nested fields. "
                        f"Got intrinsic_shape={fd.intrinsic_shape} on {orig.__name__}.{name}."
                    )
                nested_view = _build_view_type(fd.dtype)
                annotations[name] = FieldDescriptor.scalar(dtype=nested_view)
            else:
                if fd.bits is None:
                    raise ValueError(
                        f"aggregate_bitpack requires FieldDescriptor.bits on every primitive leaf. "
                        f"Missing on {orig.__name__}.{name}."
                    )
                annotations[name] = FieldDescriptor.tensor(
                    dtype=default_unpack_dtype(int(fd.bits)),
                    shape=tuple(fd.intrinsic_shape),
                    fill_value=0,
                )
        View.__annotations__ = annotations

        # Delay import to avoid circular dependency during decorator import graph.
        from xtructure.core.xtructure_decorators import (
            xtructure_dataclass as _xtructure_dataclass,
        )

        View = _xtructure_dataclass(View, validate=False, aggregate_bitpack=False)  # type: ignore[assignment]
        view_cache[orig] = View
        return View

    return _build_view_type(root_cls), view_cache
