"""Layout-owned traversal and reconstruction helpers."""

from __future__ import annotations

from typing import Any, Mapping

import jax.numpy as jnp

from .type_layout import get_type_layout
from .types import LeafLayout


def get_path_value(instance: Any, path: tuple[str, ...]) -> Any:
    """Return the value at a nested xtructure path."""
    value = instance
    for part in path:
        value = getattr(value, part)
    return value


def iter_leaf_values(instance: Any):
    """Yield `(LeafLayout, value)` pairs for primitive leaves in layout order."""
    type_layout = get_type_layout(instance.__class__)
    for leaf in type_layout.leaves:
        yield leaf, get_path_value(instance, leaf.path)


def build_instance_from_leaf_values(
    cls: type,
    leaf_values: Mapping[tuple[str, ...], Any],
    *,
    type_map: Mapping[type, type] | None = None,
    cast_declared: bool = False,
    prefix: tuple[str, ...] = (),
) -> Any:
    """Reconstruct a nested instance from primitive leaf values.

    `type_map` lets aggregate bitpack rebuild logical view classes while using
    the original Type Layout as the traversal source of truth.
    """
    type_layout = get_type_layout(cls)
    target_cls = type_map.get(cls, cls) if type_map is not None else cls
    kwargs: dict[str, Any] = {}

    for field in type_layout.fields:
        path = prefix + (field.name,)
        if field.is_nested:
            kwargs[field.name] = build_instance_from_leaf_values(
                field.nested_type,  # type: ignore[arg-type]
                leaf_values,
                type_map=type_map,
                cast_declared=cast_declared,
                prefix=path,
            )
            continue

        value = leaf_values[path]
        if cast_declared:
            try:
                value = jnp.asarray(value).astype(field.dtype)
            except TypeError:
                pass
        kwargs[field.name] = value

    return target_cls(**kwargs)


__all__ = [
    "LeafLayout",
    "build_instance_from_leaf_values",
    "get_path_value",
    "iter_leaf_values",
]
