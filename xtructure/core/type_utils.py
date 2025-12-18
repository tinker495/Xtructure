from __future__ import annotations

from typing import Any


def is_xtructure_dataclass_type(value: Any) -> bool:
    """Return True if `value` is an @xtructure_dataclass type.

    Convention: xtructure marks decorated classes by setting `is_xtructed = True`.
    """
    return isinstance(value, type) and bool(getattr(value, "is_xtructed", False))


def is_xtructure_dataclass_instance(value: Any) -> bool:
    """Return True if `value` is an instance of an @xtructure_dataclass type."""
    return bool(getattr(value, "is_xtructed", False))
