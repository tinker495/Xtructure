"""Deduplication utilities for dataclass batches."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["unique_mask"]


def __getattr__(name: str) -> Any:
    if name != "unique_mask":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(".optimized_unique_ops", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
