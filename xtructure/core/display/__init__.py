"""Display Module — host-side string rendering of BATCHED xtructure instances."""

from __future__ import annotations

import importlib
from typing import Any

SHOW_BATCH_SIZE = 2
MAX_PRINT_BATCH_SIZE = 4

__all__ = [
    "BatchedRenderer",
    "MAX_PRINT_BATCH_SIZE",
    "RenderBackend",
    "RichBackend",
    "SHOW_BATCH_SIZE",
]

_EXPORTS = {
    "BatchedRenderer": (".renderer", "BatchedRenderer"),
    "RenderBackend": (".backend", "RenderBackend"),
    "RichBackend": (".rich_backend", "RichBackend"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(importlib.import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
