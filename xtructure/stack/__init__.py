from __future__ import annotations

import importlib
from typing import Any

__all__ = ["Stack"]


def __getattr__(name: str) -> Any:
    if name != "Stack":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(".stack", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
