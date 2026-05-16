"""Xtructure Layout Interface."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["get_instance_layout", "get_type_layout"]

_EXPORTS = {
    "get_instance_layout": (".instance_layout", "get_instance_layout"),
    "get_type_layout": (".type_layout", "get_type_layout"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
