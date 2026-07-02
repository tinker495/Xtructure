"""Xtructure Layout Interface."""

from __future__ import annotations

from xtructure._lazy_imports import lazy_dir, load_lazy_export

__all__ = ["get_instance_layout", "get_type_layout"]

_EXPORTS = {
    "get_instance_layout": (".instance_layout", "get_instance_layout"),
    "get_type_layout": (".type_layout", "get_type_layout"),
}


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
