"""Xtructure Layout Interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xtructure._lazy_imports import lazy_dir, load_lazy_export

if TYPE_CHECKING:
    from .instance_layout import get_instance_layout as get_instance_layout
    from .type_layout import get_type_layout as get_type_layout

_EXPORTS = {
    "get_instance_layout": (".instance_layout", "get_instance_layout"),
    "get_type_layout": (".type_layout", "get_type_layout"),
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
