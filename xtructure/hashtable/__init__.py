from __future__ import annotations

from xtructure._lazy_imports import lazy_dir, load_lazy_export

__all__ = ["HashTable", "HashIdx"]

_EXPORTS = {
    "HashTable": (".table", "HashTable"),
    "HashIdx": (".types", "HashIdx"),
}


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
