"""Display Module — host-side string rendering of BATCHED xtructure instances."""

from __future__ import annotations

from xtructure._lazy_imports import lazy_dir, load_lazy_export

SHOW_BATCH_SIZE = 2
MAX_PRINT_BATCH_SIZE = 4

__all__ = [
    "BatchedRenderer",
    "MAX_PRINT_BATCH_SIZE",
    "SHOW_BATCH_SIZE",
]

_EXPORTS = {
    "BatchedRenderer": (".renderer", "BatchedRenderer"),
}


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
