from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "merge_arrays_indices_loop",
    "merge_arrays_parallel",
    "merge_sort_split_idx",
]

_EXPORTS = {
    "merge_arrays_indices_loop": (".loop", "merge_arrays_indices_loop"),
    "merge_arrays_parallel": (".parallel", "merge_arrays_parallel"),
    "merge_sort_split_idx": (".split", "merge_sort_split_idx"),
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
