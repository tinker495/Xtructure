from __future__ import annotations

from xtructure._lazy_imports import lazy_dir, load_lazy_export

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


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
