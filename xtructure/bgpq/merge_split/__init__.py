from __future__ import annotations

from typing import TYPE_CHECKING

from xtructure._lazy_imports import lazy_dir, load_lazy_export

if TYPE_CHECKING:
    from .parallel import merge_arrays_parallel as merge_arrays_parallel
    from .split import merge_sort_split_idx as merge_sort_split_idx

_EXPORTS = {
    "merge_arrays_parallel": (".parallel", "merge_arrays_parallel"),
    "merge_sort_split_idx": (".split", "merge_sort_split_idx"),
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
