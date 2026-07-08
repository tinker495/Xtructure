"""Dataclass operations package facade."""

from __future__ import annotations

from xtructure._lazy_imports import lazy_dir, load_lazy_export

__all__ = [
    "concat",
    "pad",
    "stack",
    "take",
    "take_along_axis",
    "tile",
    "split",
    "expand_dims",
    "squeeze",
    "repeat",
    "where",
    "where_no_broadcast",
    "unique_mask",
    "update_on_condition",
    "full_like",
    "zeros_like",
    "ones_like",
]

_EXPORTS = {
    "concat": (".batch_ops", "concat"),
    "pad": (".batch_ops", "pad"),
    "stack": (".batch_ops", "stack"),
    "take": (".batch_ops", "take"),
    "take_along_axis": (".batch_ops", "take_along_axis"),
    "tile": (".batch_ops", "tile"),
    "split": (".batch_ops", "split"),
    "expand_dims": (".shape_ops", "expand_dims"),
    "squeeze": (".shape_ops", "squeeze"),
    "repeat": (".shape_ops", "repeat"),
    "where": (".logical_ops", "where"),
    "where_no_broadcast": (".logical_ops", "where_no_broadcast"),
    "unique_mask": (".unique_ops", "unique_mask"),
    "update_on_condition": (".logical_ops", "update_on_condition"),
    "full_like": (".fill_ops", "full_like"),
    "zeros_like": (".fill_ops", "zeros_like"),
    "ones_like": (".fill_ops", "ones_like"),
}


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
