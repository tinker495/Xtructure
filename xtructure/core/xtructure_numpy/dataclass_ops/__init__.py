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
    "reshape",
    "flatten",
    "transpose",
    "swapaxes",
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
    "concat": ".batch_ops",
    "pad": ".batch_ops",
    "stack": ".batch_ops",
    "take": ".batch_ops",
    "take_along_axis": ".batch_ops",
    "tile": ".batch_ops",
    "split": ".batch_ops",
    "reshape": ".shape_ops",
    "flatten": ".shape_ops",
    "transpose": ".shape_ops",
    "swapaxes": ".shape_ops",
    "expand_dims": ".shape_ops",
    "squeeze": ".shape_ops",
    "repeat": ".shape_ops",
    "where": ".logical_ops",
    "where_no_broadcast": ".logical_ops",
    "update_on_condition": ".logical_ops",
    "unique_mask": ".unique_ops",
    "full_like": ".fill_ops",
    "zeros_like": ".fill_ops",
    "ones_like": ".fill_ops",
}


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
