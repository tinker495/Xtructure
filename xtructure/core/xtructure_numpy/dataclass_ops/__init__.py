"""Dataclass operations package facade."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xtructure._lazy_imports import lazy_dir, load_lazy_export

if TYPE_CHECKING:
    from .batch_ops import concat as concat
    from .batch_ops import pad as pad
    from .batch_ops import split as split
    from .batch_ops import stack as stack
    from .batch_ops import take as take
    from .batch_ops import take_along_axis as take_along_axis
    from .batch_ops import tile as tile
    from .fill_ops import full_like as full_like
    from .fill_ops import ones_like as ones_like
    from .fill_ops import zeros_like as zeros_like
    from .logical_ops import update_on_condition as update_on_condition
    from .logical_ops import where as where
    from .logical_ops import where_no_broadcast as where_no_broadcast
    from .shape_ops import expand_dims as expand_dims
    from .shape_ops import repeat as repeat
    from .shape_ops import squeeze as squeeze
    from .unique_ops import unique_mask as unique_mask

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
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    return load_lazy_export(name, __name__, _EXPORTS, globals())


def __dir__() -> list[str]:
    return lazy_dir(globals(), __all__)
