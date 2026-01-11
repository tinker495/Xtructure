"""Refactored dataclass operations package."""

from __future__ import annotations

from .batch_ops import concat, pad, split, stack, take, take_along_axis, tile
from .fill_ops import full_like, ones_like, zeros_like
from .logical_ops import update_on_condition, where, where_no_broadcast
from .shape_ops import (
    expand_dims,
    flatten,
    repeat,
    reshape,
    squeeze,
    swapaxes,
    transpose,
)
from .unique_ops import unique_mask

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
