"""Refactored dataclass operations package."""

from __future__ import annotations

from .batch_ops import (
    block,
    column_stack,
    concat,
    dstack,
    hstack,
    pad,
    split,
    stack,
    take,
    take_along_axis,
    tile,
    vstack,
)
from .comparison_ops import allclose, equal, isclose, not_equal
from .fill_ops import full_like, ones_like, zeros_like
from .logical_ops import update_on_condition, where, where_no_broadcast
from .shape_ops import (
    atleast_1d,
    atleast_2d,
    atleast_3d,
    broadcast_arrays,
    broadcast_to,
    expand_dims,
    flatten,
    moveaxis,
    repeat,
    reshape,
    squeeze,
    swapaxes,
    transpose,
)
from .spatial_ops import flip, roll, rot90
from .type_ops import astype, can_cast, result_type
from .unique_ops import unique_mask

__all__ = [
    "concat",
    "pad",
    "stack",
    "vstack",
    "hstack",
    "dstack",
    "column_stack",
    "block",
    "take",
    "take_along_axis",
    "tile",
    "split",
    "reshape",
    "flatten",
    "transpose",
    "swapaxes",
    "moveaxis",
    "broadcast_to",
    "broadcast_arrays",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
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
    "equal",
    "not_equal",
    "isclose",
    "allclose",
    "flip",
    "roll",
    "rot90",
    "astype",
    "result_type",
    "can_cast",
]
