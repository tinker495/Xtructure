"""
Xtructure NumPy - A collection of NumPy-like operations for xtructure dataclasses.

This module provides direct access to xtructure_numpy functionality.
You can import it as: import xtructure.xtructure_numpy as xnp
"""

from .core.xtructure_numpy import (
    concat,
    concatenate,
    expand_dims,
    flatten,
    full_like,
    ones_like,
    pad,
    ravel,
    repeat,
    reshape,
    split,
    squeeze,
    stack,
    swapaxes,
    take,
    take_along_axis,
    tile,
    transpose,
    unique_mask,
    update_on_condition,
    where,
    where_no_broadcast,
    zeros_like,
)

__all__ = [
    "concat",
    "concatenate",
    "pad",
    "stack",
    "reshape",
    "ravel",
    "flatten",
    "where",
    "where_no_broadcast",
    "take",
    "take_along_axis",
    "tile",
    "transpose",
    "swapaxes",
    "unique_mask",
    "update_on_condition",
    "expand_dims",
    "squeeze",
    "repeat",
    "split",
    "zeros_like",
    "ones_like",
    "full_like",
]
