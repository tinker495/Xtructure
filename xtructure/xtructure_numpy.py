"""
Xtructure NumPy - A collection of NumPy-like operations for xtructure dataclasses.

This module provides direct access to xtructure_numpy functionality.
You can import it as: import xtructure.xtructure_numpy as xnp
"""

from .core.xtructure_numpy import (
    concat,
    concatenate,
    flatten,
    pad,
    reshape,
    stack,
    take,
    unique_mask,
    update_on_condition,
    where,
)

__all__ = [
    "concat",
    "concatenate",
    "pad",
    "stack",
    "reshape",
    "flatten",
    "where",
    "take",
    "unique_mask",
    "update_on_condition",
]
