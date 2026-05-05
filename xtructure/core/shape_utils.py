"""Shape normalization helpers shared by schema and layout code."""

from __future__ import annotations

from typing import Any

import numpy as np


def normalize_shape(shape: Any) -> tuple[int, ...]:
    """Normalize xtructure intrinsic/batch shape declarations to tuples."""
    if shape is None:
        return ()
    if isinstance(shape, tuple):
        return shape
    if isinstance(shape, (int, np.integer)):
        return (int(shape),)
    return tuple(shape)


__all__ = ["normalize_shape"]
