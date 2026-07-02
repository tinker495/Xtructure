"""Display Module — host-side string rendering of BATCHED xtructure instances."""

from __future__ import annotations

from .renderer import BatchedRenderer

SHOW_BATCH_SIZE = 2
MAX_PRINT_BATCH_SIZE = 4

__all__ = [
    "BatchedRenderer",
    "MAX_PRINT_BATCH_SIZE",
    "SHOW_BATCH_SIZE",
]
