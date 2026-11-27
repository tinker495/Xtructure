"""Display Module — host-side string rendering of BATCHED xtructure instances."""

from __future__ import annotations

from .renderer import BatchDisplayOptions, BatchedRenderer

SHOW_BATCH_SIZE = BatchDisplayOptions.show_size
MAX_PRINT_BATCH_SIZE = BatchDisplayOptions.max_size

__all__ = [
    "BatchDisplayOptions",
    "BatchedRenderer",
    "MAX_PRINT_BATCH_SIZE",
    "SHOW_BATCH_SIZE",
]
