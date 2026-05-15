"""Display Module — host-side string rendering of BATCHED xtructure instances.

See ``CONTEXT.md``:
- **Batched Dataclass Renderer** — :class:`BatchedRenderer`
- **Render Backend** — :class:`RenderBackend` Protocol; :class:`RichBackend`
"""

from .backend import RenderBackend
from .renderer import BatchedRenderer
from .rich_backend import RichBackend

SHOW_BATCH_SIZE = 2
MAX_PRINT_BATCH_SIZE = 4

__all__ = [
    "BatchedRenderer",
    "MAX_PRINT_BATCH_SIZE",
    "RenderBackend",
    "RichBackend",
    "SHOW_BATCH_SIZE",
]
