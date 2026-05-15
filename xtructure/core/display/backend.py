"""Render Backend Protocol.

The Interface that **Batched Dataclass Renderer** consumes for grid assembly.
Backends return opaque values for ``Cell`` / ``Row`` / ``Grid`` / ``Frame``;
the Renderer only threads them between the six methods without inspecting them.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence


class RenderBackend(Protocol):
    """Semantic grid-assembly Interface.

    Implementations own:
    - ANSI / styling for cell content
    - Alignment for the ellipsis placeholder
    - Row, table, and outer-frame construction
    - Final serialization to ``str``

    The Renderer owns:
    - 2D-grid-vs-1D-flat layout decision
    - Truncation index selection
    - Per-cell content via the single-item formatter
    """

    def cell(self, content: str) -> Any:
        """Build a cell from a (possibly ANSI-coloured) string."""

    def ellipsis_cell(self) -> Any:
        """Build the truncation placeholder cell."""

    def row(self, cells: Sequence[Any]) -> Any:
        """Bundle cells into one row."""

    def grid(self, rows: Sequence[Any]) -> Any:
        """Bundle rows into a grid."""

    def frame(self, grid: Any, *, title: str, subtitle: str) -> Any:
        """Wrap the grid in an outer frame carrying title + subtitle."""

    def to_str(self, frame: Any) -> str:
        """Serialize the frame to a string."""
