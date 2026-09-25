"""Batched dataclass renderer."""

from __future__ import annotations

import re
from typing import Any, Callable, List, Optional

import jax.numpy as jnp
import numpy as np

# Cell text may carry ANSI SGR colors (e.g. PuXle boards); they take no columns.
# ponytail: width counts code points, so wide (CJK/emoji) glyphs misalign; use wcwidth if needed.
_SGR_RE = re.compile(r"\x1b\[[0-9;]*m")


def _visible_len(line: str) -> int:
    return len(_SGR_RE.sub("", line))


class BatchedRenderer:
    """Render a BATCHED xtructure instance to a string.

    Created once per xtructure class at decorator-application time with a
    class-bound single-item formatter, then captured in a closure for the
    class's ``__str__`` / ``str``.
    """

    __slots__ = ("_format_item",)

    def __init__(self, single_formatter: Callable[..., str]) -> None:
        self._format_item = single_formatter

    def render(
        self,
        instance: Any,
        *,
        max_size: int,
        show_size: int,
        title: str,
        **formatter_kwargs: Any,
    ) -> str:
        batch_shape = instance.shape.batch

        if len(batch_shape) == 2:
            rows = self._build_2d_rows(instance, batch_shape, show_size, formatter_kwargs)
        else:
            rows = [
                self._build_1d_row(
                    instance,
                    batch_shape,
                    max_size,
                    show_size,
                    formatter_kwargs,
                )
            ]

        frame = self._frame(
            rows,
            title=title,
            subtitle=f"shape: {batch_shape}",
        )
        return frame

    def _build_2d_rows(
        self,
        instance: Any,
        batch_shape: tuple,
        show_size: int,
        formatter_kwargs: dict,
    ) -> list:
        rows_n, cols_n = batch_shape
        row_indices = self._truncate_indices(rows_n, show_size)
        col_indices = self._truncate_indices(cols_n, show_size)

        rows = []
        for r in row_indices:
            if r is None:
                cells = [self._ellipsis_cell() for _ in col_indices]
            else:
                cells = []
                for c in col_indices:
                    if c is None:
                        cells.append(self._ellipsis_cell())
                    else:
                        item = instance[(r, c)]
                        cells.append(self._cell(self._format_item(item, **formatter_kwargs)))
            rows.append(self._row(cells))
        return rows

    def _build_1d_row(
        self,
        instance: Any,
        batch_shape: tuple,
        max_size: int,
        show_size: int,
        formatter_kwargs: dict,
    ) -> Any:
        total = int(np.prod(batch_shape))
        cells: list = []

        if total <= max_size:
            for i in range(total):
                idx = jnp.unravel_index(i, batch_shape)
                cells.append(self._cell(self._format_item(instance[idx], **formatter_kwargs)))
        else:
            for i in range(show_size):
                idx = jnp.unravel_index(i, batch_shape)
                cells.append(self._cell(self._format_item(instance[idx], **formatter_kwargs)))
            cells.append(self._ellipsis_cell())
            for i in range(total - show_size, total):
                idx = jnp.unravel_index(i, batch_shape)
                cells.append(self._cell(self._format_item(instance[idx], **formatter_kwargs)))

        return self._row(cells)

    @staticmethod
    def _truncate_indices(n: int, show_size: int) -> List[Optional[int]]:
        """Return indices to render; ``None`` marks an ellipsis position."""
        if n <= show_size * 2:
            return list(range(n))
        return list(range(show_size)) + [None] + list(range(n - show_size, n))

    @staticmethod
    def _cell(content: str) -> str:
        return str(content)

    @staticmethod
    def _ellipsis_cell() -> str:
        return "..."

    @staticmethod
    def _row(cells: list[str]) -> tuple[str, ...]:
        return tuple(cells)

    @staticmethod
    def _frame(rows: list[tuple[str, ...]], *, title: str, subtitle: str) -> str:
        # Cells are usually multi-line (records, boards): lay them out as columns
        # joined line by line, each cell centered vertically and padded to its column.
        grid = [[cell.splitlines() for cell in cells] for cells in rows]
        widths = [
            max((_visible_len(line) for cells in grid for line in cells[col]), default=0)
            for col in range(max((len(cells) for cells in grid), default=0))
        ]
        body = []
        for cells in grid:
            height = max((len(cell) for cell in cells), default=0)
            for i in range(height):
                parts = []
                for col, cell in enumerate(cells):
                    j = i - (height - len(cell)) // 2
                    line = cell[j] if 0 <= j < len(cell) else ""
                    parts.append(line + " " * (widths[col] - _visible_len(line)))
                body.append(" | ".join(parts).rstrip())
        return "\n".join([title, subtitle, *body]) + "\n"
