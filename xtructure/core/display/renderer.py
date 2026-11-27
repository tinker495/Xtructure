"""Batched dataclass renderer."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Callable

import numpy as np
from rich.cells import cell_len
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclass(frozen=True)
class BatchDisplayOptions:
    """Item limits and spacing (in terminal columns) for a batch preview."""

    max_size: int = 4
    show_size: int = 2
    cell_spacing: int = 2

    def __post_init__(self) -> None:
        for value in (self.max_size, self.show_size, self.cell_spacing):
            if not isinstance(value, int):
                raise TypeError("Batch display limits and spacing must be integers")
            if value < 0:
                raise ValueError("Batch display limits and spacing must be nonnegative")


class BatchedRenderer:
    """Render a BATCHED xtructure instance to a string.

    Created once during class preparation and stored with the class's
    single-item formatter for the inherited ``__str__`` / ``str`` methods.
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
        batch_display: BatchDisplayOptions | None = None,
        **formatter_kwargs: Any,
    ) -> str:
        if batch_display is None:
            batch_display = BatchDisplayOptions(max_size=max_size, show_size=show_size)
        elif not isinstance(batch_display, BatchDisplayOptions):
            raise TypeError("batch_display must be a BatchDisplayOptions instance")

        shape = instance.shape
        show = batch_display.show_size
        if len(shape) == 2:
            columns = self._truncate_indices(shape[1], show)
            indices = [
                [None if r is None or c is None else (r, c) for c in columns]
                for r in self._truncate_indices(shape[0], show)
            ]
        else:
            total = prod(shape)
            flat = (
                range(total)
                if total <= batch_display.max_size
                else self._truncate_indices(total, show)
            )
            indices = [[None if i is None else np.unravel_index(i, shape) for i in flat]]

        rows = [
            [
                Text("...", justify="center")
                if index is None
                else Text.from_ansi(self._format_item(instance[index], **formatter_kwargs))
                for index in row
            ]
            for row in indices
        ]
        console = Console()
        # Batch strings are also embedded in a panel with a border and one space on each side.
        console.width = max(1, console.width - 4)
        subtitle = Text(f"shape: {shape}", style="bold green")
        panel = Panel(
            self._grid(rows, batch_display.cell_spacing, console.width - 6),
            title=Text(title, style="bold yellow"),
            subtitle=subtitle,
            padding=(0, 2),
        )
        # Rich measures the title and body but can truncate a longer subtitle.
        panel.width = max(console.measure(panel).maximum, subtitle.cell_len + 6)
        with console.capture() as capture:
            console.print(panel)
        return capture.get()

    @classmethod
    def _grid(cls, rows: list[list[Text]], spacing: int, available: int) -> Group:
        """Keep whole columns together when boards exceed the terminal width."""
        widths = [
            max((cell_len(line) for cell in column for line in cell.plain.splitlines()), default=0)
            for column in zip(*rows)
        ]
        if len(rows) > 1 and sum(widths) + spacing * (len(widths) - 1) > available:
            # Separate logical rows when one row occupies multiple terminal lines.
            return Group(
                *(Group(cls._grid([row], spacing, available), Text("")) for row in rows[:-1]),
                cls._grid([rows[-1]], spacing, available),
            )
        grids = []
        start = 0
        while start < len(widths):
            end = start + 1
            width = widths[start]
            while end < len(widths) and width + spacing + widths[end] <= available:
                width += spacing + widths[end]
                end += 1
            grid = Table.grid(padding=(0, spacing))
            for _ in range(start, end):
                grid.add_column(vertical="middle", overflow="fold")
            for row in rows:
                grid.add_row(*row[start:end])
            grids.append(grid)
            start = end
        return Group(*grids)

    @staticmethod
    def _truncate_indices(n: int, show_size: int) -> list[int | None]:
        """Return indices to render; None marks an ellipsis position."""
        if n <= show_size * 2:
            return list(range(n))
        return list(range(show_size)) + [None] + list(range(n - show_size, n))
