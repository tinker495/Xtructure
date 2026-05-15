"""Rich-based **Render Backend** implementation."""

from __future__ import annotations

from io import StringIO
from typing import Any, Sequence

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class RichBackend:
    """First-party Render Backend backed by ``rich``.

    Opaque types:
    - ``Cell`` → ``rich.text.Text`` or ``rich.align.Align``
    - ``Row``  → ``tuple`` of cells (consumed by ``Table.add_row(*cells)``)
    - ``Grid`` → ``rich.table.Table``
    - ``Frame`` → ``rich.panel.Panel``
    """

    def cell(self, content: str) -> Any:
        return Text.from_ansi(content)

    def ellipsis_cell(self) -> Any:
        return Align.center(Text("..."), vertical="middle")

    def row(self, cells: Sequence[Any]) -> Any:
        return tuple(cells)

    def grid(self, rows: Sequence[Any]) -> Any:
        table = Table(show_header=False, show_edge=False, box=None)
        for cells in rows:
            table.add_row(*cells)
        return table

    def frame(self, grid: Any, *, title: str, subtitle: str) -> Any:
        return Panel(
            grid,
            title=f"[yellow bold]{title}[/yellow bold]",
            subtitle=f"[green bold]{subtitle}[/green bold]",
            expand=False,
        )

    def to_str(self, frame: Any) -> str:
        buf = StringIO()
        Console(file=buf, force_terminal=True).print(frame)
        return buf.getvalue()
