"""Batched Dataclass Renderer.

Owns string serialization of BATCHED xtructure instances. The 2D-grid-vs-1D-flat
layout and truncation index calculation live here; cell / row / grid / frame
assembly is delegated to the **Render Backend** Interface.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import jax.numpy as jnp
import numpy as np

from .backend import RenderBackend


class BatchedRenderer:
    """Render a BATCHED xtructure instance to a string.

    Created once per xtructure class at decorator-application time with a
    class-bound single-item formatter and a backend, then captured in a
    closure for the class's ``__str__`` / ``str``.
    """

    __slots__ = ("_format_item", "_backend")

    def __init__(
        self,
        single_formatter: Callable[..., str],
        backend: RenderBackend,
    ) -> None:
        self._format_item = single_formatter
        self._backend = backend

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
        backend = self._backend

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

        frame = backend.frame(
            backend.grid(rows),
            title=title,
            subtitle=f"shape: {batch_shape}",
        )
        return backend.to_str(frame)

    def _build_2d_rows(
        self,
        instance: Any,
        batch_shape: tuple,
        show_size: int,
        formatter_kwargs: dict,
    ) -> list:
        backend = self._backend
        rows_n, cols_n = batch_shape
        row_indices = self._truncate_indices(rows_n, show_size)
        col_indices = self._truncate_indices(cols_n, show_size)

        rows = []
        for r in row_indices:
            if r is None:
                cells = [backend.ellipsis_cell() for _ in col_indices]
            else:
                cells = []
                for c in col_indices:
                    if c is None:
                        cells.append(backend.ellipsis_cell())
                    else:
                        item = instance[(r, c)]
                        cells.append(backend.cell(self._format_item(item, **formatter_kwargs)))
            rows.append(backend.row(cells))
        return rows

    def _build_1d_row(
        self,
        instance: Any,
        batch_shape: tuple,
        max_size: int,
        show_size: int,
        formatter_kwargs: dict,
    ) -> Any:
        backend = self._backend
        total = int(np.prod(batch_shape))
        cells: list = []

        if total <= max_size:
            for i in range(total):
                idx = jnp.unravel_index(i, batch_shape)
                cells.append(backend.cell(self._format_item(instance[idx], **formatter_kwargs)))
        else:
            for i in range(show_size):
                idx = jnp.unravel_index(i, batch_shape)
                cells.append(backend.cell(self._format_item(instance[idx], **formatter_kwargs)))
            cells.append(backend.ellipsis_cell())
            for i in range(total - show_size, total):
                idx = jnp.unravel_index(i, batch_shape)
                cells.append(backend.cell(self._format_item(instance[idx], **formatter_kwargs)))

        return backend.row(cells)

    @staticmethod
    def _truncate_indices(n: int, show_size: int) -> List[Optional[int]]:
        """Return indices to render; ``None`` marks an ellipsis position."""
        if n <= show_size * 2:
            return list(range(n))
        return list(range(show_size)) + [None] + list(range(n - show_size, n))
