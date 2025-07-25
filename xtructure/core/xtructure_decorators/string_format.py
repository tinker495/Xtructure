from io import StringIO
from typing import Type, TypeVar

import jax.numpy as jnp
import numpy as np
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from xtructure.core.structuredtype import StructuredType

from .annotate import MAX_PRINT_BATCH_SIZE, SHOW_BATCH_SIZE

T = TypeVar("T")


def add_string_representation_methods(cls: Type[T]) -> Type[T]:
    """
    Adds custom `__str__` and `str` methods to the class for generating
    a more informative string representation.

    It handles instances categorized by `structured_type` differently:
    - `SINGLE`: Uses the original `__str__` (or `repr` if basic) of the instance.
    - `BATCHED`: Provides a summarized view if the batch is large, showing
      the first few and last few elements, along with the batch shape.
      Uses `rich` for formatting.
    - `UNSTRUCTURED`: Indicates that the data is unstructured relative to its
      default shape.
    """
    _original_str_method = getattr(cls, "__str__", None)

    if _original_str_method is None or _original_str_method == object.__str__:
        _single_item_formatter = _custom_pretty_formatter
    else:

        def _single_item_formatter(item, **k):
            return _original_str_method(item, **k)

    def get_str(self, use_kwargs: bool = False, **kwargs) -> str:
        structured_type = self.structured_type

        if structured_type == StructuredType.SINGLE:
            if use_kwargs:
                return _single_item_formatter(self, **kwargs)
            else:
                return _single_item_formatter(self)

        elif structured_type == StructuredType.BATCHED:
            batch_shape = self.shape.batch
            console_buffer = StringIO()
            console = Console(file=console_buffer, force_terminal=True)

            def get_slice_str(current_state_slice):
                if use_kwargs:
                    return Text.from_ansi(_single_item_formatter(current_state_slice, **kwargs))
                else:
                    return Text.from_ansi(_single_item_formatter(current_state_slice))

            table = Table(show_header=False, show_edge=False, box=None)

            # Handle 2D batches as a grid
            if len(batch_shape) == 2:
                rows, cols = batch_shape
                truncate_rows = rows > (SHOW_BATCH_SIZE * 2)
                truncate_cols = cols > (SHOW_BATCH_SIZE * 2)

                row_indices = list(range(rows))
                if truncate_rows:
                    row_indices = (
                        list(range(SHOW_BATCH_SIZE))
                        + [None]
                        + list(range(rows - SHOW_BATCH_SIZE, rows))
                    )

                col_indices = list(range(cols))
                if truncate_cols:
                    col_indices = (
                        list(range(SHOW_BATCH_SIZE))
                        + [None]
                        + list(range(cols - SHOW_BATCH_SIZE, cols))
                    )

                for r in row_indices:
                    if r is None:
                        # Add a row of '...' for truncated rows
                        table.add_row(
                            *[Align.center(Text("..."), vertical="middle") for _ in col_indices]
                        )
                        continue

                    row_cells = []
                    for c in col_indices:
                        if c is None:
                            row_cells.append(Align.center(Text("..."), vertical="middle"))
                        else:
                            item_str = get_slice_str(self[(r, c)])
                            row_cells.append(item_str)
                    table.add_row(*row_cells)
            # Handle 1D and >2D batches as a flattened horizontal line
            else:
                py_batch_len = int(np.prod(batch_shape))

                results = []
                if py_batch_len <= MAX_PRINT_BATCH_SIZE:
                    for i in range(py_batch_len):
                        index = jnp.unravel_index(i, batch_shape)
                        results.append(get_slice_str(self[index]))
                else:
                    for i in range(SHOW_BATCH_SIZE):
                        index = jnp.unravel_index(i, batch_shape)
                        results.append(get_slice_str(self[index]))

                    results.append(Align.center(Text("..."), vertical="middle"))

                    for i in range(py_batch_len - SHOW_BATCH_SIZE, py_batch_len):
                        index = jnp.unravel_index(i, batch_shape)
                        results.append(get_slice_str(self[index]))
                table.add_row(*results)

            panel = Panel(
                table,
                title=f"[yellow bold]Batched {cls.__name__}[/yellow bold]",
                subtitle=f"[green bold]shape: {batch_shape}[/green bold]",
                expand=False,
            )
            console.print(panel)
            return console_buffer.getvalue()

        else:  # UNSTRUCTURED or any other case
            return f"<Unstructured {cls.__name__} data, shape: {self.shape}, default_shape: {self.default_shape}>"

    setattr(cls, "__str__", lambda self, **kwargs: get_str(self, use_kwargs=False, **kwargs))
    setattr(cls, "str", lambda self, **kwargs: get_str(self, use_kwargs=True, **kwargs))
    return cls


def _custom_pretty_formatter(item, **_kwargs):
    class_name = item.__class__.__name__

    field_values = {}
    if hasattr(item, "__dataclass_fields__"):
        for field_name_df in getattr(item, "__dataclass_fields__", {}).keys():
            try:
                field_values[field_name_df] = getattr(item, field_name_df)
            except AttributeError:
                pass
    elif hasattr(item, "__dict__"):
        field_values = item.__dict__
    else:
        return repr(item)

    if not field_values:
        return f"{class_name}()"

    parts = []
    for name, value in field_values.items():
        try:
            value_str = str(value)
        except Exception:
            value_str = "<error converting value to string>"

        if "\n" in value_str:
            indented_value = "\n".join(["    " + line for line in value_str.split("\n")])
            parts.append(f"  {name}: \n{indented_value}")
        else:
            parts.append(f"  {name}: {value_str}")

    return f"{class_name}(\n" + ",\n".join(parts) + "\n)"
