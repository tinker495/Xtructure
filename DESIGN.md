---
version: alpha
name: Xtructure Text Renders
description: Visual design of Xtructure's host-side string renders - single records, batched instances and benchmark tables.
typography:
  body:
    fontFamily: monospace
    fontWeight: 400
spacing:
  cell-separator: 2
  field-indent: 2
  nested-indent: 4
  show-size: 2
  max-print-size: 4
  table-gutter: 2
components:
  batched-title:
    typography: "{typography.body}"
  batched-shape:
    typography: "{typography.body}"
  batched-cell:
    typography: "{typography.body}"
    width: "{spacing.cell-separator}"
  batched-ellipsis:
    typography: "{typography.body}"
  record:
    typography: "{typography.body}"
    padding: "{spacing.field-indent}"
  record-nested-value:
    padding: "{spacing.nested-indent}"
  bench-table:
    typography: "{typography.body}"
    padding: "{spacing.table-gutter}"
---

# Xtructure Text Rendering Design

## Overview

Xtructure renders batch previews with a Rich table inside a fitted panel. PuXle supplies
the per-item board strings and JAxtar composes the previews into its search setup panel.

Three surfaces:

| Surface | Trigger | Code |
| --- | --- | --- |
| Record | `str(x)` / `x.str(**kw)` on a SINGLE instance | the class's own `__str__`, else `_custom_pretty_formatter` (`xtructure/core/display/formatting.py`) |
| Batch | `str(x)` / `x.str(**kw)` on a BATCHED instance | `BatchedRenderer` (`xtructure/core/display/renderer.py`), options in `BatchDisplayOptions` |
| Benchmark tables | `xtructure_benchmarks` scripts | `xtructure_benchmarks/common.py` `print_results_table`, `xtructure_benchmarks/run_all.py` |

Rendering requires valid data. Ordinary construction rejects invalid field shapes and dtypes;
unchecked PyTree reconstructions are validated when rendering starts.

Out of scope: benchmark plots (`xtructure_benchmarks/visualize.py`, default matplotlib style,
10×6 in figures) and developer `__repr__`s such as `FieldDescriptor.__repr__`.

## Colors

The batch title is bold yellow and the shape is bold green. Rich decodes ANSI styles
from each item with `Text.from_ansi` and honors the console color settings when serializing the panel.

## Typography

Monospace, literal cell text. Text elements:

| Element | Form | Example |
| --- | --- | --- |
| Record | `ClassName(` / `  field: value,` lines / `)` | `P(\n  id: 4294967295,\n  v: [inf inf inf]\n)` |
| Multi-line field value | `  field: ` on its own line, then the value indented 4 spaces | nested dataclasses, boards |
| Batch title | `Batched <ClassName>` | `Batched State` |
| Batch subtitle | `shape: <batch shape tuple>` | `shape: (2,)` |
| Elision | `...` | |
| Benchmark counts | `human_format`: 3 significant digits, `K/M/B/T` suffix | `1.5M` |
| Benchmark batch size | thousands separators | `10,000` |
| Spread | `±` + `human_format(IQR)` | `±12K` |

The benchmark `human_format` rounds to 3 significant digits before choosing the suffix, so
999,999 prints as `1M`.

## Layout

The unit is one character cell.

- The rounded batch frame centers its title in the top border and its shape in the bottom border.
  The body has two columns of horizontal padding on each side. Cells are separated by two spaces,
  padded to their column's visible width and vertically centered. ANSI SGR codes take no columns.
  The preview reserves four terminal columns for an enclosing panel. Columns that do not fit
  continue below in reading order. A blank line separates wrapped rows of a 2-D batch.
  The output ends with a newline.
- 1-D and N-D batches (except 2-D) are flattened. If the total is at most `max_size`
  (`MAX_PRINT_BATCH_SIZE` = 4), every item is shown. Otherwise the render shows the first
  `show_size` (`SHOW_BATCH_SIZE` = 2) items, `...`, and the last `show_size` items.
- 2-D batches render as a grid of rows and columns. Each axis is truncated independently to
  `show_size` head + `...` + `show_size` tail once it is longer than `2 × show_size`. A truncated
  row becomes a row of `...` cells. Column widths are shared across rows, so the grid lines up.
- Pass `batch.str(batch_display=BatchDisplayOptions(max_size=8, show_size=3, cell_spacing=1))`
  to customize the preview. Import `BatchDisplayOptions` from `xtructure`. The options object takes precedence over
  direct `max_size` and `show_size` keywords. Overlapping head and tail ranges display each item
  once. Additional formatter arguments such as `solve_config` are forwarded to each item.
- Records: fields indented 2 spaces (`field-indent`); multi-line values indented 4 spaces
  (`nested-indent`).
- Benchmark tables: a title line, a header row, a dashed rule matching each column width, and
  data rows. Every column is right-justified, columns are separated by 2 spaces (`table-gutter`),
  and widths come from the widest cell. The batch size appears only on the first (`xtructure`)
  row of each pair.

## Elevation & Depth

Batch borders group the item grid under its title. Field indentation groups nested records.

## Shapes

Rounded corners and single-line box drawing enclose the batch. Elided items use `...`.

## Components

| Component | Code | Contract |
| --- | --- | --- |
| `batched-title`, `batched-shape` | `BatchedRenderer.render` | Centered labels in the top and bottom borders. |
| `batched-cell` | `Text.from_ansi` | `str(...)` of the per-item formatter's output: multi-line text, SGR colors allowed, no rich objects. |
| `batched-ellipsis` | `BatchedRenderer._truncate_indices` | One `...` per elided run; `None` in the index list marks its position. |
| `record` | `_custom_pretty_formatter` | Default only when the class defines no `__str__`. A class-defined `__str__(self, **kwargs)` always wins. |
| `bench-table` | `print_results_table`, `xtructure_benchmarks/run_all.py` summary | Right-justified plain-text table on stdout. |

The truncation and slicing behavior is pinned by `tests/display/test_renderer.py`.

## Do's and Don'ts

- Do make custom `__str__` accept `**kwargs`. The batch renderer forwards unknown keyword
  arguments to it.
- Do keep `BatchedRenderer` returning `str`. Callers print it or wrap it with rich's
  `Text.from_ansi`.
- Don't change `Batched <Class>` / `shape: ...` wording without checking downstream renders; they
  appear inside JAxtar panels.
- Don't render large batches in full. Respect `max_size` / `show_size`, because every shown item
  costs a host-side format call.
