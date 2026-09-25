---
version: alpha
name: Xtructure Text Renders
description: Visual design of Xtructure's host-side string renders - single records, batched instances and benchmark tables.
omitted:
  - section: colors
    reason: Xtructure emits no color. Cell text from user formatters (for example PuXle's ANSI boards) passes through unchanged.
  - section: rounded
    reason: Plain text only; no frames or corners.
typography:
  body:
    fontFamily: monospace
    fontWeight: 400
spacing:
  cell-separator: 3
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

Xtructure's renders are plain, colorless and dependency-free text. The library sits at the bottom
of the Xtructure → PuXle → JAxtar stack, so its job is to describe a data structure faithfully and
leave styling to the layers above. PuXle supplies colored per-item strings; JAxtar wraps
everything in rich panels.

Three surfaces:

| Surface | Trigger | Code |
| --- | --- | --- |
| Record | `str(x)` / `x.str(**kw)` on a SINGLE instance | the class's own `__str__`, else `_custom_pretty_formatter` (`xtructure/core/xtructure_decorators/pytree_adapters/string_format.py`) |
| Batch | `str(x)` / `x.str(**kw)` on a BATCHED instance | `BatchedRenderer` (`xtructure/core/display/renderer.py`), limits in `xtructure/core/display/__init__.py` |
| Benchmark tables | `xtructure_benchmarks` scripts | `xtructure_benchmarks/common.py` `print_results_table`, `xtructure_benchmarks/run_all.py` |

An instance with an unexpected shape (UNSTRUCTURED) renders as a one-line diagnostic:
`<Unstructured P data, shape: ..., default_shape: ...>`.

History: `e9878b1` introduced `BatchedRenderer` with a rich backend, and `413b70a` inlined that
rich code: a Panel titled in bold yellow, a bold green `shape:` subtitle, a borderless grid of
`Text.from_ansi` cells and a vertically centered `...`. `90d12bc` ("remove rich from xtructure
runtime") dropped `rich` from the runtime dependencies. The plain-text frame below keeps that grid
layout (cells side by side, vertically centered) without rich. Rich now lives only in JAxtar.

Out of scope: benchmark plots (`xtructure_benchmarks/visualize.py`, default matplotlib style,
10×6 in figures) and developer `__repr__`s such as `FieldDescriptor.__repr__`.

## Colors

None. Xtructure writes no SGR codes of its own. Cell strings produced by a class's `__str__` are
copied through byte for byte, so a PuXle state batch keeps PuXle's colors.

## Typography

Monospace, regular weight, no markup. Text elements:

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

- Batch frame: the title line, the subtitle line, then the rows. Each cell is split into lines
  and laid out as a column. Lines are padded to the column's visible width (ANSI SGR codes take
  no columns), shorter cells are centered vertically, and the cells are joined line by line with
  ` | ` (`cell-separator`, 3 columns). Trailing spaces are stripped, and the output ends with a
  newline.
- 1-D and N-D batches (except 2-D) are flattened. If the total is at most `max_size`
  (`MAX_PRINT_BATCH_SIZE` = 4), every item is shown. Otherwise the render shows the first
  `show_size` (`SHOW_BATCH_SIZE` = 2) items, `...`, and the last `show_size` items.
- 2-D batches render as a grid of rows and columns. Each axis is truncated independently to
  `show_size` head + `...` + `show_size` tail once it is longer than `2 × show_size`. A truncated
  row becomes a row of `...` cells. Column widths are shared across rows, so the grid lines up.
- `max_size` and `show_size` can be passed to `str(...)`. Every other keyword argument goes to the
  per-item formatter (for example `solve_config=` for PuXle states).
- Records: fields indented 2 spaces (`field-indent`); multi-line values indented 4 spaces
  (`nested-indent`).
- Benchmark tables: a title line, a header row, a dashed rule matching each column width, and
  data rows. Every column is right-justified, columns are separated by 2 spaces (`table-gutter`),
  and widths come from the widest cell. The batch size appears only on the first (`xtructure`)
  row of each pair.

## Elevation & Depth

None. Hierarchy is expressed by line order (title → subtitle → rows) and indentation only.

## Shapes

No frames, corners or box drawing. The only structural characters are ` | ` (repeated on every
line of a batch row, so it reads as a vertical divider), `...` and the dashed table rule.

## Components

| Component | Code | Contract |
| --- | --- | --- |
| `batched-title`, `batched-shape` | `BatchedRenderer._frame` | Two header lines, always present for BATCHED instances. |
| `batched-cell` | `BatchedRenderer._cell`, `_frame` | `str(...)` of the per-item formatter's output: multi-line text, SGR colors allowed, no rich objects. |
| `batched-ellipsis` | `BatchedRenderer._ellipsis_cell`, `_truncate_indices` | One `...` per elided run; `None` in the index list marks its position. |
| `record` | `_custom_pretty_formatter` | Default only when the class defines no `__str__`. A class-defined `__str__(self, **kwargs)` always wins. |
| `bench-table` | `print_results_table`, `xtructure_benchmarks/run_all.py` summary | Right-justified plain-text table on stdout. |

The truncation and slicing behavior is pinned by `tests/display/test_renderer.py`.

## Do's and Don'ts

- Do keep the runtime free of rich and of any color. Styling belongs to PuXle (item content) and
  JAxtar (chrome).
- Do make custom `__str__` accept `**kwargs`. The batch renderer forwards unknown keyword
  arguments to it.
- Do keep `BatchedRenderer` returning `str`. Callers print it or wrap it with rich's
  `Text.from_ansi`.
- Don't change `Batched <Class>` / `shape: ...` wording without checking downstream renders; they
  appear inside JAxtar panels.
- Don't render large batches in full. Respect `max_size` / `show_size`, because every shown item
  costs a host-side format call.

## Known Gaps

- `_frame` measures width in code points (`_visible_len`), so East Asian wide characters and emoji
  in cell text misalign the columns. Measuring with `wcwidth` would fix it if such cells appear.
