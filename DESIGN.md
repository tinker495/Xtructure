# Xtructure Text Rendering Design

## Stack Layering & Why

Xtructure sits at the bottom of the `Xtructure -> PuXle -> JAxtar` stack, so its host-side
renders emit plain, colorless, dependency-free text with zero SGR codes and no `rich`
dependency. Styling belongs to upper layers: PuXle supplies colored per-item strings
(whose ANSI SGR bytes pass through unchanged via `BatchedRenderer` so batched PuXle states
keep their colors), and JAxtar wraps the output in `rich` panels via `Text.from_ansi`.

## Surfaces & Rendering Contracts

- **Validation-on-render invariant**: Rendering requires valid shapes and dtypes.
  Construction validates eagerly; unchecked PyTree reconstructions (`tree_map` /
  `tree_unflatten`) are validated before rendering begins.
- **Records (`xtructure/core/display/formatting.py`)**: `_custom_pretty_formatter` is the
  fallback when a class defines no custom `__str__`. It indents fields by 2 spaces and
  multi-line field values by 4 spaces.
- **Batched instances (`xtructure/core/display/renderer.py`, limits in `__init__.py`)**:
  - `BatchedRenderer` always returns a plain `str` ending in a newline so callers can
    print directly or wrap with `rich.Text.from_ansi`.
  - Header lines keep the exact `Batched <ClassName>` title and
    `shape: <batch shape tuple>` subtitle relied on by downstream JAxtar panels.
  - Custom `__str__(self, **kwargs)` must accept `**kwargs`, because `BatchedRenderer`
    forwards unknown keyword arguments (such as `solve_config=`) from `str(batch)` /
    `batch.str(**kwargs)` to each item formatter.
  - Multi-line cells are laid out side by side separated by ` | `, centering shorter cells
    vertically and padding to visible column width while counting ANSI SGR sequences as
    zero columns. Cell strings produced by an item formatter pass through byte-for-byte.
  - Respect `max_size` (`MAX_PRINT_BATCH_SIZE = 4` for 1-D/N-D) and `show_size`
    (`SHOW_BATCH_SIZE = 2` for truncated head/tail and 2-D grid axes); never render large
    batches in full because every shown item costs a host-side formatting call.
- **Benchmark tables (`xtructure_benchmarks/common.py`, `run_all.py`)**: Right-justified
  plain-text tables with 2-space column gutters; `human_format` rounds to 3 significant
  digits before choosing `K/M/B/T`, so `999,999` prints as `1M`.

Out of scope: benchmark plots (`xtructure_benchmarks/visualize.py`) and developer
`__repr__`s such as `FieldDescriptor.__repr__`.

## Known Gaps

- `_visible_len` in `xtructure/core/display/renderer.py` strips SGR codes and counts
  Unicode code points, so East Asian wide characters and emoji in cell text misalign
  columns. Measuring with `wcwidth` would fix this if such cells appear.
