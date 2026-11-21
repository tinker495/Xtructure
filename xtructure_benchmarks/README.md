# xtructure Performance Benchmarks

## Purpose

This suite measures and compares the performance of `xtructure`'s JAX-based data structures against standard Python equivalents across batch sizes:

- `xtructure.HashTable` vs. Python `dict`
- `xtructure.BGPQ` (Heap) vs. Python `heapq`
- `xtructure.Queue` vs. Python `collections.deque`
- `xtructure.Stack` vs. Python `list`

## Prerequisites

Install benchmark dependencies in the workspace environment (or add `xtructure[benchmark]`):

```bash
uv sync --extra benchmark
```

*Note: Follow the official JAX documentation to install the CUDA or TPU extra matching your accelerator.*

## Usage

Run the full suite and plot generation from the `Xtructure` directory:

```bash
uv run python xtructure_benchmarks/run_all.py
```

Run `uv run python xtructure_benchmarks/run_all.py --help` (or `--help` on any individual benchmark script) to inspect available CLI flags (`--mode`, `--trials`, `--batch-sizes`, etc.).

## Output

All outputs are saved under `xtructure_benchmarks/results/`:

- `*_results.json`: Raw median/IQR/P99 throughput measurements (`ops_per_sec`) and recorded environment metadata.
- `*_performance.png`: Log-log scaling plots comparing `xtructure` against the Python baseline.

## Measurement caveats

- **Pass timed operands as jit arguments, not closure constants.** A jitted
  callable whose operands are all captured by closure can be constant-folded
  by XLA into precomputed outputs; remove-side ops (pop/dequeue/delete_mins)
  then report impossible throughput (40M+ ops/s). The harness passes the
  container through `args_supplier` for this reason — keep that pattern when
  adding benchmarks.
- **Per-process compile lottery.** Identical code can land in fast or slow
  XLA plans per process (observed 2-5× on hashtable/heap ops). Single-process
  A/B numbers are not comparable; compare medians from multiple alternating
  processes of each arm.
- **Host-dispatch sensitivity.** Small-kernel ops (queue/stack) are dominated
  by per-call dispatch + device-sync host time on some hosts (notably WSL2),
  which can pin walls regardless of kernel count. Interpret absolute ops/s
  with the environment block recorded in the results JSON.
