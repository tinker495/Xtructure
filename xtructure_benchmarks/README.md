# xtructure Performance Benchmarks

## Purpose

This suite of benchmarks is designed to measure and compare the performance of `xtructure`'s JAX-based data structures against their equivalent counterparts in standard Python. The primary focus is on how performance scales with increasing batch sizes, highlighting the advantages of GPU-based parallel processing for large-scale data manipulation.

The following data structures are benchmarked:
- `xtructure.HashTable` vs. Python `dict`
- `xtructure.BGPQ` (Heap) vs. Python `heapq`
- `xtructure.Queue` vs. Python `collections.deque`
- `xtructure.Stack` vs. Python `list`

## Prerequisites

Ensure you have the necessary libraries installed. You can install them using pip:

```bash
pip install xtructure "jax[gpu]" matplotlib
```
*Note: Please follow the official JAX documentation for detailed instructions on installing for your specific CUDA version.*

## Usage

The entire benchmark and visualization process can be run from a single script. From the root directory of the project, execute the following command:

```bash
python3 xtructure_benchmarks/run_all.py
```

This will:
1. Run the benchmarks for each data structure.
2. Save the raw performance data into `.json` files.
3. Generate performance comparison graphs from the data.
4. Save the graphs as `.png` image files.

## Output

All output files are saved in the `xtructure_benchmarks/results/` directory.

- **JSON Files (`*_results.json`)**: These files contain the raw benchmark data, measuring "operations per second" for both the `xtructure` and standard Python implementations across different batch sizes.
- **PNG Files (`*_performance.png`)**: These files are graphical visualizations of the performance data.
  - **X-axis**: Batch Size (on a log base 2 scale)
  - **Y-axis**: Operations per Second (on a log scale)
  - **Interpretation**: Higher values on the y-axis indicate better performance. The graphs clearly illustrate the performance difference as the amount of data processed in parallel increases.
