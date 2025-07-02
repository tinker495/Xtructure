# Xtructure Benchmarks

This directory contains comprehensive speed benchmarks comparing Xtructure's JAX-optimized data structures with standard Python implementations.

## Files

- `benchmarks.py` - Full benchmark suite (includes BGPQ, requires GPU for full functionality)
- `simple_benchmarks.py` - Simplified benchmark suite (CPU-compatible, excludes BGPQ)
- `benchmark_results.md` - Detailed analysis of benchmark results
- `BENCHMARKS_README.md` - This file

## Quick Start

### Prerequisites

Install the required dependencies:

```bash
pip install jax chex numpy tabulate
```

### Running Benchmarks

For CPU-only testing (recommended):
```bash
python3 simple_benchmarks.py
```

For full testing with GPU support:
```bash
python3 benchmarks.py
```

## Benchmark Overview

The benchmarks compare three main data structures:

1. **Stack** (Xtructure) vs **list** (Python)
2. **Queue** (Xtructure) vs **deque** (Python)  
3. **HashTable** (Xtructure) vs **dict** (Python)
4. **BGPQ** (Xtructure) vs **heapq** (Python) - GPU only

## Test Data

All benchmarks use a structured dataclass with:
- `id`: 32-bit unsigned integer
- `data`: 4-element float32 array  
- `flag`: boolean value

Test sizes: 100, 1,000, 5,000, 10,000 elements

## Key Results Summary

### Performance Crossover Points

- **Stack Push**: Xtructure becomes faster at 5,000+ elements
- **Queue Enqueue**: Xtructure becomes faster at 5,000+ elements  
- **HashTable Insert**: Xtructure becomes competitive at 10,000+ elements
- **Pop/Dequeue**: Python maintains advantage across all sizes

### When to Use Xtructure

✅ **Large datasets** (5,000+ elements)  
✅ **Batch operations**  
✅ **GPU acceleration plans**  
✅ **Structured/nested data**  
✅ **ML/scientific computing**  

### When to Use Python Standard Libraries

✅ **Small datasets** (<5,000 elements)  
✅ **Individual operations**  
✅ **Maximum single-op performance**  
✅ **Traditional applications**  
✅ **Simplicity over scalability**  

## Output

The benchmarks generate:

1. **Console output** with detailed timing results
2. **CSV file** with raw benchmark data
3. **Performance comparisons** showing speedup/slowdown ratios

Example output:
```
================================================================================
PERFORMANCE COMPARISON (Xtructure vs Python)
================================================================================
Xtructure_Stack push_batch (size 5000): 4.65x FASTER than Python_list
Xtructure_Queue enqueue_batch (size 10000): 7.32x FASTER than Python_deque
```

## Understanding Results

- **Higher ops/sec = better performance**
- **"x FASTER"** means Xtructure outperforms Python
- **"x SLOWER"** means Python outperforms Xtructure
- **Crossover points** indicate where Xtructure becomes advantageous

## Technical Notes

- Tests run on CPU backend for fair comparison
- JAX JIT compilation overhead affects small-scale performance
- Xtructure excels at batch operations due to vectorization
- Results may vary significantly on GPU backends

## Troubleshooting

### BGPQ Errors on CPU
If you see "Only interpret mode is supported on CPU backend" errors, use `simple_benchmarks.py` instead of the full benchmark suite.

### Import Errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Performance Variations
Benchmark results can vary based on:
- System load
- Python/JAX versions
- Hardware specifications
- Background processes

For more detailed analysis, see `benchmark_results.md`.