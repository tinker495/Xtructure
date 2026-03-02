import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.table import Table

from xtructure_benchmarks.common import human_format

# Benchmark scripts with standard CLI (--mode, --trials, --batch-sizes).
# Specialized scripts (benchmark_heap_small, benchmark_heap_components) have
# different CLI interfaces and must be run individually.
BENCHMARK_SCRIPTS = {
    "benchmark_hashtable": "HashTable",
    "benchmark_heap": "Heap (BGPQ)",
    "benchmark_queue": "Queue",
    "benchmark_stack": "Stack",
}


def display_summary_table():
    """
    Finds all result files and displays a summary table of the performance
    at the largest batch size.
    """
    console = Console()
    table = Table(
        title="🏆 Benchmark Summary (Largest Batch Size) 🏆",
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("Data Structure", style="cyan")
    table.add_column("Operation", style="green")
    table.add_column("xtructure Ops/Sec", justify="right", style="bold blue")
    table.add_column("Python Ops/Sec", justify="right", style="bold blue")
    table.add_column("Ratio (xtructure/Python)", justify="right")

    results_dir = Path(__file__).parent / "results"
    for results_file in sorted(results_dir.glob("*_results.json")):
        with open(results_file, "r") as f:
            data = json.load(f)

        data_structure = results_file.stem.split("_")[0].capitalize()

        # Get the index for the largest batch size's results
        largest_batch_idx = -1

        xtructure_data = data["xtructure"]
        python_data = data.get("python")
        if python_data is None:
            continue
        operations = list(xtructure_data.keys())

        for i, op in enumerate(operations):
            op_name = op.replace("_ops_per_sec", "")

            xtructure_result = xtructure_data[op][largest_batch_idx]
            python_result = python_data[op][largest_batch_idx]

            # Handle both old and new result formats
            if isinstance(xtructure_result, dict):
                xtructure_perf = xtructure_result["median"]
            else:
                xtructure_perf = xtructure_result

            if isinstance(python_result, dict):
                python_perf = python_result["median"]
            else:
                python_perf = python_result

            ratio = xtructure_perf / python_perf if python_perf > 0 else float("inf")

            ratio_style = "bold green" if ratio > 1 else "bold red"

            table.add_row(
                data_structure if i == 0 else "",
                op_name,
                f"{human_format(xtructure_perf)}",
                f"{human_format(python_perf)}",
                f"[{ratio_style}]{ratio:.2f}x[/{ratio_style}]",
            )

    console.print(table)


def run_script(script_path: Path, extra_args: Optional[List[str]] = None):
    """
    Runs a Python script in a subprocess with the correct PYTHONPATH.
    """
    print(f"--- Running {script_path.name} ---")

    # Copy the current environment and add the project root to PYTHONPATH
    env = os.environ.copy()
    project_root = Path(__file__).parent.parent
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")

    # Enforce deterministic hash seed for fair comparison
    env["PYTHONHASHSEED"] = "42"

    # Check if we need to enable JAX 64-bit mode
    # We pass this via environment variable so it's set before any JAX import in the subprocess
    if "JAX_ENABLE_X64" in os.environ:
        env["JAX_ENABLE_X64"] = os.environ["JAX_ENABLE_X64"]

    try:
        cmd = [sys.executable, str(script_path)]
        if extra_args:
            cmd.extend(extra_args)
        subprocess.run(
            cmd,
            check=True,
            env=env,
            text=True,
        )
        print(f"--- Finished {script_path.name} successfully ---")
    except subprocess.CalledProcessError as e:
        print(f"!!! Error running {script_path.name} !!!")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
        # Re-raise the exception to stop the main script
        raise


def _filter_scripts(
    benchmarks_dir: Path,
    run_all: bool,
    only: Optional[str],
) -> List[Path]:
    """Return the list of benchmark scripts to run based on CLI flags."""
    # Only consider scripts registered in BENCHMARK_SCRIPTS (excludes meta-scripts
    # like benchmark_branches.py which have incompatible CLI interfaces)
    all_scripts = sorted(
        s for s in benchmarks_dir.glob("benchmark_*.py") if s.stem in BENCHMARK_SCRIPTS
    )

    if run_all or only is None:
        return all_scripts

    # Parse --only filter (comma-separated, matched against stem or display name)
    requested = {s.strip().lower() for s in only.split(",") if s.strip()}
    filtered: List[Path] = []
    for script in all_scripts:
        stem = script.stem  # e.g. "benchmark_hashtable"
        short_name = stem.replace("benchmark_", "")  # e.g. "hashtable"
        display_name = BENCHMARK_SCRIPTS.get(stem, "").lower()

        if short_name in requested or stem in requested or display_name in requested:
            filtered.append(script)

    if not filtered:
        available = ", ".join(s.stem.replace("benchmark_", "") for s in all_scripts)
        print(f"⚠️  No benchmarks matched --only={only}")
        print(f"   Available: {available}")

    return filtered


def main():
    """
    Finds and runs all benchmark scripts, then runs the visualization script.
    """
    benchmarks_dir = Path(__file__).parent

    # Ensure results directory exists
    results_dir = benchmarks_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Build pass-through args for each benchmark
    parser = argparse.ArgumentParser(description="Run xtructure benchmarks")
    parser.add_argument(
        "--all",
        action="store_true",
        dest="run_all",
        help="Run all benchmarks (default behavior)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated list of benchmarks to run (e.g. hashtable,heap,queue,stack)",
    )
    parser.add_argument("--mode", choices=["kernel", "e2e"], default="e2e")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="",
        help="Comma-separated batch sizes (e.g. 1024,4096,16384)",
    )
    parser.add_argument(
        "--python-heap-insert-mode",
        choices=["auto", "bulk", "incremental"],
        default="auto",
        help="Choose Python heap insert algorithm for fairness",
    )
    parser.add_argument(
        "--use-float32",
        action="store_true",
        help="Use JAX default precision (float32) instead of strict float64 (Default is strict mode)",
    )
    args, _ = parser.parse_known_args()

    # Filter benchmark scripts
    benchmark_scripts = _filter_scripts(benchmarks_dir, args.run_all, args.only)
    print(f"Found {len(benchmark_scripts)} benchmark scripts to run.")

    common_args: List[str] = ["--mode", args.mode, "--trials", str(args.trials)]
    if args.batch_sizes:
        common_args += ["--batch-sizes", args.batch_sizes]

    if not args.use_float32:
        # Set environment variable for this process and all subprocesses
        os.environ["JAX_ENABLE_X64"] = "True"
        print("🚀 Strict Fairness Mode (Default): JAX 64-bit floats ENABLED")
    else:
        print("⚠️  Relaxed Mode: Using JAX default precision (float32)")

    for script in benchmark_scripts:
        extra = list(common_args)
        if script.name == "benchmark_heap.py":
            extra += ["--python-heap-insert-mode", args.python_heap_insert_mode]
        run_script(script, extra_args=extra)

    # 2. Run the visualization script
    visualize_script = benchmarks_dir / "visualize.py"
    if visualize_script.exists():
        run_script(visualize_script)
    else:
        print(f"Warning: Visualization script not found at {visualize_script}")

    print("\n✅ All benchmarks and visualizations completed successfully!")

    display_summary_table()


if __name__ == "__main__":
    main()
