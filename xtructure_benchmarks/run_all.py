import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from common import human_format
from rich.console import Console
from rich.table import Table


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
        python_data = data["python"]
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


def main():
    """
    Finds and runs all benchmark scripts, then runs the visualization script.
    """
    benchmarks_dir = Path(__file__).parent

    # Ensure results directory exists
    results_dir = benchmarks_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # 1. Find and run all benchmark scripts
    benchmark_scripts = sorted(benchmarks_dir.glob("benchmark_*.py"))
    print(f"Found {len(benchmark_scripts)} benchmark scripts to run.")

    # Build pass-through args for each benchmark
    parser = argparse.ArgumentParser(description="Run all xtructure benchmarks")
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
