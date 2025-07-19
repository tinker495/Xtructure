import json
import os
import subprocess
import sys
from pathlib import Path

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

            xtructure_perf = xtructure_data[op][largest_batch_idx]
            python_perf = python_data[op][largest_batch_idx]

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


def run_script(script_path: Path):
    """
    Runs a Python script in a subprocess with the correct PYTHONPATH.
    """
    print(f"--- Running {script_path.name} ---")

    # Copy the current environment and add the project root to PYTHONPATH
    env = os.environ.copy()
    project_root = Path(__file__).parent.parent
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")

    try:
        subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            env=env,
            capture_output=True,  # Disable capturing to see rich output in real-time
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

    # 1. Find and run all benchmark scripts
    benchmark_scripts = sorted(benchmarks_dir.glob("benchmark_*.py"))
    print(f"Found {len(benchmark_scripts)} benchmark scripts to run.")

    for script in benchmark_scripts:
        run_script(script)

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
