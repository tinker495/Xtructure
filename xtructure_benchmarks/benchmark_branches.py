import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

CONSOLE = Console()


def run_command(cmd, cwd=None, env=None):
    try:
        if isinstance(cmd, str):
            cmd = shlex.split(cmd)
        subprocess.run(cmd, check=True, cwd=cwd, env=env)
    except subprocess.CalledProcessError as e:
        CONSOLE.print(f"[bold red]Error running command:[/bold red] {cmd}")
        raise e


def get_current_branch():
    return subprocess.check_output(["git", "branch", "--show-current"]).decode().strip()


def checkout_branch(branch):
    CONSOLE.print(f"[bold cyan]Switching to branch: {branch}[/]")
    run_command(f"git checkout {branch}")


def run_benchmarks(output_dir):
    # Ensure output dir exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine repo root (parent of the directory containing this script)
    script_dir = Path(__file__).parent.resolve()
    repo_root = (
        script_dir.parent if script_dir.name == "xtructure_benchmarks" else script_dir
    )

    # Define benchmarks to run (relative to repo root)
    scripts = [
        "xtructure_benchmarks/benchmark_hashtable.py",
        "xtructure_benchmarks/benchmark_heap.py",
        "xtructure_benchmarks/benchmark_queue.py",
        "xtructure_benchmarks/benchmark_stack.py",
    ]

    # Run each script
    # We use the .venv python
    python_exe = sys.executable

    common_args = "--trials 3 --batch-sizes 1024,4096,16384"

    for script_rel in scripts:
        script_path = repo_root / script_rel
        CONSOLE.print(f"  Running {script_path.name}...")
        try:
            # We run from repo root to ensure imports (like 'xtructure') work
            run_command(f"{python_exe} {script_rel} {common_args}", cwd=repo_root)
        except Exception as e:
            CONSOLE.print(f"[bold red]Failed to run {script_rel}: {e}[/]")
            # Continue to next script to gather partial results

    # Move results to output_dir
    results_src = repo_root / "xtructure_benchmarks" / "results"
    if results_src.exists():
        for f in results_src.glob("*.json"):
            shutil.copy2(f, output_dir / f.name)


def compare_results(branches, result_dirs):
    if len(branches) < 2:
        return

    base_branch = branches[0]
    target_branch = branches[1]  # Compare first two

    base_dir = result_dirs[base_branch]
    target_dir = result_dirs[target_branch]

    # Re-use logic from compare_benchmarks.py or import it effectively
    # faster to just re-implement simple table here

    files = sorted([f.name for f in base_dir.glob("*.json")])

    for filename in files:
        base_path = base_dir / filename
        target_path = target_dir / filename

        if not target_path.exists():
            continue

        with open(base_path) as f:
            base_data = json.load(f)
        with open(target_path) as f:
            target_data = json.load(f)

        title = filename.replace("_results.json", "").capitalize()
        table = Table(title=f"{title}: {base_branch} vs {target_branch}")
        table.add_column("Op", style="cyan")
        table.add_column("Batch", justify="right")
        table.add_column(f"{base_branch} (Ops/s)", justify="right")
        table.add_column(f"{target_branch} (Ops/s)", justify="right")
        table.add_column("Speedup", justify="right")

        tasks = list(base_data["xtructure"].keys())
        batch_sizes = base_data["batch_sizes"]

        for task in tasks:
            task_clean = task.replace("_ops_per_sec", "")
            base_vals = base_data["xtructure"][task]
            target_vals = target_data["xtructure"].get(task, [])

            for i, size in enumerate(batch_sizes):
                if i >= len(target_vals):
                    break

                b_val = (
                    base_vals[i]["median"]
                    if isinstance(base_vals[i], dict)
                    else base_vals[i]
                )
                t_val = (
                    target_vals[i]["median"]
                    if isinstance(target_vals[i], dict)
                    else target_vals[i]
                )

                if b_val == 0:
                    ratio_str = "Inf"
                else:
                    ratio = t_val / b_val
                    color = (
                        "green" if ratio > 1.05 else "red" if ratio < 0.95 else "white"
                    )
                    ratio_str = f"[{color}]{ratio:.2f}x[/{color}]"

                table.add_row(
                    task_clean if i == 0 else "",
                    str(size),
                    f"{b_val:.2e}",
                    f"{t_val:.2e}",
                    ratio_str,
                )
            table.add_section()
        CONSOLE.print(table)


def main():
    parser = argparse.ArgumentParser(description="Automated Branch Benchmarking")
    parser.add_argument(
        "branches", nargs="+", help="List of branches to benchmark (space separated)"
    )
    parser.add_argument(
        "--keep-results", action="store_true", help="Keep results directories after run"
    )
    args = parser.parse_args()

    original_branch = get_current_branch()
    CONSOLE.print(f"Starting benchmark from: {original_branch}")

    result_dirs = {}

    try:
        for branch in args.branches:
            checkout_branch(branch)
            # Determine repo root again in case CWD changed
            # (though checkout_branch doesn't change python's CWD view usually)
            # Actually we just use relative path from where we are running.
            # But consistent with run_benchmarks, let's pass a path.

            # Note: run_command uses shell=True, so 'git checkout' changes the repo state.

            output_dir = Path(f"xtructure_benchmarks/results_{branch}")
            # If running from inside xtructure_benchmarks, this might create nested dirs?
            # Let's fix this script to assume it is run from ROOT.
            # But we added logic in run_benchmarks to detect root.

            # Let's assume output_dir is relative to CWD.
            run_benchmarks(output_dir)
            result_dirs[branch] = output_dir

    except Exception as e:
        CONSOLE.print(f"[bold red]Critical Error:[/bold red] {e}")
    finally:
        # Always return to original branch
        checkout_branch(original_branch)

    # Compare
    if len(args.branches) >= 2:
        compare_results(args.branches, result_dirs)

    # Cleanup
    if not args.keep_results:
        CONSOLE.print("Cleaning up result files...")
        for d in result_dirs.values():
            if d.exists():
                shutil.rmtree(d)
        # Also clean default results dir
        clean_dir = Path("xtructure_benchmarks/results")
        if clean_dir.exists():
            for f in clean_dir.glob("*.json"):
                f.unlink()


if __name__ == "__main__":
    main()
