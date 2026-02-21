import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

# Initialize Console globally to ensure consistent output style
CONSOLE = Console()


def load_result(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def format_perf(val):
    if val > 1e9:
        return f"{val / 1e9:.2f}G"
    if val > 1e6:
        return f"{val / 1e6:.2f}M"
    if val > 1e3:
        return f"{val / 1e3:.2f}K"
    return f"{val:.2f}"


@click.command()
@click.argument("base_branch", required=False, default="main")
@click.argument("target_branch", required=False, default="dev")
def main(base_branch, target_branch):
    """
    Compare benchmark results between BASE_BRANCH and TARGET_BRANCH.

    Defaults: main vs dev
    """
    # Determine repo root relative to this script
    script_dir = Path(__file__).parent.resolve()
    # If script is in xtructure_benchmarks/, parent is repo root
    # But we want to find results folders which are usually in xtructure_benchmarks/
    # Let's verify where results are.

    # Results are expected in xtructure_benchmarks/results_{branch}
    # If running from repo root, it is xtructure_benchmarks/results_{branch}
    # If running from script dir, it is results_{branch}

    # Let's try to locate the results relatively
    base_dir_name = f"results_{base_branch}"
    target_dir_name = f"results_{target_branch}"

    # Check possible locations
    candidates = [
        script_dir / base_dir_name,
        script_dir.parent / "xtructure_benchmarks" / base_dir_name,
        Path.cwd() / "xtructure_benchmarks" / base_dir_name,
    ]

    base_dir = None
    for c in candidates:
        if c.exists():
            base_dir = c
            break

    if not base_dir:
        CONSOLE.print(f"[bold red]Could not find results for branch '{base_branch}'[/]")
        CONSOLE.print("Checked locations:")
        for c in candidates:
            CONSOLE.print(f" - {c}")
        return

    # Try to find target dir in same parent as found base_dir
    target_dir = base_dir.parent / target_dir_name

    if not target_dir.exists():
        CONSOLE.print(
            f"[bold red]Could not find results for branch '{target_branch}'[/] at {target_dir}"
        )
        return

    CONSOLE.print(
        f"[bold cyan]Comparing {base_branch} (Base) vs {target_branch} (Target)[/]"
    )

    files = sorted([f.name for f in base_dir.glob("*_results.json")])

    for filename in files:
        base_path = base_dir / filename
        target_path = target_dir / filename

        base_data = load_result(base_path)
        target_data = load_result(target_path)

        if not target_data:
            CONSOLE.print(f"[bold red]Missing {target_branch} result for {filename}[/]")
            continue

        title = filename.replace("_results.json", "").capitalize()
        table = Table(
            title=f"Benchmark Comparison: {title} ({base_branch} vs {target_branch})"
        )
        table.add_column("Operation", style="cyan")
        table.add_column("Batch Size", justify="right")
        table.add_column(f"{base_branch} Ops/s", justify="right", style="red")
        table.add_column(f"{target_branch} Ops/s", justify="right", style="green")
        table.add_column("Speedup", justify="right", style="bold yellow")

        try:
            tasks = list(base_data["xtructure"].keys())
            batch_sizes = base_data["batch_sizes"]
        except KeyError:
            CONSOLE.print(f"[yellow]Skipping malformed file: {filename}[/]")
            continue

        for task in tasks:
            task_name = task.replace("_ops_per_sec", "")

            base_vals = base_data["xtructure"][task]
            target_vals = target_data["xtructure"].get(task)

            if not target_vals:
                continue

            for i, size in enumerate(batch_sizes):
                if i >= len(base_vals) or i >= len(target_vals):
                    break

                b_val = base_vals[i]
                t_val = target_vals[i]

                b_median = b_val["median"] if isinstance(b_val, dict) else b_val
                t_median = t_val["median"] if isinstance(t_val, dict) else t_val

                if b_median == 0:
                    speedup = "Inf"
                else:
                    ratio = t_median / b_median
                    color = (
                        "green"
                        if ratio >= 1.05
                        else "red"
                        if ratio <= 0.95
                        else "white"
                    )
                    speedup = f"[{color}]{ratio:.2f}x[/{color}]"

                table.add_row(
                    task_name if i == 0 else "",
                    str(size),
                    format_perf(b_median),
                    format_perf(t_median),
                    speedup,
                )
            table.add_section()

        CONSOLE.print(table)
        CONSOLE.print("\n")


if __name__ == "__main__":
    main()
