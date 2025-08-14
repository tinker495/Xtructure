import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from xtructure_benchmarks.common import human_format

matplotlib.use("Agg")


def plot_performance(results_path: Path):
    """
    Loads benchmark results from a JSON file and plots the performance comparison.
    """
    with open(results_path, "r") as f:
        data = json.load(f)

    batch_sizes = data["batch_sizes"]
    xtructure_data = data["xtructure"]
    python_data = data["python"]

    data_structure_name = results_path.stem.split("_")[0].capitalize()

    # Determine the operations (e.g., insert, lookup, push, pop)
    operations = list(xtructure_data.keys())

    for op in operations:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()  # Get current axes

        op_name = op.replace("_ops_per_sec", "").capitalize()
        title = f"{data_structure_name} {op_name} Performance"

        # Extract median values and error bars from new format
        xtructure_values = []
        xtructure_errors = []
        python_values = []
        python_errors = []

        for result in xtructure_data[op]:
            if isinstance(result, dict):
                xtructure_values.append(result["median"])
                xtructure_errors.append(result["iqr"] / 2)  # Half IQR for error bars
            else:
                # Backward compatibility
                xtructure_values.append(result)
                xtructure_errors.append(0)

        for result in python_data[op]:
            if isinstance(result, dict):
                python_values.append(result["median"])
                python_errors.append(result["iqr"] / 2)  # Half IQR for error bars
            else:
                # Backward compatibility
                python_values.append(result)
                python_errors.append(0)

        plt.errorbar(
            batch_sizes,
            xtructure_values,
            yerr=xtructure_errors,
            fmt="o-",
            label=f"xtructure.{data_structure_name}",
            capsize=5,
        )
        plt.errorbar(
            batch_sizes,
            python_values,
            yerr=python_errors,
            fmt="s--",
            label="Python Baseline",
            capsize=5,
        )

        plt.xscale("log", base=2)
        plt.yscale("log")

        # Apply human-readable format to the y-axis
        ax.yaxis.set_major_formatter(FuncFormatter(human_format))

        plt.xlabel("Batch Size (log scale)")
        plt.ylabel("Operations per Second")
        plt.title(title)
        plt.legend()
        plt.grid(True, which="both", ls="--")

        # Save the plot
        output_filename = (
            results_path.parent / f"{data_structure_name.lower()}_{op_name.lower()}_performance.png"
        )
        plt.savefig(output_filename)
        print(f"Saved plot to {output_filename}")
        plt.close()


def visualize_all():
    """
    Finds all benchmark result JSON files and generates plots for them.
    """
    results_dir = Path(__file__).parent / "results"

    for results_file in results_dir.glob("*_results.json"):
        print(f"Processing {results_file}...")
        plot_performance(results_file)


if __name__ == "__main__":
    visualize_all()
