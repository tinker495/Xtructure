import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

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

        op_name = op.replace("_ops_per_sec", "").capitalize()
        title = f"{data_structure_name} {op_name} Performance"

        plt.plot(batch_sizes, xtructure_data[op], "o-", label=f"xtructure.{data_structure_name}")
        plt.plot(batch_sizes, python_data[op], "s--", label="Python Baseline")

        plt.xscale("log", base=2)
        plt.yscale("log")

        plt.xlabel("Batch Size (log scale)")
        plt.ylabel("Operations per Second (log scale)")
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
