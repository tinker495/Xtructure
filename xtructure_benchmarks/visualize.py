import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


def _params_summary(params: dict) -> str:
    if not params:
        return "-"
    return ", ".join(f"{k}={params[k]}" for k in sorted(params))


def _plot_new_schema(data: dict, path: Path) -> list[Path]:
    out = []
    records = data.get("records", [])
    if not records:
        return out
    for idx, rec in enumerate(records):
        metrics = rec.get("metrics", {})
        plt.figure(figsize=(8, 4))
        plt.bar(
            ["step_time_ms", "items_per_sec"],
            [
                metrics.get("step_time_ms_median", 0),
                metrics.get("items_per_sec_median", 0),
            ],
        )
        plt.title(f"{rec.get('name', 'workload')} ({path.stem})")
        plt.tight_layout()
        output = path.parent / f"{path.stem}_record{idx}.png"
        plt.savefig(output)
        plt.close()
        out.append(output)
    return out


def _plot_legacy_schema(data: dict, path: Path) -> list[Path]:
    out = []
    if "batch_sizes" not in data or "xtructure" not in data:
        return out
    batch_sizes = data["batch_sizes"]
    for op, vals in data["xtructure"].items():
        y = [v["median"] if isinstance(v, dict) else v for v in vals]
        plt.figure(figsize=(8, 4))
        plt.plot(batch_sizes, y, marker="o")
        plt.xscale("log", base=2)
        plt.yscale("log")
        plt.title(f"{path.stem}: {op}")
        plt.tight_layout()
        output = path.parent / f"{path.stem}_{op}.png"
        plt.savefig(output)
        plt.close()
        out.append(output)
    return out


def visualize_all() -> None:
    results_dir = Path(__file__).parent / "results"
    report_lines = ["# Benchmark Report", ""]
    for result_file in sorted(results_dir.glob("*_results.json")):
        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "schema_version" in data:
            images = _plot_new_schema(data, result_file)
            report_lines.append(f"## {result_file.name}")
            report_lines.append(f"- schema_version: {data.get('schema_version')}")
            for rec in data.get("records", []):
                params = rec.get("params", {})
                report_lines.append(
                    f"- {rec['name']}: step_time_ms={rec['metrics'].get('step_time_ms_median', 0):.3f}, items_per_sec={rec['metrics'].get('items_per_sec_median', 0):.3f}"
                )
                report_lines.append(f"- params: {_params_summary(params)}")
            for image in images:
                report_lines.append(f"![{image.name}]({image.name})")
            report_lines.append("")
        else:
            _plot_legacy_schema(data, result_file)
    report_path = results_dir / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    visualize_all()
