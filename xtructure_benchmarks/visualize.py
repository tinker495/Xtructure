import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def _params_summary(params: dict) -> str:
    if not params:
        return "-"
    return ", ".join(f"{k}={params[k]}" for k in sorted(params))


def _param_space_summary(records: list[dict]) -> str:
    values: dict[str, set] = defaultdict(set)
    for rec in records:
        for key, value in rec.get("params", {}).items():
            values[key].add(value)
    if not values:
        return "-"

    parts: list[str] = []
    for key in sorted(values):
        vals = sorted(values[key])
        if len(vals) == 1:
            parts.append(f"{key}={vals[0]}")
        elif len(vals) <= 6:
            joined = ", ".join(str(v) for v in vals)
            parts.append(f"{key}=[{joined}]")
        else:
            parts.append(f"{key}[{len(vals)}]={vals[0]}..{vals[-1]}")
    return "; ".join(parts)


def _select_metric_key(records: list[dict]) -> str:
    preferred = (
        "accepted_per_sec_median",
        "processed_per_sec_median",
        "items_per_sec_median",
    )
    for key in preferred:
        if any(key in rec.get("metrics", {}) for rec in records):
            return key
    return "items_per_sec_median"


def _plot_heatmap(
    records: list[dict],
    path: Path,
    *,
    x_param: str,
    y_param: str,
    metric_key: str,
) -> Path | None:
    grouped: dict[tuple[float, float], list[float]] = defaultdict(list)
    for rec in records:
        params = rec.get("params", {})
        if x_param not in params or y_param not in params:
            continue
        metric = rec.get("metrics", {}).get(metric_key)
        if metric is None:
            continue
        grouped[(float(params[x_param]), float(params[y_param]))].append(float(metric))

    if not grouped:
        return None

    x_vals = sorted({xy[0] for xy in grouped})
    y_vals = sorted({xy[1] for xy in grouped})
    if len(x_vals) < 2 or len(y_vals) < 2:
        return None

    x_index = {v: i for i, v in enumerate(x_vals)}
    y_index = {v: i for i, v in enumerate(y_vals)}
    matrix = np.full((len(y_vals), len(x_vals)), np.nan, dtype=np.float64)

    for (xv, yv), vals in grouped.items():
        matrix[y_index[yv], x_index[xv]] = float(np.mean(vals))

    plt.figure(figsize=(7, 5))
    im = plt.imshow(matrix, origin="lower", aspect="auto")
    plt.colorbar(im, label=metric_key)
    plt.xticks(
        range(len(x_vals)),
        [str(v) for v in x_vals],
        rotation=45,
        ha="right",
    )
    plt.yticks(range(len(y_vals)), [str(v) for v in y_vals])
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.title(f"{path.stem}: {y_param} x {x_param}")
    plt.tight_layout()
    output = path.parent / f"{path.stem}_{y_param}_x_{x_param}_heatmap.png"
    plt.savefig(output)
    plt.close()
    return output


def _plot_new_schema(data: dict, path: Path) -> list[Path]:
    out = []
    records = data.get("records", [])
    if not records:
        return out

    metric_key = _select_metric_key(records)
    for idx, rec in enumerate(records):
        metrics = rec.get("metrics", {})
        labels = [
            "step_time_ms",
            "items_per_sec",
            "processed_per_sec",
            "accepted_per_sec",
        ]
        values = [
            metrics.get("step_time_ms_median", 0.0),
            metrics.get("items_per_sec_median", 0.0),
            metrics.get("processed_per_sec_median", 0.0),
            metrics.get("accepted_per_sec_median", 0.0),
        ]
        plt.figure(figsize=(8, 4))
        plt.bar(labels, values)
        plt.title(f"{rec.get('name', 'workload')} ({path.stem})")
        plt.tight_layout()
        output = path.parent / f"{path.stem}_record{idx}.png"
        plt.savefig(output)
        plt.close()
        out.append(output)

    for y_param, x_param in (("hit_ratio", "dup_ratio"), ("occupancy", "batch_size")):
        heatmap = _plot_heatmap(
            records,
            path,
            x_param=x_param,
            y_param=y_param,
            metric_key=metric_key,
        )
        if heatmap is not None:
            out.append(heatmap)
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
            records = data.get("records", [])
            images = _plot_new_schema(data, result_file)
            report_lines.append(f"## {result_file.name}")
            report_lines.append(f"- schema_version: {data.get('schema_version')}")
            report_lines.append(f"- record_count: {len(records)}")
            report_lines.append(f"- param_space: {_param_space_summary(records)}")
            metric_key = _select_metric_key(records)
            if records:
                best = max(
                    records,
                    key=lambda r: float(r.get("metrics", {}).get(metric_key, 0.0)),
                )
                report_lines.append(
                    f"- best_{metric_key}: {best.get('metrics', {}).get(metric_key, 0.0):.3f} ({_params_summary(best.get('params', {}))})"
                )

            listed = records[:12]
            for rec in listed:
                params = rec.get("params", {})
                report_lines.append(
                    f"- {rec['name']}: step_time_ms={rec['metrics'].get('step_time_ms_median', 0):.3f}, items_per_sec={rec['metrics'].get('items_per_sec_median', 0):.3f}, processed_per_sec={rec['metrics'].get('processed_per_sec_median', 0):.3f}, accepted_per_sec={rec['metrics'].get('accepted_per_sec_median', 0):.3f}"
                )
                report_lines.append(f"- params: {_params_summary(params)}")
            if len(records) > len(listed):
                report_lines.append(
                    f"- ... {len(records) - len(listed)} additional records omitted in markdown summary"
                )
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
