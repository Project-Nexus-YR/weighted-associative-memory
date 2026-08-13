"""Simple publication-style matplotlib figures for benchmark artifacts."""

from __future__ import annotations

from collections import defaultdict
import os
import tempfile
from pathlib import Path
from typing import Iterable


def _plt():
    cache_dir = Path(tempfile.gettempdir()) / "wam-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    _plt().close(fig)


def _bar(summary: list[dict[str, object]], field: str, title: str, ylabel: str, path: Path) -> None:
    plt = _plt()
    workloads = sorted({str(row["workload"]) for row in summary})
    predictors = sorted({str(row["predictor"]) for row in summary})
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.8 / max(1, len(predictors))
    for index, predictor in enumerate(predictors):
        values = []
        for workload in workloads:
            row = next((item for item in summary if item["workload"] == workload and item["predictor"] == predictor), None)
            values.append(float(row.get(field, 0.0)) if row else 0.0)
        positions = [i + index * width for i in range(len(workloads))]
        ax.bar(positions, values, width, label=predictor)
    ax.set_xticks([i + width * max(0, len(predictors) - 1) / 2 for i in range(len(workloads))], workloads, rotation=25, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize="small", ncol=2)
    _save(fig, path)


def _wam_line(summary: list[dict[str, object]], field: str, title: str, ylabel: str, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 5))
    workloads = sorted({str(row["workload"]) for row in summary if str(row["predictor"]).startswith("WAM")})
    for workload in workloads:
        rows = [row for row in summary if row["workload"] == workload and str(row["predictor"]).startswith("WAM")]
        rows.sort(key=lambda row: int(str(row["predictor"]).split("=")[-1]))
        ax.plot([int(str(row["predictor"]).split("=")[-1]) for row in rows], [float(row.get(field, 0.0)) for row in rows], marker="o", label=workload)
    ax.set_xlabel("Context depth")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize="small")
    _save(fig, path)


def generate_benchmark_plots(summary: list[dict[str, object]], detailed: list[dict[str, object]], learning: list[dict[str, object]], break_even: list[dict[str, object]], output_dir: str | Path) -> None:
    del detailed
    output = Path(output_dir)
    _bar(summary, "mean_speedup_over_none", "Speedup by workload and predictor", "Speedup over no prefetch", output / "speedup_by_workload_predictor.png")
    _bar(summary, "mean_average_access_latency", "Average effective latency by workload and predictor", "Cycles/access", output / "average_latency_by_workload_predictor.png")
    _wam_line(summary, "mean_top1_accuracy", "Prediction accuracy vs context depth", "Top-1 accuracy", output / "prediction_accuracy_vs_context_depth.png")
    _wam_line(summary, "mean_speedup_over_none", "Speedup vs context depth", "Speedup", output / "speedup_vs_context_depth.png")
    _wam_line(summary, "mean_estimated_bytes", "Predictor storage vs context depth", "Estimated bytes", output / "predictor_storage_vs_context_depth.png")

    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in summary:
        if str(row["predictor"]).startswith("WAM"):
            ax.scatter(float(row["mean_estimated_bytes"]), float(row["mean_speedup_over_none"]), label=f"{row['workload']} / {row['predictor']}")
    ax.set_xlabel("Estimated predictor bytes")
    ax.set_ylabel("Speedup over no prefetch")
    ax.set_title("Speedup vs predictor storage")
    ax.legend(fontsize="x-small")
    _save(fig, output / "speedup_vs_predictor_storage.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for row in summary:
        if str(row["predictor"]).startswith("WAM"):
            ax.scatter(float(row["mean_prefetch_coverage"]), float(row["mean_prefetch_precision"]), label=f"{row['workload']} / {row['predictor']}")
    ax.set_xlabel("Prefetch coverage")
    ax.set_ylabel("Prefetch precision")
    ax.set_title("Prefetch precision vs coverage")
    ax.legend(fontsize="x-small")
    _save(fig, output / "prefetch_precision_vs_coverage.png")

    if learning:
        accesses = [int(row["accesses_observed"]) for row in learning]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(accesses, [float(row["top1_accuracy"]) for row in learning], marker="o")
        ax.set_xlabel("Number of accesses observed")
        ax.set_ylabel("Top-1 accuracy")
        ax.set_title("Online learning curve: accuracy")
        _save(fig, output / "accuracy_vs_accesses_observed.png")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(accesses, [float(row["average_access_latency"]) for row in learning], marker="o", label="effective latency")
        ax.set_xlabel("Number of accesses observed")
        ax.set_ylabel("Cycles/access")
        ax.set_title("Online learning curve: latency")
        _save(fig, output / "latency_vs_accesses_observed.png")

    thresholds: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in break_even:
        thresholds[int(row["dram_latency"])].append(row)
    fig, ax = plt.subplots(figsize=(8, 5))
    for dram, rows in sorted(thresholds.items()):
        rows.sort(key=lambda row: float(row["threshold"]))
        ax.plot([float(row["threshold"]) for row in rows], [float(row["speedup"]) for row in rows], marker="o", label=f"DRAM {dram} cycles")
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup vs confidence threshold")
    ax.legend()
    _save(fig, output / "speedup_vs_confidence_threshold.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for dram, rows in sorted(thresholds.items()):
        ax.plot([float(row["accuracy"]) for row in rows], [float(row["speedup"]) for row in rows], marker="o", label=f"DRAM {dram} cycles")
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xlabel("Prediction accuracy")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup vs prediction accuracy")
    ax.legend()
    _save(fig, output / "speedup_vs_prediction_accuracy.png")


# Compatibility helpers retained for the original MVP API.
def plot_speedup(rows: Iterable[dict], output_path: str | Path) -> None:
    rows = list(rows)
    _bar(rows, "speedup", "Experiment speedup", "Speedup", Path(output_path))


def plot_accuracy_vs_depth(rows: Iterable[dict], output_path: str | Path) -> None:
    rows = list(rows)
    _wam_line(rows, "accuracy", "Accuracy vs context depth", "Top-1 accuracy", Path(output_path))


def plot_latency_vs_threshold(rows: Iterable[dict], output_path: str | Path) -> None:
    rows = list(rows)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    for workload in sorted({row["workload"] for row in rows}):
        subset = sorted((row for row in rows if row["workload"] == workload), key=lambda row: row["threshold"])
        ax.plot([row["threshold"] for row in subset], [row["latency"] for row in subset], marker="o", label=workload)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Average latency")
    ax.legend(fontsize="small")
    _save(fig, Path(output_path))


def plot_precision_coverage(rows: Iterable[dict], output_path: str | Path) -> None:
    rows = list(rows)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    for row in rows:
        ax.scatter(row["coverage"], row["precision"], label=f"{row['workload']} / {row['predictor']}")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Precision")
    ax.legend(fontsize="small")
    _save(fig, Path(output_path))
