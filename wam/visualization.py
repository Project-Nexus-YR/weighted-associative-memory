"""Optional matplotlib plots for experiment results."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _plt():
    import matplotlib

    # Experiments are batch jobs; never require a desktop display.
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def plot_speedup(rows: Iterable[dict], output_path: str | Path) -> None:
    plt = _plt()
    rows = list(rows)
    labels = [f"{row['workload']}\n{row['predictor']}" for row in rows]
    values = [row["speedup"] for row in rows]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(labels, values)
    ax.set_ylabel("Speedup vs no prefetch")
    ax.set_title("Weighted Associative Memory experiment speedup")
    ax.axhline(1.0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_accuracy_vs_depth(rows: Iterable[dict], output_path: str | Path) -> None:
    plt = _plt()
    rows = list(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    for workload in sorted({row["workload"] for row in rows}):
        subset = [row for row in rows if row["workload"] == workload]
        subset.sort(key=lambda row: row["depth"])
        ax.plot([row["depth"] for row in subset], [row["accuracy"] for row in subset], marker="o", label=workload)
    ax.set_xlabel("Context depth")
    ax.set_ylabel("Top-1 prediction accuracy")
    ax.set_title("Accuracy vs context depth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_latency_vs_threshold(rows: Iterable[dict], output_path: str | Path) -> None:
    plt = _plt()
    rows = list(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    for workload in sorted({row["workload"] for row in rows}):
        subset = [row for row in rows if row["workload"] == workload]
        subset.sort(key=lambda row: row["threshold"])
        ax.plot([row["threshold"] for row in subset], [row["latency"] for row in subset], marker="o", label=workload)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Average access latency (cycles)")
    ax.set_title("Latency vs prediction confidence threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_precision_coverage(rows: Iterable[dict], output_path: str | Path) -> None:
    plt = _plt()
    rows = list(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    for row in rows:
        ax.scatter(row["coverage"], row["precision"], label=f"{row['workload']} / {row['predictor']}")
    ax.set_xlabel("Prefetch coverage")
    ax.set_ylabel("Prefetch precision")
    ax.set_title("Prefetch precision vs coverage")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
