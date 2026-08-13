"""Command-line experiment runner: ``python -m wam.experiment``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .predictor import LastTransitionPredictor, NextLinePredictor, WeightedTriePredictor
from .simulator import SimulatorConfig, SimulationResult, simulate
from .workloads import all_workloads, to_byte_addresses


def _split(trace: list[int], fraction: float = 0.6) -> tuple[list[int], list[int]]:
    split = max(1, min(len(trace) - 1, int(len(trace) * fraction)))
    return trace[:split], trace[split:]


def run_workload(name: str, trace: list[int], config: SimulatorConfig | None = None) -> list[dict]:
    config = config or SimulatorConfig()
    training, test = _split(trace)
    predictors = [
        ("None", None, False),
        ("NextLine", NextLinePredictor(), True),
        ("Markov-1", LastTransitionPredictor(), True),
        ("WeightedTrie", WeightedTriePredictor(context_depth=2, threshold=0.05), True),
    ]
    rows: list[dict] = []
    baseline: SimulationResult | None = None
    for label, predictor, enable_prefetch in predictors:
        if predictor is not None:
            predictor.fit(training)
        result = simulate(to_byte_addresses(test, config.cache_line_size), predictor, config, enable_prefetch=enable_prefetch, initial_context=training[-getattr(predictor, "context_depth", 1) :] if predictor is not None else ())
        if baseline is None:
            baseline = result
        metrics = result.metrics
        rows.append({
            "workload": name,
            "predictor": label,
            "result": result,
            "l1_hit_rate": metrics.l1_hit_rate,
            "l2_hit_rate": metrics.l2_hit_rate,
            "latency": metrics.average_access_latency,
            "accuracy": metrics.top1_accuracy,
            "topk_accuracy": metrics.topk_accuracy,
            "speedup": 1.0 if baseline is result else baseline.cycles / result.cycles,
            "precision": metrics.prefetch_precision,
            "coverage": metrics.prefetch_coverage,
        })
    return rows


def context_depth_rows(workloads: dict[str, list[int]], depths: tuple[int, ...] = (1, 2, 3, 4)) -> list[dict]:
    rows: list[dict] = []
    config = SimulatorConfig()
    for workload, trace in workloads.items():
        training, test = _split(trace)
        for depth in depths:
            predictor = WeightedTriePredictor(context_depth=depth, threshold=0.05).fit(training)
            result = simulate(to_byte_addresses(test, config.cache_line_size), predictor, config, initial_context=training[-depth:])
            rows.append({"workload": workload, "depth": depth, "accuracy": result.metrics.top1_accuracy, "latency": result.metrics.average_access_latency, "storage_bytes": result.predictor_storage["estimated_bytes"]})
    return rows


def threshold_rows(workloads: dict[str, list[int]], thresholds: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 0.9)) -> list[dict]:
    rows: list[dict] = []
    config = SimulatorConfig()
    for workload, trace in workloads.items():
        training, test = _split(trace)
        for threshold in thresholds:
            predictor = WeightedTriePredictor(context_depth=2, threshold=threshold).fit(training)
            result = simulate(to_byte_addresses(test, config.cache_line_size), predictor, config, initial_context=training[-2:])
            rows.append({"workload": workload, "threshold": threshold, "latency": result.metrics.average_access_latency})
    return rows


def print_table(rows: list[dict]) -> None:
    header = f"{'Workload':<12} {'Predictor':<14} {'L1 hit':>8} {'L2 hit':>8} {'Avg cyc':>9} {'Top-1':>8} {'Speedup':>9} {'Prec.':>8} {'Cover.':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['workload']:<12} {row['predictor']:<14} {row['l1_hit_rate']:>7.1%} {row['l2_hit_rate']:>7.1%} {row['latency']:>9.2f} {row['accuracy']:>7.1%} {row['speedup']:>8.2f}x {row['precision']:>7.1%} {row['coverage']:>7.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=512, help="number of accesses per workload")
    parser.add_argument("--plot-dir", type=Path, help="write the four matplotlib plots to this directory")
    args = parser.parse_args()
    workloads = all_workloads(args.length)
    rows = [row for name, trace in workloads.items() for row in run_workload(name, trace)]
    print_table(rows)
    if args.plot_dir:
        from .visualization import plot_accuracy_vs_depth, plot_latency_vs_threshold, plot_precision_coverage, plot_speedup

        args.plot_dir.mkdir(parents=True, exist_ok=True)
        plot_speedup(rows, args.plot_dir / "speedup_by_workload.png")
        plot_accuracy_vs_depth(context_depth_rows(workloads), args.plot_dir / "accuracy_vs_context_depth.png")
        plot_latency_vs_threshold(threshold_rows(workloads), args.plot_dir / "latency_vs_threshold.png")
        plot_precision_coverage(rows, args.plot_dir / "prefetch_precision_vs_coverage.png")
        print(f"\nPlots written to {args.plot_dir}")


if __name__ == "__main__":
    main()
