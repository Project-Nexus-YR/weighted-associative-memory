"""Primary comparative benchmark and automatic research report generator.

Run with ``python -m wam.benchmark``. The benchmark uses chronological
70/30 train/test splits by default; no predictor is allowed to learn from the
held-out suffix. Synthetic workload functions return cache-line IDs and are
converted to raw byte addresses before simulation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, Iterable

from .hierarchy import HierarchyConfig
from .predictor import LastTransitionPredictor, NextLinePredictor, Predictor, StridePredictor, WeightedTriePredictor
from .simulator import SimulationResult, SimulatorConfig, simulate
from .traces import load_trace, normalize_addresses
from .workloads import all_workloads, to_byte_addresses


DEFAULT_DEPTHS = (1, 2, 3, 4, 8)
SYSTEM_NAMES = ("None", "NextLine", "Stride", "Markov-1") + tuple(f"WAM depth={depth}" for depth in DEFAULT_DEPTHS)


def default_simulator_config(dram_latency: int = 150, cache_line_size: int = 64) -> SimulatorConfig:
    return SimulatorConfig(
        hierarchy=HierarchyConfig(
            l1_capacity=64,
            l2_capacity=256,
            l3_capacity=1024,
            l1_latency=4,
            l2_latency=12,
            l3_latency=40,
            dram_latency=dram_latency,
            cache_line_size=cache_line_size,
        ),
        prefetch_issue_cost=1,
        prefetch_destination="L1",
        top_k=3,
        max_outstanding_prefetches=8,
    )


def make_predictor(name: str, threshold: float = 0.05, alpha: float = 0.25, top_k: int = 3) -> Predictor | None:
    del top_k  # top-K is a simulator parameter, not predictor state.
    if name == "None":
        return None
    if name == "NextLine":
        return NextLinePredictor()
    if name == "Stride":
        return StridePredictor(confidence_threshold=2)
    if name == "Markov-1":
        return LastTransitionPredictor(threshold=threshold)
    match = re.fullmatch(r"WAM depth=(\d+)", name)
    if match:
        return WeightedTriePredictor(context_depth=int(match.group(1)), threshold=threshold)
    if name == "WAM EMA depth=2":
        return WeightedTriePredictor(context_depth=2, strategy="ema", alpha=alpha, threshold=threshold)
    raise ValueError(f"unknown system: {name}")


def _split(trace: list[int], train_fraction: float) -> tuple[list[int], list[int]]:
    split = max(1, min(len(trace) - 1, int(len(trace) * train_fraction)))
    return trace[:split], trace[split:]


def result_row(
    workload: str,
    predictor_name: str,
    trial: int,
    result: SimulationResult,
    baseline_cycles: int,
    next_line_cycles: int,
    stride_cycles: int,
    markov_cycles: int,
    mode: str = "train_test",
    threshold: float = 0.05,
    alpha: float = 0.25,
    top_k: int = 3,
) -> dict[str, object]:
    m = result.metrics
    storage = result.predictor_storage
    return {
        "workload": workload,
        "predictor": predictor_name,
        "trial": trial,
        "mode": mode,
        "threshold": threshold,
        "alpha": alpha,
        "top_k": top_k,
        "total_accesses": m.total_accesses,
        "l1_hit_rate": m.l1_hit_rate,
        "l2_hit_rate": m.l2_hit_rate,
        "l3_hit_rate": m.l3_hit_rate,
        "dram_access_rate": m.dram_access_rate,
        "total_cycles": m.cycles,
        "raw_memory_cycles": m.raw_memory_cycles,
        "average_memory_latency": m.average_memory_latency,
        "average_access_latency": m.average_access_latency,
        "top1_accuracy": m.top1_accuracy,
        "topk_accuracy": m.topk_accuracy,
        "prediction_attempts": m.prediction_attempts,
        "prefetch_requests": m.prefetch_requests,
        "prefetches_issued": m.prefetches_issued,
        "prefetches_completed": m.prefetches_completed,
        "dropped_prefetches": m.dropped_prefetches,
        "useful_prefetches": m.useful_prefetches,
        "late_prefetches": m.late_prefetches,
        "unused_prefetches": m.unused_prefetches,
        "duplicate_prefetches": m.duplicate_prefetches,
        "prefetch_precision": m.prefetch_precision,
        "prefetch_coverage": m.prefetch_coverage,
        "bandwidth_bytes": m.bandwidth_bytes,
        "pollution_misses": m.pollution_misses,
        "cache_evictions_caused_by_prefetching": m.cache_evictions_caused_by_prefetching,
        "predictor_lookup_overhead": m.predictor_lookup_overhead,
        "predictor_update_overhead": m.predictor_update_overhead,
        "prefetch_overhead": m.prefetch_overhead,
        "latency_saved_by_useful_prefetches": m.latency_saved_by_useful_prefetches,
        "net_latency_benefit": m.net_latency_benefit,
        "context_reuse_ratio": m.context_reuse_ratio,
        "fallback_count": m.fallback_count,
        "unseen_context_count": m.unseen_context_count,
        "mean_context_observations": m.mean_context_observations,
        "mean_context_entropy": m.mean_context_entropy,
        "entries": storage.get("entries", 0),
        "nodes": storage.get("nodes", 0),
        "edges": storage.get("edges", 0),
        "counters": storage.get("counters", 0),
        "weights": storage.get("weights", 0),
        "estimated_bytes": storage.get("estimated_bytes", 0),
        "bytes_per_transition": storage.get("estimated_bytes", 0) / storage.get("entries", 1) if storage.get("entries", 0) else 0.0,
        "speedup_over_none": baseline_cycles / m.cycles if m.cycles else 0.0,
        "speedup_over_nextline": next_line_cycles / m.cycles if m.cycles else 0.0,
        "speedup_over_stride": stride_cycles / m.cycles if m.cycles else 0.0,
        "speedup_over_markov": markov_cycles / m.cycles if m.cycles else 0.0,
    }


def evaluate_trace(
    workload: str,
    line_trace: list[int],
    trial: int,
    config: SimulatorConfig,
    train_fraction: float = 0.7,
    system_names: Iterable[str] = SYSTEM_NAMES,
    threshold: float = 0.05,
    alpha: float = 0.25,
    top_k: int = 3,
) -> list[dict[str, object]]:
    train, evaluation = _split(line_trace, train_fraction)
    raw_evaluation = to_byte_addresses(evaluation, config.cache_line_size)
    rows_by_name: dict[str, dict[str, object]] = {}
    results: dict[str, SimulationResult] = {}
    for name in system_names:
        predictor = make_predictor(name, threshold=threshold, alpha=alpha, top_k=top_k)
        if predictor is not None:
            predictor.fit(train)
        result = simulate(
            raw_evaluation,
            predictor,
            config,
            enable_prefetch=name != "None",
            learning=False,
            initial_context=train[-getattr(predictor, "context_depth", 1) :] if predictor is not None else (),
        )
        results[name] = result

    baseline_cycles = results["None"].cycles
    next_line_cycles = results.get("NextLine", results["None"]).cycles
    stride_cycles = results.get("Stride", results["None"]).cycles
    markov_cycles = results.get("Markov-1", results["None"]).cycles
    for name, result in results.items():
        rows_by_name[name] = result_row(workload, name, trial, result, baseline_cycles, next_line_cycles, stride_cycles, markov_cycles, threshold=threshold, alpha=alpha, top_k=top_k)
    return list(rows_by_name.values())


def _mean_std(values: list[float]) -> tuple[float, float]:
    return (statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["workload"]), str(row["predictor"])), []).append(row)
    fields = [
        "speedup_over_none", "speedup_over_nextline", "speedup_over_stride", "speedup_over_markov",
        "top1_accuracy", "topk_accuracy", "average_memory_latency", "average_access_latency",
        "prefetch_precision", "prefetch_coverage", "estimated_bytes", "bytes_per_transition", "context_reuse_ratio", "mean_context_observations", "mean_context_entropy", "predictor_lookup_overhead",
    ]
    result: list[dict[str, object]] = []
    for (workload, predictor), group in sorted(groups.items()):
        item: dict[str, object] = {"workload": workload, "predictor": predictor, "trials": len(group)}
        for field in fields:
            mean, std = _mean_std([float(row[field]) for row in group])
            item[f"mean_{field}"] = mean
            item[f"std_{field}"] = std
        result.append(item)
    return result


def online_learning_curve(line_trace: list[int], config: SimulatorConfig, workload: str = "PhaseChanging", points: int = 10) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for point in range(1, points + 1):
        count = max(2, int(len(line_trace) * point / points))
        prefix = line_trace[:count]
        predictor = WeightedTriePredictor(context_depth=2, threshold=0.05)
        raw = to_byte_addresses(prefix, config.cache_line_size)
        result = simulate(raw, predictor, config, learning=True)
        baseline = simulate(raw, None, config, enable_prefetch=False)
        rows.append({
            "workload": workload,
            "accesses_observed": count,
            "top1_accuracy": result.metrics.top1_accuracy,
            "average_memory_latency": result.metrics.average_memory_latency,
            "average_access_latency": result.metrics.average_access_latency,
            "speedup_over_none": baseline.cycles / result.cycles if result.cycles else 0.0,
            "prefetch_precision": result.metrics.prefetch_precision,
        })
    return rows


def break_even_rows(line_trace: list[int], top_k: int = 3) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dram_latency in (150, 300):
        config = default_simulator_config(dram_latency=dram_latency)
        train, evaluation = _split(line_trace, 0.7)
        baseline = simulate(to_byte_addresses(evaluation, 64), None, config, enable_prefetch=False)
        for threshold in (0.0, 0.25, 0.5, 0.65, 0.8, 0.9):
            predictor = WeightedTriePredictor(context_depth=2, threshold=threshold).fit(train)
            result = simulate(to_byte_addresses(evaluation, 64), predictor, config, initial_context=train[-2:], learning=False)
            rows.append({
                "dram_latency": dram_latency,
                "threshold": threshold,
                "accuracy": result.metrics.top1_accuracy,
                "speedup": baseline.cycles / result.cycles if result.cycles else 0.0,
                "average_access_latency": result.metrics.average_access_latency,
                "prefetch_precision": result.metrics.prefetch_precision,
            })
    return rows


def sweep_rows(line_trace: list[int], config: SimulatorConfig) -> list[dict[str, object]]:
    train, evaluation = _split(line_trace, 0.7)
    baseline = simulate(to_byte_addresses(evaluation, config.cache_line_size), None, config, enable_prefetch=False)
    rows: list[dict[str, object]] = []
    for depth in DEFAULT_DEPTHS:
        for threshold in (0.5, 0.65, 0.8, 0.9):
            predictor = WeightedTriePredictor(context_depth=depth, threshold=threshold).fit(train)
            result = simulate(to_byte_addresses(evaluation, config.cache_line_size), predictor, config, initial_context=train[-depth:])
            rows.append({"family": "depth_threshold", "depth": depth, "threshold": threshold, "alpha": 0.25, "top_k": config.top_k, "destination": config.prefetch_destination, "cache_capacity": config.hierarchy.l1_capacity, "speedup": baseline.cycles / result.cycles, "accuracy": result.metrics.top1_accuracy, "storage_bytes": result.predictor_storage["estimated_bytes"], "latency": result.metrics.average_access_latency})
    for alpha in (0.01, 0.05, 0.1, 0.25):
        predictor = WeightedTriePredictor(context_depth=2, strategy="ema", alpha=alpha, threshold=0.5).fit(train)
        result = simulate(to_byte_addresses(evaluation, config.cache_line_size), predictor, config, initial_context=train[-2:])
        rows.append({"family": "ema_alpha", "depth": 2, "threshold": 0.5, "alpha": alpha, "top_k": config.top_k, "destination": config.prefetch_destination, "cache_capacity": config.hierarchy.l1_capacity, "speedup": baseline.cycles / result.cycles, "accuracy": result.metrics.top1_accuracy, "storage_bytes": result.predictor_storage["estimated_bytes"], "latency": result.metrics.average_access_latency})
    for top_k in (1, 3, 5):
        varied = replace(config, top_k=top_k)
        predictor = WeightedTriePredictor(context_depth=2, threshold=0.5).fit(train)
        result = simulate(to_byte_addresses(evaluation, varied.cache_line_size), predictor, varied, initial_context=train[-2:])
        rows.append({"family": "top_k", "depth": 2, "threshold": 0.5, "alpha": 0.25, "top_k": top_k, "destination": varied.prefetch_destination, "cache_capacity": varied.hierarchy.l1_capacity, "speedup": baseline.cycles / result.cycles, "accuracy": result.metrics.top1_accuracy, "storage_bytes": result.predictor_storage["estimated_bytes"], "latency": result.metrics.average_access_latency})
    for destination in ("L1", "L2"):
        varied = replace(config, prefetch_destination=destination)
        predictor = WeightedTriePredictor(context_depth=2, threshold=0.5).fit(train)
        result = simulate(to_byte_addresses(evaluation, varied.cache_line_size), predictor, varied, initial_context=train[-2:])
        rows.append({"family": "destination", "depth": 2, "threshold": 0.5, "alpha": 0.25, "top_k": varied.top_k, "destination": destination, "cache_capacity": varied.hierarchy.l1_capacity, "speedup": baseline.cycles / result.cycles, "accuracy": result.metrics.top1_accuracy, "storage_bytes": result.predictor_storage["estimated_bytes"], "latency": result.metrics.average_access_latency})
    for capacity in (32, 64, 128):
        varied = replace(config, hierarchy=replace(config.hierarchy, l1_capacity=capacity))
        varied_baseline = simulate(to_byte_addresses(evaluation, varied.cache_line_size), None, varied, enable_prefetch=False)
        predictor = WeightedTriePredictor(context_depth=2, threshold=0.5).fit(train)
        result = simulate(to_byte_addresses(evaluation, varied.cache_line_size), predictor, varied, initial_context=train[-2:])
        rows.append({"family": "cache_capacity", "depth": 2, "threshold": 0.5, "alpha": 0.25, "top_k": varied.top_k, "destination": varied.prefetch_destination, "cache_capacity": capacity, "speedup": varied_baseline.cycles / result.cycles, "accuracy": result.metrics.top1_accuracy, "storage_bytes": result.predictor_storage["estimated_bytes"], "latency": result.metrics.average_access_latency})
    for outstanding in (2, 8, 16):
        varied = replace(config, max_outstanding_prefetches=outstanding)
        predictor = WeightedTriePredictor(context_depth=2, threshold=0.5).fit(train)
        result = simulate(to_byte_addresses(evaluation, varied.cache_line_size), predictor, varied, initial_context=train[-2:])
        rows.append({"family": "prefetch_bandwidth", "depth": 2, "threshold": 0.5, "alpha": 0.25, "top_k": varied.top_k, "destination": varied.prefetch_destination, "cache_capacity": outstanding, "speedup": baseline.cycles / result.cycles, "accuracy": result.metrics.top1_accuracy, "storage_bytes": result.predictor_storage["estimated_bytes"], "latency": result.metrics.average_access_latency})
    return rows


def ablation_rows(line_trace: list[int], config: SimulatorConfig) -> list[dict[str, object]]:
    train, evaluation = _split(line_trace, 0.7)
    raw = to_byte_addresses(evaluation, config.cache_line_size)
    baseline = simulate(raw, None, config, enable_prefetch=False)
    specs = [("depth=1", 1, "frequency", 0.05, True), ("depth=2", 2, "frequency", 0.05, True), ("no threshold", 2, "frequency", 0.0, True), ("EMA", 2, "ema", 0.25, True), ("no prefetch", 2, "frequency", 0.05, False)]
    rows = []
    for label, depth, strategy, threshold, prefetch in specs:
        predictor = WeightedTriePredictor(context_depth=depth, strategy=strategy, threshold=threshold).fit(train)
        result = simulate(raw, predictor, config, enable_prefetch=prefetch, initial_context=train[-depth:])
        rows.append({"ablation": label, "speedup": baseline.cycles / result.cycles, "accuracy": result.metrics.top1_accuracy, "latency": result.metrics.average_access_latency, "storage_bytes": result.predictor_storage["estimated_bytes"], "prefetch_precision": result.metrics.prefetch_precision})
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict[str, object], field: str) -> float:
    return float(row.get(field, 0.0) or 0.0)


def write_report(output: Path, summary: list[dict[str, object]], learning: list[dict[str, object]], break_even: list[dict[str, object]], ablations: list[dict[str, object]], sweep: list[dict[str, object]]) -> str:
    wam = [row for row in summary if str(row["predictor"]).startswith("WAM")]
    best = max(wam, key=lambda row: _float(row, "mean_speedup_over_none")) if wam else None
    workloads = sorted({str(row["workload"]) for row in wam})
    wins: list[str] = []
    losses: list[str] = []
    for workload in workloads:
        candidates = [row for row in wam if row["workload"] == workload]
        other = [row for row in summary if row["workload"] == workload and not str(row["predictor"]).startswith("WAM")]
        best_wam = max(candidates, key=lambda row: _float(row, "mean_speedup_over_none"))
        best_other = max(other, key=lambda row: _float(row, "mean_speedup_over_none")) if other else None
        if _float(best_wam, "mean_speedup_over_none") > 1.0 and (best_other is None or _float(best_wam, "mean_speedup_over_none") > _float(best_other, "mean_speedup_over_none")):
            wins.append(workload)
        else:
            losses.append(workload)
    best_per_workload = [max((row for row in wam if row["workload"] == workload), key=lambda row: _float(row, "mean_speedup_over_none")) for workload in workloads]
    geometric_mean = math.exp(statistics.mean([math.log(max(1e-9, _float(row, "mean_speedup_over_none"))) for row in best_per_workload])) if best_per_workload else 0.0
    max_speedup = max((_float(row, "mean_speedup_over_none") for row in wam), default=0.0)
    best_threshold_row = max(break_even, key=lambda row: _float(row, "speedup")) if break_even else None
    deeper_help: list[str] = []
    depth_winners: list[str] = []
    for workload in workloads:
        workload_rows = [row for row in wam if row["workload"] == workload]
        depth_one = next((row for row in workload_rows if row["predictor"] == "WAM depth=1"), None)
        winner = max(workload_rows, key=lambda row: _float(row, "mean_speedup_over_none"))
        if winner["predictor"] != "WAM depth=1":
            depth_winners.append(f"{workload}: {winner['predictor']}")
        if depth_one and _float(winner, "mean_top1_accuracy") > _float(depth_one, "mean_top1_accuracy") + 0.01:
            deeper_help.append(f"{workload}: {winner['predictor']} ({_float(depth_one, 'mean_top1_accuracy'):.1%} -> {_float(winner, 'mean_top1_accuracy'):.1%})")
    best_sweep = max(sweep, key=lambda row: _float(row, "speedup")) if sweep else None
    ema = next((row for row in ablations if row["ablation"] == "EMA"), None)
    depth_two = next((row for row in ablations if row["ablation"] == "depth=2"), None)
    depth_one_reference = next((row for row in wam if row["predictor"] == "WAM depth=1"), None)
    accuracy_gain = (_float(best, "mean_top1_accuracy") - _float(depth_one_reference, "mean_top1_accuracy")) * 100 if best and depth_one_reference else 0.0
    bytes_per_accuracy_point = ((_float(best, "mean_estimated_bytes") - _float(depth_one_reference, "mean_estimated_bytes")) / accuracy_gain) if accuracy_gain > 0 else None
    final_accuracy = _float(learning[-1], "top1_accuracy") if learning else 0.0
    warmup = next((int(row["accesses_observed"]) for row in learning if _float(row, "top1_accuracy") >= final_accuracy * 0.8), 0) if final_accuracy else 0
    break_even_text = []
    for dram in (150, 300):
        points = sorted((row for row in break_even if int(row["dram_latency"]) == dram), key=lambda row: _float(row, "accuracy"))
        point = next((row for row in points if _float(row, "speedup") > 1.0), None)
        break_even_text.append(f"{dram} cycles: {(_float(point, 'accuracy') if point else 'not reached')}")
    best_depth = best["predictor"] if best else "not available"
    lines = [
        "# Weighted Associative Memory Research Report",
        "",
        "This report is generated from deterministic chronological train/test benchmark runs. Synthetic traces are represented as cache-line IDs and converted to raw byte addresses using a 64-byte default line size. The reported latencies are simulation parameters, not claims about a particular CPU.",
        "",
        "## Verdict",
        "",
        f"- WAM wins: {', '.join(wins) if wins else 'none under the default configuration'}.",
        f"- WAM loses or does not beat the best simpler system: {', '.join(losses) if losses else 'none' }.",
        f"- Best observed WAM configuration: {best_depth} with mean speedup {_float(best, 'mean_speedup_over_none'):.3f}x." if best else "- Best observed WAM configuration: unavailable.",
        f"- Geometric-mean speedup of the best WAM depth per workload: {geometric_mean:.3f}x.",
        f"- Maximum mean WAM speedup observed: {max_speedup:.3f}x.",
        f"- Best confidence threshold in the break-even sweep: {_float(best_threshold_row, 'threshold'):.2f} (DRAM={best_threshold_row['dram_latency']}), or not reached." if best_threshold_row else "- Best confidence threshold: unavailable.",
        f"- Storage at the best WAM row: {int(_float(best, 'mean_estimated_bytes'))} bytes." if best else "- Storage at the best WAM row: unavailable.",
        f"- Incremental storage per accuracy percentage point versus depth 1: {bytes_per_accuracy_point:.1f} bytes/point." if bytes_per_accuracy_point is not None else "- Incremental storage per accuracy percentage point versus depth 1: not applicable (no positive gain).",
        f"- Online learning warm-up to approximately 80% of final top-1 accuracy: {warmup} accesses.",
        f"- Approximate accuracy break-even: {'; '.join(break_even_text)}.",
        "",
        "The result should be read workload-by-workload. Sequential and constant-stride streams favor conventional prefetchers; WAM is only expected to justify its state and lookup overhead where higher-order context survives the held-out split. Random access is a negative control.",
        "",
        "## Where context helped",
        "",
        f"Depths beyond 1 were the speedup winner on: {', '.join(depth_winners) if depth_winners else 'none' }.",
        f"Depths that improved top-1 accuracy by more than one percentage point over depth 1: {', '.join(deeper_help) if deeper_help else 'none under this split'}.",
        f"The best contextual sweep point was {best_sweep['depth']} depth, threshold {best_sweep['threshold']}, top-K {best_sweep['top_k']} at {_float(best_sweep, 'speedup'):.3f}x." if best_sweep else "No contextual sweep point was available.",
        "The benchmark does not assume monotonic improvement: larger contexts increase storage and lookup cost, and can under-train when a trace is short or phase-changing.",
        "",
        "## Adaptive weighting and ablations",
        "",
        f"The ablation table compares frequency weighting, EMA weighting, thresholds, depth, and no-prefetch operation. On the contextual ablation, EMA speedup was {_float(ema, 'speedup'):.3f}x versus {_float(depth_two, 'speedup'):.3f}x for frequency depth 2." if ema and depth_two else "The ablation table compares frequency weighting, EMA weighting, thresholds, depth, and no-prefetch operation.",
        "EMA should be judged primarily on the phase-changing trace, where stale frequency counts can remain misleading after a transition.",
        "",
        "## Limitations and next experiment",
        "",
        "The simulator is not cycle-accurate hardware: it models serialized demand progress, bounded outstanding prefetches, cache pollution attribution, and a documented predictor overhead. The single most important next experiment is replaying the same benchmark against long traces captured from representative programs, with a calibrated memory-level parallelism and bandwidth model.",
        "",
        "Artifacts: `summary.csv`, `detailed_results.csv`, `sweep.csv`, `ablation.csv`, `learning_curve.csv`, `break_even.csv`, `config.json`, and `plots/`.",
    ]
    report = "\n".join(lines) + "\n"
    (output / "report.md").write_text(report, encoding="utf-8")
    return report


def _print_verdict(summary: list[dict[str, object]], learning: list[dict[str, object]], break_even: list[dict[str, object]]) -> None:
    wam = [row for row in summary if str(row["predictor"]).startswith("WAM")]
    best = max(wam, key=lambda row: _float(row, "mean_speedup_over_none")) if wam else None
    best_per_workload = [max((row for row in wam if row["workload"] == workload), key=lambda row: _float(row, "mean_speedup_over_none")) for workload in sorted({str(row["workload"]) for row in wam})]
    geo = math.exp(statistics.mean([math.log(max(1e-9, _float(row, "mean_speedup_over_none"))) for row in best_per_workload])) if best_per_workload else 0.0
    warmup = next((row["accesses_observed"] for row in learning if _float(row, "top1_accuracy") >= _float(learning[-1], "top1_accuracy") * 0.8), "n/a") if learning else "n/a"
    print("\n" + "=" * 69)
    print("Weighted Associative Memory Benchmark Verdict")
    print("=" * 69)
    print(f"Best context configuration: {best['predictor'] if best else 'n/a'}")
    print(f"Maximum mean speedup: {_float(best, 'mean_speedup_over_none'):.3f}x" if best else "Maximum mean speedup: n/a")
    print(f"Geometric-mean speedup across workloads: {geo:.3f}x")
    print(f"Best confidence threshold: {_float(max(break_even, key=lambda row: _float(row, 'speedup')), 'threshold'):.2f}" if break_even else "Best confidence threshold: n/a")
    print(f"Storage at best configuration: {int(_float(best, 'mean_estimated_bytes'))} bytes" if best else "Storage at best configuration: n/a")
    print(f"Online learning warm-up: {warmup} accesses")
    for dram in (150, 300):
        point = next((row for row in sorted((row for row in break_even if int(row["dram_latency"]) == dram), key=lambda row: _float(row, "accuracy")) if _float(row, "speedup") > 1), None)
        print(f"Break-even at DRAM={dram}: {(_float(point, 'accuracy') if point else 'not reached')}")


def run_benchmark(output: Path, length: int = 360, trials: int = 10, train_fraction: float = 0.7, trace: Path | None = None) -> dict[str, list[dict[str, object]]]:
    output.mkdir(parents=True, exist_ok=True)
    config = default_simulator_config()
    if trace:
        workloads = {"ExternalTrace": list(normalize_addresses(load_trace(trace), config.cache_line_size))}
    else:
        workloads = all_workloads(length, seed=0)
    detailed: list[dict[str, object]] = []
    for trial in range(1 if trace else trials):
        trial_workloads = workloads if trace else all_workloads(length, seed=trial)
        for workload, line_trace in trial_workloads.items():
            detailed.extend(evaluate_trace(workload, line_trace, trial, config, train_fraction=train_fraction))
    summary = summarize(detailed)
    learning = online_learning_curve(workloads.get("PhaseChanging", next(iter(workloads.values())) ,), config)
    break_even = break_even_rows(workloads.get("Probabilistic", next(iter(workloads.values()))))
    sweep = sweep_rows(workloads.get("Contextual", next(iter(workloads.values()))), config)
    ablations = ablation_rows(workloads.get("Contextual", next(iter(workloads.values()))), config)
    _write_csv(output / "detailed_results.csv", detailed)
    _write_csv(output / "summary.csv", summary)
    _write_csv(output / "learning_curve.csv", learning)
    _write_csv(output / "break_even.csv", break_even)
    _write_csv(output / "sweep.csv", sweep)
    _write_csv(output / "ablation.csv", ablations)
    config_json = {"simulator": asdict(config), "length": length, "trials": trials, "train_fraction": train_fraction, "systems": SYSTEM_NAMES, "depths": DEFAULT_DEPTHS, "thresholds": [0.5, 0.65, 0.8, 0.9], "ema_alpha": [0.01, 0.05, 0.1, 0.25], "workloads": list(workloads)}
    (output / "config.json").write_text(json.dumps(config_json, indent=2), encoding="utf-8")
    write_report(output, summary, learning, break_even, ablations, sweep)
    from .visualization import generate_benchmark_plots

    generate_benchmark_plots(summary, detailed, learning, break_even, output / "plots")
    _print_verdict(summary, learning, break_even)
    return {"summary": summary, "detailed": detailed, "learning": learning, "break_even": break_even, "sweep": sweep, "ablation": ablations}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--length", type=int, default=360)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--trace", type=Path, help="optional plain-text hexadecimal/integer trace")
    args = parser.parse_args()
    run_benchmark(args.output, args.length, args.trials, args.train_fraction, args.trace)


if __name__ == "__main__":
    main()
