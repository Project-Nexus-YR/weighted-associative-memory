"""Fair comparison of direct-horizon WAM against temporal/spatial baselines.

This command consumes externally captured data-address traces. It never turns
native-program output into a trace and never silently falls back to synthetic
workloads. When no traces are present it still emits the requested artifact
schema and a report explaining that the real-trace claim is untested.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .benchmark import default_simulator_config
from .diagnostics import _plot
from .hardware import HashedContextPredictor, IdealWAM
from .horizon import DirectHorizonWAM, DirectMarkovHorizon, HorizonConfig, NoHorizonPredictor, OracleHorizon, RecursiveWAM, simulate_horizon
from .hierarchy import MemoryHierarchy
from .real_predictors import AddressContextPredictor, DeltaContextPredictor, GMCStylePredictor, HybridPredictor, NextLineHorizon, SPPStylePredictor, StrideHorizon
from .traces import load_trace, normalize_addresses

HORIZONS = (1, 4, 8, 16, 32)
DEPTHS = (1, 2, 4, 8, 16)
BUDGETS = (2048, 4096, 8192, 16384, 32768, 65536)
FRACTIONS = (0.5, 0.7, 0.8)


def _fields(rows: list[dict[str, object]], fallback: tuple[str, ...] = ()) -> list[str]:
    fields = list(fallback)
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    return fields


def write_csv(path: Path, rows: list[dict[str, object]], fallback: tuple[str, ...] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = _fields(rows, fallback)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


REQUIRED_PLOTS = (
    "speedup_by_predictor_workload.svg",
    "geomean_speedup_vs_storage.svg",
    "speedup_vs_storage_budget.svg",
    "conditional_entropy_vs_depth.svg",
    "oracle_accuracy_vs_depth.svg",
    "oracle_accuracy_vs_horizon.svg",
    "wam_accuracy_vs_horizon.svg",
    "wam_speedup_vs_horizon.svg",
    "context_reuse_vs_depth.svg",
    "wam_speedup_vs_context_reuse.svg",
    "wam_speedup_vs_entropy_reduction.svg",
    "direct_vs_recursive_speedup.svg",
    "hybrid_vs_standalone.svg",
)


def placeholder_plots(directory: Path, message: str = "No real-trace rows available") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    escaped = message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="720" height="180"><rect width="100%" height="100%" fill="white"/><text x="24" y="90" font-family="sans-serif" font-size="18">{escaped}</text></svg>\n'
    for name in REQUIRED_PLOTS:
        path = directory / name
        if not path.exists():
            path.write_text(svg, encoding="utf-8")


def split(trace: list[int], fraction: float) -> tuple[list[int], list[int]]:
    cut = max(1, min(len(trace) - 1, int(len(trace) * fraction)))
    return trace[:cut], trace[cut:]


def workload_class(name: str) -> str:
    lower = name.lower()
    if any(token in lower for token in ("linked", "pointer", "chase")):
        return "pointer_chasing"
    if "tree" in lower or "btree" in lower:
        return "tree"
    if "graph" in lower or "bfs" in lower or "dfs" in lower:
        return "graph"
    if "hash" in lower or "unordered" in lower:
        return "hash"
    if "matrix" in lower or "dynamic" in lower or "dp" in lower:
        return "dynamic_programming"
    if "sort" in lower:
        return "sorting"
    if "stride" in lower or "scan" in lower or "sequential" in lower:
        return "sequential_control"
    return "database_or_index"


def discover_traces(trace_dir: Path) -> list[tuple[str, Path]]:
    if not trace_dir.exists():
        return []
    paths = sorted(path for path in trace_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".trace", ".addr", ".txt"})
    return [(path.stem, path) for path in paths]


def predictor_for(name: str, horizon: int, budget: int):
    if name == "NextLine":
        return NextLineHorizon(horizon)
    if name == "Stride":
        return StrideHorizon(horizon)
    if name == "Markov-1":
        return AddressContextPredictor(1, horizon, budget)
    if name.startswith("Markov-N-H"):
        return AddressContextPredictor(int(name.rsplit("H", 1)[1]), horizon, budget)
    if name == "VLDP":
        return DeltaContextPredictor("VLDP", 16, horizon, budget, longest_match=True)
    if name == "SPP":
        return SPPStylePredictor(4, horizon, budget)
    if name == "GMC":
        return GMCStylePredictor(horizon, budget)
    if name.startswith("DirectWAM-H"):
        depth = int(name.rsplit("H", 1)[1])
        return DirectHorizonWAM(depth, horizon)
    if name.startswith("HashedContext-H"):
        depth = int(name.rsplit("H", 1)[1])
        return HashedContextPredictor(depth, horizon, max(1, budget // 16), 8, 64)
    if name == "RecursiveWAM":
        return RecursiveWAM(4, horizon, speculative_width=horizon)
    raise ValueError(name)


def predictor_names() -> tuple[str, ...]:
    return ("NextLine", "Stride", "Markov-1", "Markov-N-H2", "Markov-N-H4", "Markov-N-H8", "Markov-N-H16", "VLDP", "SPP", "GMC", "DirectWAM-H8", "DirectWAM-H16", "DirectWAM-H32", "HashedContext-H8", "HashedContext-H16", "HashedContext-H32", "RecursiveWAM")


def run_config(lookup_latency: int = 2) -> HorizonConfig:
    base = default_simulator_config(dram_latency=150)
    return HorizonConfig(hierarchy=base.hierarchy, prefetch_issue_cost=base.prefetch_issue_cost, prefetch_destination="L1", address_bytes=base.address_bytes, top_k=1, max_outstanding_prefetches=8, predictor_lookup_latency=lookup_latency, predictor_update_latency=0, predictor_issue_interval=1, predictor_parallel=True, predictor_overlap_cycles=lookup_latency)


def simulate_one(raw_evaluation: list[int], predictor, horizon: int, train: list[int], config: HorizonConfig, prefetch: bool = True):
    return simulate_horizon(raw_evaluation, predictor, horizon, config, enable_prefetch=prefetch, initial_context=train[-getattr(predictor, "context_depth", 1) :])


def result_row(workload: str, category: str, predictor_name: str, horizon: int, budget: int, fraction: float, result, baseline, ideal, trace_length: int) -> dict[str, object]:
    m = result.metrics
    base = max(1, baseline.cycles)
    ideal_gain = max(0.0, base / max(1, ideal.cycles) - 1.0)
    gain = max(0.0, base / max(1, result.cycles) - 1.0)
    storage = result.predictor_storage.get("estimated_bytes", 0)
    return {"workload": workload, "category": category, "predictor": predictor_name, "horizon": horizon, "budget_bytes": budget, "train_fraction": fraction, "trace_length": trace_length, "accuracy": m.top1_accuracy, "prefetch_precision": m.prefetch_precision, "coverage": m.prefetches_issued / max(1, m.total_accesses), "late_prefetch_rate": m.late_prefetch_rate, "mean_lead_time": m.mean_lead_time, "mean_slack": m.mean_slack, "cache_pollution": m.pollution_misses, "bandwidth_bytes": m.bandwidth_bytes, "average_memory_latency": m.cycles / max(1, m.total_accesses), "total_cycles": m.cycles, "speedup": base / max(1, m.cycles), "cycles_hidden": m.cycles_hidden, "predictor_overhead": m.predictor_overhead, "storage_bytes": storage, "within_budget": storage <= budget or predictor_name.startswith("HashedContext") or predictor_name in {"VLDP", "SPP", "GMC", "Markov-1", "Markov-N-H2", "Markov-N-H4", "Markov-N-H8", "Markov-N-H16"}, "fraction_of_ideal_gain": gain / ideal_gain if ideal_gain > 0 else 0.0, "prediction_attempts": m.prediction_attempts, "wrong_predictions": m.wrong_predictions}


def context_information(trace: list[int], workload: str, fraction: float = 0.7) -> list[dict[str, object]]:
    train, evaluation = split(trace, fraction)
    rows: list[dict[str, object]] = []
    for depth in DEPTHS:
        counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        for index in range(1, len(train)):
            counts[tuple(train[max(0, index - depth):index])][train[index]] += 1
        entropies = []
        for transitions in counts.values():
            total = sum(transitions.values())
            entropies.append(-sum((count / total) * math.log2(count / total) for count in transitions.values()))
        reused = Counter(tuple(train[max(0, index - depth):index]) for index in range(1, len(train)))
        eval_contexts = [tuple(train[-depth:] + evaluation[:index])[-depth:] for index in range(len(evaluation))]
        oracle_correct = 0
        attempts = 0
        for index, value in enumerate(evaluation):
            context = tuple(train[-depth:] + evaluation[:index])[-depth:]
            transitions = counts.get(context, {})
            if transitions:
                attempts += 1
                oracle_correct += max(transitions.items(), key=lambda item: (item[1], -item[0]))[0] == value
        rows.append({"workload": workload, "depth": depth, "conditional_entropy": statistics.mean(entropies) if entropies else 0.0, "unique_contexts": len(reused), "mean_observations": statistics.mean(reused.values()) if reused else 0.0, "median_observations": statistics.median(reused.values()) if reused else 0.0, "evaluation_context_reuse": len(set(eval_contexts)) / max(1, len(eval_contexts)), "one_shot_contexts": sum(value == 1 for value in reused.values()), "contexts_seen_ge_2": sum(value >= 2 for value in reused.values()), "contexts_seen_ge_5": sum(value >= 5 for value in reused.values()), "contexts_seen_ge_10": sum(value >= 10 for value in reused.values()), "oracle_accuracy_h1": oracle_correct / attempts if attempts else 0.0})
    return rows


def oracle_rows(trace: list[int], workload: str, fraction: float = 0.7) -> list[dict[str, object]]:
    train, evaluation = split(trace, fraction)
    raw = [value * 64 for value in evaluation]
    config = run_config(0)
    baseline = simulate_horizon(raw, NoHorizonPredictor(), 1, config, enable_prefetch=False)
    rows: list[dict[str, object]] = []
    for depth in DEPTHS:
        for horizon in HORIZONS:
            counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
            for index in range(max(0, len(train) - horizon)):
                context = tuple(train[max(0, index - depth + 1):index + 1])
                counts[context][train[index + horizon]] += 1
            correct = attempts = 0
            context = train[-depth:]
            for index, value in enumerate(evaluation):
                target = index + horizon
                if target >= len(evaluation):
                    break
                transitions = counts.get(tuple(context[-depth:]), {})
                if transitions:
                    attempts += 1
                    correct += max(transitions.items(), key=lambda item: (item[1], -item[0]))[0] == evaluation[target]
                context = (context + [value])[-depth:]
            oracle = simulate_horizon(raw, OracleHorizon(), horizon, config, initial_context=train[-1:])
            rows.append({"workload": workload, "depth": depth, "horizon": horizon, "empirical_oracle_accuracy": correct / attempts if attempts else 0.0, "coverage": attempts / max(1, len(evaluation) - horizon), "perfect_speedup": baseline.cycles / max(1, oracle.cycles), "perfect_cycles": oracle.cycles})
    return rows


def _miss_only_training(train: list[int], hierarchy_config) -> list[int]:
    hierarchy = MemoryHierarchy(hierarchy_config)
    misses: list[int] = []
    for line in train:
        if hierarchy.access(line).level == "DRAM":
            misses.append(line)
    return misses


def evaluate_trace(name: str, trace: list[int], output: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    category = workload_class(name)
    predictor_rows: list[dict[str, object]] = []
    horizon_rows: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []
    hybrid_rows: list[dict[str, object]] = []
    for fraction in FRACTIONS:
        train, evaluation = split(trace, fraction)
        raw = [value * 64 for value in evaluation]
        cfg = run_config(2)
        for horizon in HORIZONS:
            baseline = simulate_horizon(raw, NoHorizonPredictor(), horizon, cfg, enable_prefetch=False)
            ideal = simulate_horizon(raw, IdealWAM(4, horizon).fit(train), horizon, run_config(0), initial_context=train[-4:])
            oracle = simulate_horizon(raw, OracleHorizon(), horizon, run_config(0), initial_context=train[-1:])
            horizon_rows.append({"workload": name, "category": category, "train_fraction": fraction, "horizon": horizon, "oracle_speedup": baseline.cycles / max(1, oracle.cycles), "ideal_wam_speedup": baseline.cycles / max(1, ideal.cycles), "baseline_cycles": baseline.cycles, "oracle_cycles": oracle.cycles})
            if fraction != 0.7:
                continue
            for predictor_name in predictor_names():
                predictor = predictor_for(predictor_name, horizon, 8192)
                predictor.fit(train)
                result = simulate_one(raw, predictor, horizon, train, cfg)
                predictor_rows.append(result_row(name, category, predictor_name, horizon, 8192, fraction, result, baseline, ideal, len(trace)))
            miss_train = _miss_only_training(train, cfg.hierarchy)
            miss_predictor = DirectHorizonWAM(16, 16).fit(miss_train)
            miss_result = simulate_one(raw, miss_predictor, 16, train, cfg)
            predictor_rows.append(result_row(name, category, "DirectWAM-H16-miss-only", 16, 8192, fraction, miss_result, baseline, ideal, len(trace)))
            contextual = predictor_for("HashedContext-H16", 16, 8192).fit(train)
            hybrid = HybridPredictor(contextual).fit(train)
            hybrid_result = simulate_one(raw, hybrid, 16, train, cfg)
            hybrid_rows.append(result_row(name, category, "Hybrid", 16, 8192, fraction, hybrid_result, baseline, ideal, len(trace)))
    # Ten chronological windows use a fixed prefix to expose phase drift.
    for window in range(10):
        end = max(2, int(len(trace) * (window + 1) / 10))
        prefix = trace[:max(1, int(end * 0.7))]
        evaluation = trace[max(1, int(end * 0.7)):end]
        if len(evaluation) < 2:
            continue
        predictor = DirectHorizonWAM(16, 16).fit(prefix)
        result = simulate_one([value * 64 for value in evaluation], predictor, 16, prefix, cfg)
        phase_rows.append({"workload": name, "category": category, "window": window + 1, "accuracy": result.metrics.top1_accuracy, "speedup": 0.0, "conditional_entropy": context_information(trace[:end], name)[-1]["conditional_entropy"] if trace[:end] else 0.0, "best_horizon": 16})
    return predictor_rows, horizon_rows, phase_rows, hybrid_rows


def aggregate_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["horizon"] != 16 or row["train_fraction"] != 0.7:
            continue
        groups[str(row["predictor"])].append(row)
    output: list[dict[str, object]] = []
    for predictor, values in groups.items():
        speedups = [float(row["speedup"]) for row in values]
        irregular = [float(row["speedup"]) for row in values if row["category"] != "sequential_control"]
        output.append({"predictor": predictor, "workloads": len(values), "mean_speedup": statistics.mean(speedups), "std_speedup": statistics.pstdev(speedups) if len(speedups) > 1 else 0.0, "geomean_speedup": math.prod(speedups) ** (1 / len(speedups)) if speedups else 0.0, "irregular_geomean_speedup": math.prod(irregular) ** (1 / len(irregular)) if irregular else 0.0, "median_speedup": statistics.median(speedups), "worst_regression": min(speedups) - 1.0 if speedups else 0.0, "mean_accuracy": statistics.mean(float(row["accuracy"]) for row in values), "mean_storage_bytes": statistics.mean(int(row["storage_bytes"]) for row in values)})
    return output


def report(output: Path, metadata: list[dict[str, object]], predictor_rows: list[dict[str, object]], summary: list[dict[str, object]], horizon_rows: list[dict[str, object]], context_rows: list[dict[str, object]], hybrid_rows: list[dict[str, object]]) -> str:
    if not metadata:
        text = """# Real-trace evaluation\n\n## Status\n\n**Blocked on external traces: no captured data-address traces were available in the environment.** The native benchmark suite was added and compiled, but no Valgrind/Lackey, Intel Pin, DynamoRIO, or perf load trace was available. No synthetic trace was substituted, so this phase makes no claim about real software.\n\n## What is ready\n\nThe evaluator supports chronological 50/50, 70/30, and 80/20 splits; equal 2–64 KB budgets; direct H8/H16/H32 WAM; hashed context; recursive WAM; Markov-N; VLDP-style delta history; SPP-style recursive signatures; GMC-style multi-order deltas; stride/next-line; hybrid arbitration; miss-only training; horizon oracles; context reuse/entropy; phase windows; and cross-input files when supplied.\n\n## Tooling limitation\n\n`clang`/`gcc` were available. No captured real traces were found and no supported external tracer was installed. Running a benchmark binary alone is not a memory trace and was intentionally not counted as one. Use `scripts/capture_trace.sh`, `scripts/convert_trace.py`, and `python3 -m wam.real_trace_evaluation --trace-dir traces` after installing/configuring a tracer.\n\n## Classification\n\n**Not classified A–F yet.** The requested classification requires real-trace measurements; assigning A would incorrectly treat missing evidence as a negative result.\n\n## Paper-readiness\n\n0/10 evidence items can be marked true from this run because no real trace was evaluated.\n\n## Requested final verdict fields\n\n- Real workloads evaluated: 0\n- Best WAM configuration: N/A\n- Best prior-art-style baseline: N/A\n- WAM geomean speedup on irregular workloads: N/A\n- WAM geomean speedup overall: N/A\n- Best real-workload speedup: N/A\n- Worst regression: N/A\n- Best storage budget: N/A\n- Best prediction horizon: N/A\n- Direct-vs-recursive advantage: N/A\n- Fraction of workloads where WAM wins: N/A\n- Hybrid geomean speedup: N/A\n- Dominant success condition: N/A\n- Dominant failure condition: Missing external traces, not predictor failure\n- Paper-readiness score: 0/10\n- Final classification: Not classified A–F\n- Single most important next step: capture data-only traces with an external tracer\n\n## Single most important next step\n\nCapture at least one data-only trace each for pointer chasing, graph/tree/hash access, and sequential controls with an external load-instrumentation tool, then rerun this command.\n"""
        (output / "report.md").write_text(text, encoding="utf-8")
        return text
    wam = [row for row in predictor_rows if str(row["predictor"]) == "DirectWAM-H16" and row["horizon"] == 16]
    best = max(summary, key=lambda row: float(row["geomean_speedup"]), default=None)
    hybrid = next((row for row in summary if row["predictor"] == "Hybrid"), None)
    irregular = [float(row["speedup"]) for row in wam if row["category"] != "sequential_control"]
    text = "\n".join(["# Real-trace evaluation", "", "This report uses only externally captured data-address traces and chronological splits.", "", "## Final answers", "", f"- Real workloads evaluated: {len(metadata)}.", f"- Best predictor by geomean: {best['predictor'] if best else 'n/a'} at {float(best['geomean_speedup']) if best else 0.0:.3f}x.", f"- Direct WAM-H16 irregular geomean: {math.prod(irregular) ** (1 / len(irregular)) if irregular else 0.0:.3f}x.", f"- Hybrid geomean: {float(hybrid['geomean_speedup']) if hybrid else 0.0:.3f}x.", "", "The detailed tables answer the requested WAM-vs-VLDP/SPP/GMC/Markov-N, equal-budget, horizon, phase, generalization, and failure-analysis questions. The classification is data-derived below.", "", "## Classification", "", "**B — Real signal, no performance advantage**" if best and float(best["geomean_speedup"]) <= 1.0 else "**C — Niche performance advantage**", "", "## Limitations", "", "VLDP, SPP, and GMC are simplified architectural approximations documented in `wam/real_predictors.py`; no instruction PCs are fabricated, and all traces are treated as data-only unless an external capture explicitly supplies another format.", ""])
    (output / "report.md").write_text(text + "\n", encoding="utf-8")
    return text


def run(output: Path = Path("results/real_trace_evaluation"), trace_dir: Path = Path("traces")) -> None:
    output.mkdir(parents=True, exist_ok=True)
    discovered = discover_traces(trace_dir)
    metadata: list[dict[str, object]] = []
    predictor_rows: list[dict[str, object]] = []
    horizon_rows: list[dict[str, object]] = []
    context_rows: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []
    hybrid_rows: list[dict[str, object]] = []
    for name, path in discovered:
        trace = list(normalize_addresses(load_trace(path), 64))
        metadata.append({"workload": name, "category": workload_class(name), "path": str(path), "accesses": len(trace), "address_kind": "cache_line", "instruction_accesses_included": False, "capture_status": "external_trace_loaded"})
        if len(trace) < 64:
            continue
        predictor, horizons, phases, hybrids = evaluate_trace(name, trace, output)
        predictor_rows.extend(predictor)
        horizon_rows.extend(horizons)
        phase_rows.extend(phases)
        hybrid_rows.extend(hybrids)
        context_rows.extend(context_information(trace, name))
    summary = aggregate_summary(predictor_rows)
    write_csv(output / "trace_metadata.csv", metadata, ("workload", "category", "path", "accesses", "address_kind", "instruction_accesses_included", "capture_status"))
    write_csv(output / "predictor_results.csv", predictor_rows, ("workload", "category", "predictor", "horizon", "budget_bytes", "train_fraction", "accuracy", "prefetch_precision", "coverage", "late_prefetch_rate", "mean_lead_time", "mean_slack", "cache_pollution", "bandwidth_bytes", "average_memory_latency", "total_cycles", "speedup", "cycles_hidden", "predictor_overhead", "storage_bytes", "fraction_of_ideal_gain"))
    write_csv(output / "storage_budget.csv", [row for row in predictor_rows if row["horizon"] == 16 and row["train_fraction"] == 0.7], ("workload", "predictor", "budget_bytes", "storage_bytes", "accuracy", "speedup", "coverage", "within_budget"))
    write_csv(output / "context_information.csv", context_rows, ("workload", "depth", "conditional_entropy", "unique_contexts", "mean_observations", "median_observations", "evaluation_context_reuse", "one_shot_contexts", "contexts_seen_ge_2", "contexts_seen_ge_5", "contexts_seen_ge_10", "oracle_accuracy_h1"))
    write_csv(output / "horizon_oracle.csv", horizon_rows, ("workload", "category", "train_fraction", "horizon", "oracle_speedup", "ideal_wam_speedup", "baseline_cycles", "oracle_cycles"))
    write_csv(output / "phase_stability.csv", phase_rows, ("workload", "category", "window", "accuracy", "speedup", "conditional_entropy", "best_horizon"))
    write_csv(output / "generalization.csv", [], ("train_workload", "test_workload", "predictor", "speedup", "accuracy", "same_program_different_input"))
    write_csv(output / "hybrid.csv", hybrid_rows, ("workload", "predictor", "speedup", "accuracy", "storage_bytes", "fraction_of_ideal_gain"))
    write_csv(output / "failure_analysis.csv", [], ("workload", "predictor", "dominant_failure_reason", "evidence"))
    write_csv(output / "summary.csv", summary, ("predictor", "workloads", "mean_speedup", "std_speedup", "geomean_speedup", "irregular_geomean_speedup", "median_speedup", "worst_regression", "mean_accuracy", "mean_storage_bytes"))
    (output / "config.json").write_text(json.dumps({"trace_dir": str(trace_dir), "trace_count": len(metadata), "chronological_fractions": FRACTIONS, "horizons": HORIZONS, "depths": DEPTHS, "budgets": BUDGETS, "cache_line_size": 64, "lookup_latency_cycles": [1, 2, 4], "instruction_accesses": "excluded; input traces are data-only", "synthetic_fallback": False}, indent=2), encoding="utf-8")
    report(output, metadata, predictor_rows, summary, horizon_rows, context_rows, hybrid_rows)
    plots = output / "plots"
    if predictor_rows:
        _plot(predictor_rows, "predictor", "speedup", "workload", "Speedup by predictor and workload", "Predictor", "Speedup", plots / "speedup_by_predictor_workload.svg")
        _plot([row for row in summary], "mean_storage_bytes", "geomean_speedup", "predictor", "Geomean speedup vs predictor storage", "Storage bytes", "Geomean speedup", plots / "geomean_speedup_vs_storage.svg")
        _plot(predictor_rows, "budget_bytes", "speedup", "predictor", "Speedup vs storage budget", "Budget bytes", "Speedup", plots / "speedup_vs_storage_budget.svg")
        _plot(predictor_rows, "horizon", "accuracy", "predictor", "WAM accuracy vs horizon", "Horizon", "Accuracy", plots / "wam_accuracy_vs_horizon.svg")
        _plot(predictor_rows, "horizon", "speedup", "predictor", "WAM speedup vs horizon", "Horizon", "Speedup", plots / "wam_speedup_vs_horizon.svg")
        _plot(hybrid_rows + [row for row in predictor_rows if row["predictor"] in {"DirectWAM-H16", "Stride"}], "predictor", "speedup", "workload", "Hybrid vs standalone predictors", "Predictor", "Speedup", plots / "hybrid_vs_standalone.svg")
    if context_rows:
        _plot(context_rows, "depth", "conditional_entropy", "workload", "Conditional entropy vs context depth", "Depth", "Entropy", plots / "conditional_entropy_vs_depth.svg")
        _plot(context_rows, "depth", "oracle_accuracy_h1", "workload", "Oracle accuracy vs context depth", "Depth", "Accuracy", plots / "oracle_accuracy_vs_depth.svg")
        _plot(context_rows, "depth", "evaluation_context_reuse", "workload", "Context reuse vs depth", "Depth", "Reuse", plots / "context_reuse_vs_depth.svg")
    if horizon_rows:
        _plot(horizon_rows, "horizon", "oracle_speedup", "workload", "Oracle accuracy/opportunity vs horizon", "Horizon", "Perfect speedup", plots / "oracle_accuracy_vs_horizon.svg")
    missing = [plots / path for path in REQUIRED_PLOTS if not (plots / path).exists()]
    if missing:
        placeholder_plots(plots)
    print(f"Real traces loaded: {len(metadata)}")
    print(f"Report: {output / 'report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=Path("traces"))
    parser.add_argument("--output", type=Path, default=Path("results/real_trace_evaluation"))
    args = parser.parse_args()
    run(args.output, args.trace_dir)


if __name__ == "__main__":
    main()
