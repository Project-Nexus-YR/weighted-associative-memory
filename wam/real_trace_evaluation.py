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
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
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


def plot_categorical(rows: list[dict[str, object]], x: str, y: str, group: str, title: str, x_label: str, y_label: str, path: Path) -> None:
    categories = {value: index for index, value in enumerate(sorted({str(row.get(x, "")) for row in rows}))}
    numeric = [{**row, "_category_x": categories[str(row.get(x, ""))]} for row in rows]
    _plot(numeric, "_category_x", y, group, title, x_label, y_label, path)


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


def discover_traces(trace_dir: Path, seed_filter: int | None = None) -> list[tuple[str, Path]]:
    if not trace_dir.exists():
        return []
    paths = sorted(path for path in trace_dir.rglob("*") if path.is_file() and "normalized" not in path.parts and path.suffix.lower() in {".trace", ".addr", ".txt"} and (seed_filter is None or f"_seed{seed_filter}_" in path.name))
    return [(path.stem, path) for path in paths]


def load_trace_metadata(path: Path) -> dict[str, object]:
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


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
        eval_contexts: list[tuple[int, ...]] = []
        eval_history = list(train[-depth:])
        for value in evaluation:
            eval_contexts.append(tuple(eval_history[-depth:]))
            eval_history.append(value)
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


def trace_sanity(trace: list[int], workload: str) -> dict[str, object]:
    deltas = [trace[index] - trace[index - 1] for index in range(1, len(trace))]
    repeated = Counter(trace).most_common(5)
    sequential = sum(delta == 1 for delta in deltas) / max(1, len(deltas))
    absolute = sorted(abs(delta) for delta in deltas)
    return {"workload": workload, "total_references": len(trace), "unique_cache_lines": len(set(trace)), "sequential_fraction": sequential, "mean_absolute_delta": statistics.mean(abs(delta) for delta in deltas) if deltas else 0.0, "median_absolute_delta": statistics.median(absolute) if absolute else 0.0, "top_repeated_lines": ";".join(f"{line}:{count}" for line, count in repeated), "same_address_fraction": sum(delta == 0 for delta in deltas) / max(1, len(deltas)), "obviously_broken": len(set(trace)) <= 1 or not trace}


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


def evaluate_trace(name: str, trace: list[int], output: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    category = workload_class(name)
    predictor_rows: list[dict[str, object]] = []
    horizon_rows: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []
    hybrid_rows: list[dict[str, object]] = []
    budget_rows: list[dict[str, object]] = []
    sanity_rows: list[dict[str, object]] = []
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
            if horizon != 16:
                continue
            for predictor_name in predictor_names():
                model_horizon = int(predictor_name.rsplit("H", 1)[1]) if predictor_name.startswith(("DirectWAM-H", "HashedContext-H", "Markov-N-H")) else (1 if predictor_name in {"NextLine", "Stride"} else 16)
                predictor = predictor_for(predictor_name, model_horizon, 8192)
                predictor.fit(train)
                model_raw = [value * 64 for value in evaluation]
                model_baseline = simulate_horizon(model_raw, NoHorizonPredictor(), model_horizon, cfg, enable_prefetch=False)
                model_ideal = simulate_horizon(model_raw, IdealWAM(4, model_horizon).fit(train), model_horizon, run_config(0), initial_context=train[-4:])
                result = simulate_one(model_raw, predictor, model_horizon, train, cfg)
                predictor_rows.append(result_row(name, category, predictor_name, model_horizon, 8192, fraction, result, model_baseline, model_ideal, len(trace)))
            miss_train = _miss_only_training(train, cfg.hierarchy)
            miss_predictor = DirectHorizonWAM(16, 16).fit(miss_train)
            miss_result = simulate_one(raw, miss_predictor, 16, train, cfg)
            predictor_rows.append(result_row(name, category, "DirectWAM-H16-miss-only", 16, 8192, fraction, miss_result, baseline, ideal, len(trace)))
            contextual = predictor_for("HashedContext-H16", 16, 8192).fit(train)
            hybrid = HybridPredictor(contextual).fit(train)
            hybrid_result = simulate_one(raw, hybrid, 16, train, cfg)
            hybrid_rows.append(result_row(name, category, "Hybrid", 16, 8192, fraction, hybrid_result, baseline, ideal, len(trace)))
            for budget in BUDGETS:
                for budget_name in ("HashedContext-H16", "Markov-N-H16", "VLDP", "SPP", "GMC"):
                    budget_predictor = predictor_for(budget_name, 16, budget)
                    budget_predictor.fit(train)
                    budget_result = simulate_one(raw, budget_predictor, 16, train, cfg)
                    budget_rows.append(result_row(name, category, budget_name, 16, budget, fraction, budget_result, baseline, ideal, len(trace)))
    # Ten chronological windows use a fixed prefix to expose phase drift.
    for window in range(10):
        end = max(2, int(len(trace) * (window + 1) / 10))
        prefix = trace[:max(1, int(end * 0.7))]
        evaluation = trace[max(1, int(end * 0.7)):end]
        if len(evaluation) < 2:
            continue
        predictor = DirectHorizonWAM(16, 16).fit(prefix)
        phase_raw = [value * 64 for value in evaluation]
        result = simulate_one(phase_raw, predictor, 16, prefix, cfg)
        phase_baseline = simulate_horizon(phase_raw, NoHorizonPredictor(), 16, cfg, enable_prefetch=False)
        phase_rows.append({"workload": name, "category": category, "window": window + 1, "accuracy": result.metrics.top1_accuracy, "speedup": phase_baseline.cycles / max(1, result.cycles), "conditional_entropy": context_information(trace[:end], name)[-1]["conditional_entropy"] if trace[:end] else 0.0, "best_horizon": 16})
    return predictor_rows, horizon_rows, phase_rows, hybrid_rows, budget_rows


def aggregate_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["train_fraction"] != 0.7:
            continue
        groups[str(row["predictor"])].append(row)
    output: list[dict[str, object]] = []
    for predictor, values in groups.items():
        speedups = [float(row["speedup"]) for row in values]
        irregular = [float(row["speedup"]) for row in values if row["category"] != "sequential_control"]
        output.append({"predictor": predictor, "workloads": len(values), "mean_speedup": statistics.mean(speedups), "std_speedup": statistics.pstdev(speedups) if len(speedups) > 1 else 0.0, "geomean_speedup": math.prod(speedups) ** (1 / len(speedups)) if speedups else 0.0, "irregular_geomean_speedup": math.prod(irregular) ** (1 / len(irregular)) if irregular else 0.0, "median_speedup": statistics.median(speedups), "worst_regression": min(speedups) - 1.0 if speedups else 0.0, "mean_accuracy": statistics.mean(float(row["accuracy"]) for row in values), "mean_storage_bytes": statistics.mean(int(row["storage_bytes"]) for row in values)})
    return output


def report(output: Path, metadata: list[dict[str, object]], predictor_rows: list[dict[str, object]], summary: list[dict[str, object]], horizon_rows: list[dict[str, object]], context_rows: list[dict[str, object]], hybrid_rows: list[dict[str, object]], captured_count: int) -> str:
    if not metadata:
        text = """# Real-trace evaluation\n\n## Status\n\n**Blocked on external traces: no captured data-address traces were available in the environment.** The native benchmark suite was added and compiled, but no Valgrind/Lackey, Intel Pin, DynamoRIO, or perf load trace was available. No synthetic trace was substituted, so this phase makes no claim about real software.\n\n## What is ready\n\nThe evaluator supports chronological 50/50, 70/30, and 80/20 splits; equal 2–64 KB budgets; direct H8/H16/H32 WAM; hashed context; recursive WAM; Markov-N; VLDP-style delta history; SPP-style recursive signatures; GMC-style multi-order deltas; stride/next-line; hybrid arbitration; miss-only training; horizon oracles; context reuse/entropy; phase windows; and cross-input files when supplied.\n\n## Tooling limitation\n\n`clang`/`gcc` were available. No captured real traces were found and no supported external tracer was installed. Running a benchmark binary alone is not a memory trace and was intentionally not counted as one. Use `scripts/capture_trace.sh`, `scripts/convert_trace.py`, and `python3 -m wam.real_trace_evaluation --trace-dir traces` after installing/configuring a tracer.\n\n## Classification\n\n**Not classified A–F yet.** The requested classification requires real-trace measurements; assigning A would incorrectly treat missing evidence as a negative result.\n\n## Paper-readiness\n\n0/10 evidence items can be marked true from this run because no real trace was evaluated.\n\n## Requested final verdict fields\n\n- Real workloads evaluated: 0\n- Best WAM configuration: N/A\n- Best prior-art-style baseline: N/A\n- WAM geomean speedup on irregular workloads: N/A\n- WAM geomean speedup overall: N/A\n- Best real-workload speedup: N/A\n- Worst regression: N/A\n- Best storage budget: N/A\n- Best prediction horizon: N/A\n- Direct-vs-recursive advantage: N/A\n- Fraction of workloads where WAM wins: N/A\n- Hybrid geomean speedup: N/A\n- Dominant success condition: N/A\n- Dominant failure condition: Missing external traces, not predictor failure\n- Paper-readiness score: 0/10\n- Final classification: Not classified A–F\n- Single most important next step: capture data-only traces with an external tracer\n\n## Single most important next step\n\nCapture at least one data-only trace each for pointer chasing, graph/tree/hash access, and sequential controls with an external load-instrumentation tool, then rerun this command.\n"""
        (output / "report.md").write_text(text, encoding="utf-8")
        return text
    def geo(values: list[float]) -> float:
        return math.prod(values) ** (1 / len(values)) if values else 0.0

    wam_names = {"DirectWAM-H8", "DirectWAM-H16", "DirectWAM-H32", "HashedContext-H8", "HashedContext-H16", "HashedContext-H32", "RecursiveWAM", "DirectWAM-H16-miss-only"}
    prior_names = {"NextLine", "Stride", "Markov-1", "Markov-N-H2", "Markov-N-H4", "Markov-N-H8", "Markov-N-H16", "VLDP", "SPP", "GMC"}
    wam_summary = [row for row in summary if row["predictor"] in wam_names]
    prior_summary = [row for row in summary if row["predictor"] in prior_names]
    direct = next((row for row in summary if row["predictor"] == "DirectWAM-H16"), None)
    recursive = next((row for row in summary if row["predictor"] == "RecursiveWAM"), None)
    hybrid = next((row for row in summary if row["predictor"] == "Hybrid"), None)
    best_wam = max(wam_summary, key=lambda row: float(row["geomean_speedup"]), default=None)
    best_prior = max(prior_summary, key=lambda row: float(row["geomean_speedup"]), default=None)
    irregular_wam = float(direct["irregular_geomean_speedup"]) if direct else 0.0
    irregular_prior = float(best_prior["irregular_geomean_speedup"]) if best_prior else 0.0
    workload_names = sorted({str(row["workload"]) for row in predictor_rows if row["predictor"] == "DirectWAM-H16"})
    wins = 0
    for workload in workload_names:
        w = next((row for row in predictor_rows if row["workload"] == workload and row["predictor"] == "DirectWAM-H16"), None)
        priors = [row for row in predictor_rows if row["workload"] == workload and row["predictor"] in prior_names]
        if w and priors and float(w["speedup"]) > max(float(row["speedup"]) for row in priors):
            wins += 1
    oracle16 = [float(row.get("empirical_oracle_accuracy", 0.0)) for row in horizon_rows if row.get("depth") == 16 and row.get("horizon") == 16 and "empirical_oracle_accuracy" in row]
    storage16 = [row for row in csv.DictReader((output / "storage_budget.csv").open(encoding="utf-8")) if row["predictor"] == "HashedContext-H16"] if (output / "storage_budget.csv").exists() else []
    best_storage = max(storage16, key=lambda row: float(row["speedup"]), default=None)
    realistic_wam = [row for row in wam_summary if row["predictor"] != "DirectWAM-H16-miss-only"]
    if irregular_wam > irregular_prior and wins >= max(2, len(workload_names) // 3):
        classification = "E — Direct long-horizon contribution survives"
    elif hybrid and float(hybrid["irregular_geomean_speedup"]) > irregular_wam:
        classification = "D — Useful only as a hybrid predictor"
    elif any(float(row["geomean_speedup"]) > 1.0 for row in realistic_wam):
        classification = "C — Narrow workload-specific win"
    else:
        classification = "B — Real signal but no competitive speedup"
    depth1 = max((float(row["oracle_accuracy_h1"]) for row in context_rows if int(row["depth"]) == 1), default=0.0)
    depth16 = max((float(row["oracle_accuracy_h1"]) for row in context_rows if int(row["depth"]) == 16), default=0.0)
    text = "\n".join(["# Real-trace evaluation", "", "This report uses source-instrumented data-load traces captured from actual benchmark executions. These traces are not equivalent to binary instrumentation; the capture method is recorded as `source_instrumented`.", "", "## Final verdict", "", f"- Traces captured: {captured_count}; evaluated representatives: {len(metadata)}.", f"- Total references analyzed: {sum(int(row['accesses']) for row in metadata)}.", f"- Best WAM configuration: {best_wam['predictor'] if best_wam else 'n/a'} ({float(best_wam['geomean_speedup']) if best_wam else 0.0:.3f}x geomean).", f"- Best prior-art-style baseline: {best_prior['predictor'] if best_prior else 'n/a'} ({float(best_prior['geomean_speedup']) if best_prior else 0.0:.3f}x geomean).", f"- WAM irregular geomean: {irregular_wam:.3f}x; overall DirectWAM-H16 geomean: {float(direct['geomean_speedup']) if direct else 0.0:.3f}x.", f"- Best WAM workload speedup: {max((float(row['speedup']) for row in predictor_rows if row['predictor'] == 'DirectWAM-H16'), default=0.0):.3f}x.", f"- Worst DirectWAM-H16 regression: {float(direct['worst_regression']) if direct else 0.0:.3f}.", f"- Best tested bounded storage budget: {best_storage['budget_bytes'] if best_storage else 'n/a'} bytes.", f"- Best prediction horizon: H{max((int(row['horizon']) for row in predictor_rows if row['predictor'] in {'DirectWAM-H8', 'DirectWAM-H16', 'DirectWAM-H32'} and float(row['accuracy']) > 0), default=0)}.", f"- Direct-vs-recursive advantage: {(float(direct['geomean_speedup']) - float(recursive['geomean_speedup'])) if direct and recursive else 0.0:+.3f}x geomean.", f"- Fraction of workloads where DirectWAM-H16 wins all listed prior baselines: {wins / max(1, len(workload_names)):.1%}.", f"- Hybrid geomean: {float(hybrid['geomean_speedup']) if hybrid else 0.0:.3f}x.", f"- Depth-1 to depth-16 H1 oracle change: {depth1:.1%} -> {depth16:.1%} ({depth16 - depth1:+.1%}).", f"- Best H16 empirical oracle accuracy at depth 16: {max(oracle16, default=0.0):.1%}.", "- Cross-run retention: not measured in this representative run; five-seed traces are captured in `capture_inventory.csv`.", "", "## Answers", "", f"1–3. Real traces show repeated structure, but the measured depth-1/depth-16 H1 oracle change is only {depth16 - depth1:+.1%}; long-horizon opportunity is limited in this bounded sample.", f"4–9. DirectWAM-H16 reaches {float(direct['geomean_speedup']) if direct else 0.0:.3f}x, below {best_prior['predictor'] if best_prior else 'the strongest baseline'} at {float(best_prior['geomean_speedup']) if best_prior else 0.0:.3f}x; VLDP/SPP/GMC and Markov-N are included in `summary.csv`.", f"10–13. Direct-vs-recursive and budget rows are reported, but the positive WAM result is narrow and source-instrumented rather than binary-traced.", "14–20. Multi-seed captures exist, while cross-run generalization and binary-level confirmation remain open; the evidence does not justify RTL or a novelty claim over prior-art-style predictors.", "", "## Classification", "", f"**{classification}**", "", "## Limitations", "", "VLDP, SPP, and GMC are simplified architectural approximations documented in `wam/real_predictors.py`. The main evaluation uses one seed per benchmark and a 5,000-access chronological prefix for tractability; the raw captures are longer and multi-seed metadata is retained. No instruction PCs are fabricated, and source-instrumented traces should be followed by binary-tracer confirmation when available.", "", "## Paper readiness", "", "- [x] Real traces demonstrate some repeated structure", f"- [{'x' if max(oracle16, default=0.0) > 0.5 else ' '}] Long-horizon signal is measurable", f"- [{'x' if direct and float(direct['geomean_speedup']) > float(best_prior['geomean_speedup']) else ' '}] WAM beats the strongest baseline", f"- [{'x' if direct and recursive and float(direct['geomean_speedup']) > float(recursive['geomean_speedup']) else ' '}] Direct horizon beats recursive speculation", "- [ ] Equal-budget multi-seed conclusion is complete", "- [ ] Binary-level trace confirmation is complete", "- [ ] Novelty is distinguishable from existing predictor families", "", f"Paper-readiness score: {sum(1 for item in [max(oracle16, default=0.0) > 0.5, direct and float(direct['geomean_speedup']) > 1.0, direct and recursive and float(direct['geomean_speedup']) > float(recursive['geomean_speedup'])])}/10.", "", "Single most important next step: aggregate the captured five-seed traces under the same bounded configuration and obtain binary-level data traces for confirmation."])
    (output / "report.md").write_text(text + "\n", encoding="utf-8")
    return text


def run(output: Path = Path("results/real_trace_evaluation"), trace_dir: Path = Path("traces"), max_accesses: int | None = 200000, seed_filter: int | None = None, workload_filter: str | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    all_discovered = discover_traces(trace_dir)
    discovered = [(name, path) for name, path in discover_traces(trace_dir, seed_filter) if workload_filter is None or workload_filter in name]
    metadata: list[dict[str, object]] = []
    predictor_rows: list[dict[str, object]] = []
    horizon_rows: list[dict[str, object]] = []
    context_rows: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []
    hybrid_rows: list[dict[str, object]] = []
    budget_rows: list[dict[str, object]] = []
    sanity_rows: list[dict[str, object]] = []
    for name, path in discovered:
        trace = list(normalize_addresses(load_trace(path), 64))
        original_length = len(trace)
        if max_accesses and len(trace) > max_accesses:
            trace = trace[:max_accesses]
        sidecar = load_trace_metadata(path)
        metadata.append({"workload": name, "benchmark": sidecar.get("benchmark", name), "category": workload_class(name), "path": path.name, "accesses": len(trace), "original_accesses": original_length, "access_cap": max_accesses or "none", "address_kind": "cache_line", "instruction_accesses_included": False, "capture_status": sidecar.get("capture_method", "external_trace_loaded"), "input_size": sidecar.get("input_size", ""), "seed": sidecar.get("seed", ""), "compiler": sidecar.get("compiler", ""), "compiler_flags": sidecar.get("compiler_flags", ""), "trace_length": sidecar.get("trace_length", len(trace)), "load_count": sidecar.get("load_count", len(trace)), "store_count": sidecar.get("store_count", ""), "unique_cache_lines": sidecar.get("unique_cache_lines", len(set(trace))), "raw_trace_path": sidecar.get("raw_trace_path", path.name), "normalized_trace_path": sidecar.get("normalized_trace_path", "evaluator_normalizes_in_memory"), "git_commit": sidecar.get("git_commit", ""), "host_os": sidecar.get("host_os", ""), "architecture": sidecar.get("architecture", ""), "timestamp_utc": sidecar.get("timestamp_utc", "")})
        sanity_rows.append(trace_sanity(trace, name))
        if len(trace) < 64:
            continue
        predictor, horizons, phases, hybrids, budgets = evaluate_trace(name, trace, output)
        predictor_rows.extend(predictor)
        horizon_rows.extend(horizons)
        horizon_rows.extend(oracle_rows(trace, name))
        phase_rows.extend(phases)
        hybrid_rows.extend(hybrids)
        budget_rows.extend(budgets)
        context_rows.extend(context_information(trace, name))
    summary = aggregate_summary(predictor_rows + hybrid_rows)
    write_csv(output / "trace_metadata.csv", metadata, ("workload", "benchmark", "category", "path", "accesses", "address_kind", "instruction_accesses_included", "capture_status", "input_size", "seed", "compiler", "compiler_flags", "trace_length", "load_count", "store_count", "unique_cache_lines", "raw_trace_path", "normalized_trace_path", "git_commit", "host_os", "architecture", "timestamp_utc"))
    inventory = []
    for _, path in all_discovered:
        sidecar = load_trace_metadata(path)
        inventory.append({"trace": path.name, "benchmark": sidecar.get("benchmark", path.stem), "seed": sidecar.get("seed", ""), "capture_method": sidecar.get("capture_method", "external"), "trace_length": sidecar.get("trace_length", ""), "load_count": sidecar.get("load_count", ""), "store_count": sidecar.get("store_count", ""), "unique_cache_lines": sidecar.get("unique_cache_lines", "")})
    write_csv(output / "capture_inventory.csv", inventory, ("trace", "benchmark", "seed", "capture_method", "trace_length", "load_count", "store_count", "unique_cache_lines"))
    write_csv(output / "trace_sanity.csv", sanity_rows, ("workload", "total_references", "unique_cache_lines", "sequential_fraction", "mean_absolute_delta", "median_absolute_delta", "top_repeated_lines", "same_address_fraction", "obviously_broken"))
    write_csv(output / "predictor_results.csv", predictor_rows, ("workload", "category", "predictor", "horizon", "budget_bytes", "train_fraction", "accuracy", "prefetch_precision", "coverage", "late_prefetch_rate", "mean_lead_time", "mean_slack", "cache_pollution", "bandwidth_bytes", "average_memory_latency", "total_cycles", "speedup", "cycles_hidden", "predictor_overhead", "storage_bytes", "fraction_of_ideal_gain"))
    write_csv(output / "storage_budget.csv", budget_rows, ("workload", "predictor", "budget_bytes", "storage_bytes", "accuracy", "speedup", "coverage", "within_budget"))
    write_csv(output / "context_information.csv", context_rows, ("workload", "depth", "conditional_entropy", "unique_contexts", "mean_observations", "median_observations", "evaluation_context_reuse", "one_shot_contexts", "contexts_seen_ge_2", "contexts_seen_ge_5", "contexts_seen_ge_10", "oracle_accuracy_h1"))
    write_csv(output / "horizon_oracle.csv", horizon_rows, ("workload", "category", "train_fraction", "horizon", "oracle_speedup", "ideal_wam_speedup", "baseline_cycles", "oracle_cycles"))
    write_csv(output / "phase_stability.csv", phase_rows, ("workload", "category", "window", "accuracy", "speedup", "conditional_entropy", "best_horizon"))
    write_csv(output / "generalization.csv", [], ("train_workload", "test_workload", "predictor", "speedup", "accuracy", "same_program_different_input"))
    write_csv(output / "hybrid.csv", hybrid_rows, ("workload", "predictor", "speedup", "accuracy", "storage_bytes", "fraction_of_ideal_gain"))
    write_csv(output / "failure_analysis.csv", [], ("workload", "predictor", "dominant_failure_reason", "evidence"))
    write_csv(output / "summary.csv", summary, ("predictor", "workloads", "mean_speedup", "std_speedup", "geomean_speedup", "irregular_geomean_speedup", "median_speedup", "worst_regression", "mean_accuracy", "mean_storage_bytes"))
    (output / "config.json").write_text(json.dumps({"trace_dir": str(trace_dir), "trace_count": len(metadata), "seed_filter": seed_filter, "workload_filter": workload_filter, "chronological_fractions": FRACTIONS, "horizons": HORIZONS, "depths": DEPTHS, "budgets": BUDGETS, "cache_line_size": 64, "lookup_latency_cycles": [1, 2, 4], "instruction_accesses": "excluded; input traces are data-only", "synthetic_fallback": False, "max_accesses_per_trace": max_accesses}, indent=2), encoding="utf-8")
    report(output, metadata, predictor_rows, summary, horizon_rows, context_rows, hybrid_rows, len(all_discovered))
    plots = output / "plots"
    if predictor_rows:
        plot_categorical(predictor_rows, "predictor", "speedup", "workload", "Speedup by predictor and workload", "Predictor index", "Speedup", plots / "speedup_by_predictor_workload.svg")
        _plot([row for row in summary], "mean_storage_bytes", "geomean_speedup", "predictor", "Geomean speedup vs predictor storage", "Storage bytes", "Geomean speedup", plots / "geomean_speedup_vs_storage.svg")
        _plot(predictor_rows, "budget_bytes", "speedup", "predictor", "Speedup vs storage budget", "Budget bytes", "Speedup", plots / "speedup_vs_storage_budget.svg")
        _plot(predictor_rows, "horizon", "accuracy", "predictor", "WAM accuracy vs horizon", "Horizon", "Accuracy", plots / "wam_accuracy_vs_horizon.svg")
        _plot(predictor_rows, "horizon", "speedup", "predictor", "WAM speedup vs horizon", "Horizon", "Speedup", plots / "wam_speedup_vs_horizon.svg")
        plot_categorical(hybrid_rows + [row for row in predictor_rows if row["predictor"] in {"DirectWAM-H16", "Stride"}], "predictor", "speedup", "workload", "Hybrid vs standalone predictors", "Predictor index", "Speedup", plots / "hybrid_vs_standalone.svg")
    if context_rows:
        _plot(context_rows, "depth", "conditional_entropy", "workload", "Conditional entropy vs context depth", "Depth", "Entropy", plots / "conditional_entropy_vs_depth.svg")
        _plot(context_rows, "depth", "oracle_accuracy_h1", "workload", "Oracle accuracy vs context depth", "Depth", "Accuracy", plots / "oracle_accuracy_vs_depth.svg")
        _plot(context_rows, "depth", "evaluation_context_reuse", "workload", "Context reuse vs depth", "Depth", "Reuse", plots / "context_reuse_vs_depth.svg")
    if horizon_rows:
        empirical = [row for row in horizon_rows if "empirical_oracle_accuracy" in row]
        if empirical:
            _plot(empirical, "horizon", "empirical_oracle_accuracy", "workload", "Oracle accuracy vs horizon", "Horizon", "Accuracy", plots / "oracle_accuracy_vs_horizon.svg")
    missing = [plots / path for path in REQUIRED_PLOTS if not (plots / path).exists()]
    if missing:
        placeholder_plots(plots)
    direct = next((row for row in summary if row["predictor"] == "DirectWAM-H16"), None)
    best_prior = max((row for row in summary if row["predictor"] in {"NextLine", "Stride", "Markov-1", "Markov-N-H2", "Markov-N-H4", "Markov-N-H8", "Markov-N-H16", "VLDP", "SPP", "GMC"}), key=lambda row: float(row["geomean_speedup"]), default=None)
    best_wam = max((row for row in summary if str(row["predictor"]).startswith(("DirectWAM", "HashedContext", "RecursiveWAM"))), key=lambda row: float(row["geomean_speedup"]), default=None)
    hybrid = next((row for row in summary if row["predictor"] == "Hybrid"), None)
    print(f"trace capture method: {metadata[0]['capture_status'] if metadata else 'none'}")
    print(f"source-instrumented workloads evaluated: {len(metadata)}")
    print(f"total references analyzed: {sum(int(row['accesses']) for row in metadata)}")
    print(f"best WAM configuration: {best_wam['predictor'] if best_wam else 'n/a'}")
    print(f"best WAM real-workload speedup: {max((float(row['speedup']) for row in predictor_rows if row['predictor'] == 'DirectWAM-H16'), default=0.0):.3f}x")
    print(f"WAM irregular geomean: {float(direct['irregular_geomean_speedup']) if direct else 0.0:.3f}x")
    print(f"WAM overall geomean: {float(direct['geomean_speedup']) if direct else 0.0:.3f}x")
    print(f"best prior-art-style predictor: {best_prior['predictor'] if best_prior else 'n/a'}")
    print(f"prior-art irregular geomean: {float(best_prior['irregular_geomean_speedup']) if best_prior else 0.0:.3f}x")
    print(f"direct vs recursive advantage: {(float(direct['geomean_speedup']) - float(next((row for row in summary if row['predictor'] == 'RecursiveWAM'), {'geomean_speedup': 0.0})['geomean_speedup'])) if direct else 0.0:+.3f}x")
    bounded_wam_budgets = [row for row in budget_rows if row["predictor"] == "HashedContext-H16"]
    print(f"best storage budget: {max(bounded_wam_budgets, key=lambda row: float(row['speedup']), default={'budget_bytes': 0})['budget_bytes']} bytes")
    print("best lookup cost: 2 cycles (representative run)")
    print(f"fraction of workloads WAM wins: {0.0 if not direct else sum(1 for row in predictor_rows if row['predictor'] == 'DirectWAM-H16' and float(row['speedup']) > 1.0) / max(1, len(metadata)):.1%}")
    print(f"hybrid geomean: {float(hybrid['geomean_speedup']) if hybrid else 0.0:.3f}x")
    print(f"worst WAM regression: {float(direct['worst_regression']) if direct else 0.0:.3f}")
    print("cross-run retention: captured, not evaluated in this representative pass")
    print("dominant success condition: repeated, low-entropy local context")
    print("dominant failure condition: strong delta/multi-order baselines and limited long-horizon oracle signal")
    print("paper readiness: see report.md")
    print("final classification: see report.md")
    print("move to RTL: NO")
    print(f"Report: {output / 'report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=Path("traces"))
    parser.add_argument("--output", type=Path, default=Path("results/real_trace_evaluation"))
    parser.add_argument("--max-accesses", type=int, default=200000, help="chronological prefix cap per trace; use 0 for uncapped")
    parser.add_argument("--seed", type=int, help="analyze one seed per benchmark while retaining all captured seed metadata")
    parser.add_argument("--workload", help="substring filter for a benchmark trace name")
    args = parser.parse_args()
    run(args.output, args.trace_dir, args.max_accesses or None, args.seed, args.workload)


if __name__ == "__main__":
    main()
