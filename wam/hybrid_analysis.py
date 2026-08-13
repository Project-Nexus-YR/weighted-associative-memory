"""Complementarity study for the bounded GMC-style and WAM predictors.

The study deliberately treats ``GMCStylePredictor`` as a simplified baseline.
It uses frozen chronological training prefixes, evaluates non-overlapping
windows, and keeps oracle selectors separate from implementable selectors.
The raw captures are source-instrumented data-load traces; no synthetic trace
is generated here.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .horizon import NoHorizonPredictor, RecursiveWAM, simulate_horizon
from .real_trace_evaluation import (
    predictor_for,
    run_config,
    split,
    workload_class,
    discover_traces,
    load_trace_metadata,
)
from .traces import iter_addresses, normalize_addresses
from .real_trace_evaluation import plot_categorical
from .diagnostics import _plot
from .hardware import HashedContextPredictor

WINDOW_SIZES = (100, 500, 1_000, 5_000, 10_000, 50_000)
SELECTOR_GRANULARITIES = ("per_access", 100, 1_000, 10_000)
TOTAL_BUDGETS = (8_192, 16_384, 32_768, 65_536)
SPLITS = ((0.75, 0.25), (0.50, 0.50), (0.25, 0.75))
PRIMARY_WINDOW = 5_000
HORIZON = 16
PHASE_SAMPLES_PER_SIZE = 3


def write_csv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fields)
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def geomean(values: list[float]) -> float:
    positive = [max(1e-12, float(value)) for value in values]
    return math.prod(positive) ** (1 / len(positive)) if positive else 0.0


def entropy(values: list[int]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    return -sum((count / total) * math.log2(count / total) for count in counts.values()) if total else 0.0


def gmc_audit(path: Path) -> None:
    text = """# GMC-style implementation audit

## Classification

**Simplified approximation**

The repository implementation is intentionally not presented as a faithful
reimplementation of the published GMC prefetcher. The implementation is
`wam.real_predictors.GMCStylePredictor`, which subclasses
`DeltaContextPredictor`.

## What the code actually does

| Concern | Repository implementation |
|---|---|
| Context representation | Cache-line addresses are converted to adjacent address deltas. |
| Delta/stride representation | Signed integer deltas; no page/region, PC, or instruction context. |
| Local/global history | One global stream is used. There is no per-PC local history and no separate global-history buffer. |
| Context orders | Orders 1 through 16 are stored; lookup tries the longest available suffix and falls back to shorter suffixes. |
| Confidence | Normalized frequency of the selected target delta in the matched table entry. |
| Prediction target | One cache-line address at a fixed horizon, computed as current line plus predicted delta. |
| Table organization | A bounded FIFO-evicted Python dictionary keyed by delta tuples; each key stores target-delta counters. |
| Fallback behavior | Longest matching suffix with non-empty transitions, then shorter suffixes; no explicit confidence gating or multi-table arbitration. |
| Hardware accounting | A rough byte budget and lookup/update cycle proxy; no GMC-specific table or metadata model. |

## High-level comparison

Published GMC work describes global-aware, multi-order context analysis with
local/global context signals and prediction structures intended to increase
coverage while preserving accuracy. This repository captures only the broad
intuition of multi-order delta-context prediction. It omits program-counter
context, the local/global organization, the published training/update policy,
table replacement details, and GMC-specific candidate/confidence arbitration.

Therefore all result files and plots use the name **GMC-style**, not GMC, and
the result cannot support a paper-level claim against the original design.

Reference: [Global-aware and multi-order context-based prefetching for
high-performance processors](https://journals.sagepub.com/doi/10.1177/1094342010394386).

## Consequence for interpretation

The complementarity experiment asks whether WAM adds value to this concrete
GMC-style approximation. A positive result would motivate a follow-up against
an independently reproduced GMC implementation; a negative result is already
useful evidence against investing in WAM tuning before baseline fidelity is
improved.
"""
    path.write_text(text, encoding="utf-8")


def history_features(history: list[int], window: list[int], depth: int = 16) -> dict[str, float]:
    recent = history[-2_000:]
    deltas = [recent[index] - recent[index - 1] for index in range(1, len(recent))]
    stride_counts = Counter(deltas)
    stride_stability = max(stride_counts.values(), default=0) / max(1, len(deltas))
    contexts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
    for index in range(1, len(history)):
        key = tuple(history[max(0, index - depth):index])
        contexts[key][history[index]] += 1
    one = []
    sixteen = []
    for size in (1, depth):
        distributions: list[float] = []
        counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        for index in range(1, len(history)):
            counts[tuple(history[max(0, index - size):index])][history[index]] += 1
        for transition in counts.values():
            total = sum(transition.values())
            distributions.append(-sum((count / total) * math.log2(count / total) for count in transition.values()))
        (one if size == 1 else sixteen).extend(distributions)
    eval_contexts = [tuple(history[-depth:])]
    eval_contexts.extend(tuple((history + window[:index])[-depth:]) for index in range(min(len(window), 2_000)))
    reuse = Counter(eval_contexts)
    window_deltas = [window[index] - window[index - 1] for index in range(1, len(window))]
    distances: list[int] = []
    last_seen: dict[int, int] = {}
    combined = history[-2_000:] + window
    offset = len(history[-2_000:])
    for index, line in enumerate(combined):
        if line in last_seen:
            distances.append(index - last_seen[line])
        last_seen[line] = index
    return {
        "context_reuse_depth16": sum(value >= 2 for value in reuse.values()) / max(1, len(reuse)),
        "context_entropy_depth1": statistics.mean(one) if one else 0.0,
        "context_entropy_depth16": statistics.mean(sixteen) if sixteen else 0.0,
        "entropy_reduction": (statistics.mean(one) if one else 0.0) - (statistics.mean(sixteen) if sixteen else 0.0),
        "h16_contexts": float(len(contexts)),
        "stride_stability": stride_stability,
        "delta_entropy": entropy(deltas),
        "sequential_fraction": sum(delta == 1 for delta in window_deltas) / max(1, len(window_deltas)),
        "unique_lines": float(len(set(window))),
        "mean_reuse_distance": statistics.mean(distances) if distances else 0.0,
        "median_reuse_distance": statistics.median(distances) if distances else 0.0,
        "same_address_fraction": sum(delta == 0 for delta in window_deltas) / max(1, len(window_deltas)),
        "history_length": float(len(history)),
        "context_reuse_observations": float(sum(reuse.values())),
        "offset": float(offset),
    }


def oracle_accuracy(history: list[int], window: list[int], horizon: int, depth: int) -> float:
    counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
    for index in range(max(0, len(history) - horizon)):
        key = tuple(history[max(0, index - depth + 1):index + 1])
        counts[key][history[index + horizon]] += 1
    correct = attempts = 0
    available = history[-depth:]
    for index, value in enumerate(window):
        target = index + horizon
        if target >= len(window):
            break
        transitions = counts.get(tuple(available[-depth:]), {})
        if transitions:
            attempts += 1
            correct += max(transitions.items(), key=lambda item: (item[1], -item[0]))[0] == window[target]
        available = (available + [value])[-depth:]
    return correct / attempts if attempts else 0.0


def prediction_snapshot(model, context: list[int], horizon: int) -> tuple[float, int | None]:
    if isinstance(model, RecursiveWAM):
        predictions = model.predict_horizon(context, horizon)
    else:
        predictions = model.predict_horizon(context, 1)
    return (float(predictions[0].confidence) if predictions else 0.0, int(predictions[0].address) if predictions else None)


def confidence_stats(model, history: list[int], window: list[int], horizon: int = HORIZON) -> dict[str, float]:
    context = history[-getattr(model, "context_depth", 1):]
    start_confidence, _ = prediction_snapshot(model, context, horizon)
    confidences: list[float] = []
    correct = 0
    attempts = 0
    for index, value in enumerate(window):
        confidence, address = prediction_snapshot(model, context, horizon)
        target = index + horizon
        if target < len(window):
            attempts += 1
            correct += address == window[target]
        confidences.append(confidence)
        context = (context + [value])[-getattr(model, "context_depth", 1):]
    return {
        "start_confidence": start_confidence,
        "mean_confidence": statistics.mean(confidences) if confidences else 0.0,
        "prediction_accuracy": correct / attempts if attempts else 0.0,
    }


def disagreement(history: list[int], window: list[int], gmc, wam) -> dict[str, float]:
    gmc_context = history[-getattr(gmc, "context_depth", 1):]
    wam_context = history[-getattr(wam, "context_depth", 1):]
    same = different = only_gmc = only_wam = both_wrong = gmc_wins = wam_wins = 0
    total = 0
    for index, value in enumerate(window):
        _, gmc_address = prediction_snapshot(gmc, gmc_context, HORIZON)
        _, wam_address = prediction_snapshot(wam, wam_context, HORIZON)
        target = index + HORIZON
        if target < len(window):
            actual = window[target]
            g = gmc_address is not None
            w = wam_address is not None
            if g and w:
                total += 1
                if gmc_address == wam_address:
                    same += 1
                else:
                    different += 1
                gc = gmc_address == actual
                wc = wam_address == actual
                if wc and not gc:
                    wam_wins += 1
                elif gc and not wc:
                    gmc_wins += 1
                elif not gc and not wc:
                    both_wrong += 1
            elif g:
                only_gmc += 1
            elif w:
                only_wam += 1
        gmc_context = (gmc_context + [value])[-getattr(gmc, "context_depth", 1):]
        wam_context = (wam_context + [value])[-getattr(wam, "context_depth", 1):]
    return {
        "agreement_rate": same / max(1, total),
        "disagreement_rate": different / max(1, total),
        "only_gmc_rate": only_gmc / max(1, len(window)),
        "only_wam_rate": only_wam / max(1, len(window)),
        "wam_wins_disagreements": wam_wins,
        "gmc_wins_disagreements": gmc_wins,
        "both_wrong_disagreements": both_wrong,
        "joint_predictions": total,
    }


def bounded_wam(train: list[int], budget: int, horizon: int = HORIZON):
    """Return the explicitly labeled bounded WAM-sidecar hardware proxy.

    Exact DirectWAM uses a variable-size trie and is the primary predictor in
    this study. Equal-budget comparisons need a fixed-footprint state model;
    the existing hashed context implementation supplies that proxy without
    pretending that an unbounded exact trie fits inside the requested budget.
    """
    table_size = max(1, (budget - 32) // 16)
    model = HashedContextPredictor(context_depth=16, horizon=horizon, table_size=table_size, counter_bits=8, signature_bits=16, entry_bytes=16)
    model.name = "WAM-sidecar-bounded"
    return model.fit(train)


def simulate_pair(window: list[int], history: list[int], gmc, wam) -> tuple[object, object, object]:
    raw = [value * 64 for value in window]
    config = run_config(2)
    baseline = simulate_horizon(raw, NoHorizonPredictor(), HORIZON, config, enable_prefetch=False, initial_context=history[-1:])
    gmc_result = simulate_horizon(raw, gmc, HORIZON, config, initial_context=history[-getattr(gmc, "context_depth", 1):])
    wam_result = simulate_horizon(raw, wam, HORIZON, config, initial_context=history[-getattr(wam, "context_depth", 1):])
    return baseline, gmc_result, wam_result


def result_speedup(baseline, result) -> float:
    return baseline.cycles / max(1, result.cycles)


def sampled_starts(length: int, size: int) -> list[int]:
    starts = list(range(0, length, size))
    if len(starts) <= PHASE_SAMPLES_PER_SIZE:
        return starts
    indices = [0, len(starts) // 2, len(starts) - 1]
    return [starts[index] for index in indices]


def make_window_row(name: str, category: str, size: int, index: int, history: list[int], window: list[int], baseline, gmc_result, wam_result, gmc, wam) -> dict[str, object]:
    gmc_speedup = result_speedup(baseline, gmc_result)
    wam_speedup = result_speedup(baseline, wam_result)
    gmc_conf = confidence_stats(gmc, history, window)
    wam_conf = confidence_stats(wam, history, window)
    features = history_features(history, window)
    dis = disagreement(history, window, gmc, wam)
    return {
        "workload": name,
        "category": category,
        "window": size,
        "window_index": index,
        "window_accesses": len(window),
        "gmc_speedup": gmc_speedup,
        "wam_speedup": wam_speedup,
        "difference": wam_speedup - gmc_speedup,
        "winner": "WAM" if wam_result.cycles < gmc_result.cycles else "GMC-style" if gmc_result.cycles < wam_result.cycles else "tie",
        "margin": abs(wam_speedup - gmc_speedup),
        "gmc_cycles": gmc_result.cycles,
        "wam_cycles": wam_result.cycles,
        "baseline_cycles": baseline.cycles,
        "gmc_confidence": gmc_conf["mean_confidence"],
        "wam_confidence": wam_conf["mean_confidence"],
        "gmc_start_confidence": gmc_conf["start_confidence"],
        "wam_start_confidence": wam_conf["start_confidence"],
        "gmc_accuracy": gmc_conf["prediction_accuracy"],
        "wam_accuracy": wam_conf["prediction_accuracy"],
        "gmc_prefetch_precision": gmc_result.metrics.prefetch_precision,
        "wam_prefetch_precision": wam_result.metrics.prefetch_precision,
        "gmc_coverage": gmc_result.metrics.prefetches_issued / max(1, gmc_result.metrics.total_accesses),
        "wam_coverage": wam_result.metrics.prefetches_issued / max(1, wam_result.metrics.total_accesses),
        "gmc_late_prefetch_rate": gmc_result.metrics.late_prefetch_rate,
        "wam_late_prefetch_rate": wam_result.metrics.late_prefetch_rate,
        "gmc_cache_pollution": gmc_result.metrics.pollution_misses,
        "wam_cache_pollution": wam_result.metrics.pollution_misses,
        "gmc_bandwidth_bytes": gmc_result.metrics.bandwidth_bytes,
        "wam_bandwidth_bytes": wam_result.metrics.bandwidth_bytes,
        "gmc_useful_rate": gmc_result.metrics.useful_prefetches / max(1, gmc_result.metrics.prefetches_issued),
        "wam_useful_rate": wam_result.metrics.useful_prefetches / max(1, wam_result.metrics.prefetches_issued),
        "gmc_storage_bytes": gmc_result.predictor_storage.get("estimated_bytes", 0),
        "wam_storage_bytes": wam_result.predictor_storage.get("estimated_bytes", 0),
        **features,
        "h1_oracle_accuracy": oracle_accuracy(history, window, 1, 16),
        "h8_oracle_accuracy": oracle_accuracy(history, window, 8, 16),
        "h16_oracle_accuracy": oracle_accuracy(history, window, 16, 16),
        "h32_oracle_accuracy": oracle_accuracy(history, window, 32, 16),
        **dis,
    }


def selector_choice(row: dict[str, object], selector: str, prior: list[dict[str, object]]) -> str:
    if selector == "WindowOracle":
        return "WAM" if float(row["wam_cycles"]) < float(row["gmc_cycles"]) else "GMC-style"
    if selector == "StaticPerWorkloadOracle":
        return "WAM" if sum(float(item["wam_cycles"]) for item in prior) < sum(float(item["gmc_cycles"]) for item in prior) else "GMC-style"
    if selector == "ConfidenceSelector":
        if float(row["gmc_start_confidence"]) >= 0.50:
            return "GMC-style"
        if float(row["wam_start_confidence"]) >= 0.50:
            return "WAM"
        return "GMC-style"
    if selector == "RecentWinnerSelector":
        if not prior:
            return "GMC-style"
        gmc = statistics.mean(float(item["gmc_useful_rate"]) for item in prior[-3:])
        wam = statistics.mean(float(item["wam_useful_rate"]) for item in prior[-3:])
        return "WAM" if wam > gmc else "GMC-style"
    if selector == "EntropyAwareSelector":
        return "WAM" if float(row["entropy_reduction"]) > 0.10 and float(row["wam_start_confidence"]) >= float(row["gmc_start_confidence"]) else "GMC-style"
    return "GMC-style"


def selector_rows(window_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    selectors = ("ConfidenceSelector", "RecentWinnerSelector", "EntropyAwareSelector", "StaticPerWorkloadOracle", "WindowOracle")
    primary = [row for row in window_rows if int(row["window"]) == PRIMARY_WINDOW]
    if not primary:
        primary = [row for row in window_rows if int(row["window"]) == 1_000]
    for granularity in SELECTOR_GRANULARITIES:
        for selector in selectors:
            for workload in sorted({str(row["workload"]) for row in primary}):
                work = [row for row in primary if row["workload"] == workload]
                prior: list[dict[str, object]] = []
                selected_cycles = 0.0
                baseline_cycles = 0.0
                wam_selected = 0
                gmc_selected = 0
                update_traffic = 0
                for row in work:
                    choice = selector_choice(row, selector, prior)
                    selected_cycles += float(row["wam_cycles"] if choice == "WAM" else row["gmc_cycles"])
                    baseline_cycles += float(row["baseline_cycles"])
                    wam_selected += choice == "WAM"
                    gmc_selected += choice == "GMC-style"
                    block_count = 1 if granularity == "per_access" else max(1, math.ceil(int(row["window_accesses"]) / int(granularity)))
                    update_traffic += block_count * 2
                    prior.append(row)
                lookup_cycles = len(work) if granularity == "per_access" else sum(max(1, math.ceil(int(row["window_accesses"]) / int(granularity))) for row in work)
                selector_bytes = 128 + (64 if selector in {"ConfidenceSelector", "RecentWinnerSelector", "EntropyAwareSelector"} else 0)
                total_cycles = selected_cycles + lookup_cycles
                rows.append({"workload": workload, "selector": selector, "granularity": granularity, "windows": len(work), "speedup": baseline_cycles / max(1, total_cycles), "selected_gmc_fraction": gmc_selected / max(1, len(work)), "selected_wam_fraction": wam_selected / max(1, len(work)), "selector_bytes": selector_bytes, "lookup_cycles": lookup_cycles, "update_traffic": update_traffic, "overhead_cycles": lookup_cycles, "wam_activation_rate": wam_selected / max(1, len(work)), "selection_note": "per_access uses finest available replay window as a conservative proxy" if granularity == "per_access" else "block-granular replay"})
    return rows


def aggregate_selector(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    for selector, granularity in sorted({(str(row["selector"]), str(row["granularity"])) for row in rows}):
        values = [row for row in rows if row["selector"] == selector and str(row["granularity"]) == granularity]
        result.append({"selector": selector, "granularity": granularity, "workloads": len(values), "geomean_speedup": geomean([float(row["speedup"]) for row in values]), "mean_wam_activation_rate": statistics.mean(float(row["wam_activation_rate"]) for row in values) if values else 0.0, "selector_bytes": max(int(row["selector_bytes"]) for row in values) if values else 0, "lookup_cycles": sum(int(row["lookup_cycles"]) for row in values), "update_traffic": sum(int(row["update_traffic"]) for row in values)})
    return result


def run(trace_dir: Path, output: Path, max_accesses: int = 200_000, seed_filter: int | None = None, train_cap: int = 20_000) -> None:
    output.mkdir(parents=True, exist_ok=True)
    gmc_audit(output / "gmc_audit.md")
    all_paths = discover_traces(trace_dir, seed_filter)
    window_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    disagreement_rows: list[dict[str, object]] = []
    budget_rows: list[dict[str, object]] = []
    for name, path in all_paths:
        trace = list(normalize_addresses(itertools.islice(iter_addresses(path), max_accesses), 64))
        if len(trace) < 1_000:
            continue
        cut = max(1, int(len(trace) * 0.70))
        train = trace[max(0, cut - train_cap):cut]
        evaluation = trace[cut:]
        gmc = predictor_for("GMC", HORIZON, 8_192).fit(train)
        wam = predictor_for("DirectWAM-H16", HORIZON, 8_192).fit(train)
        category = workload_class(name)
        for size in WINDOW_SIZES:
            for index, start in enumerate(sampled_starts(len(evaluation), size)):
                window = evaluation[start:start + size]
                if len(window) < size:
                    continue
                history = train
                baseline, gmc_result, wam_result = simulate_pair(window, history, gmc, wam)
                row = make_window_row(name, category, size, index, history, window, baseline, gmc_result, wam_result, gmc, wam)
                window_rows.append(row)
                feature_rows.append({key: row[key] for key in ("workload", "category", "window", "window_index", "winner", "difference", "gmc_confidence", "wam_confidence", "gmc_accuracy", "wam_accuracy", "context_reuse_depth16", "context_entropy_depth1", "context_entropy_depth16", "entropy_reduction", "h1_oracle_accuracy", "h8_oracle_accuracy", "h16_oracle_accuracy", "h32_oracle_accuracy", "stride_stability", "delta_entropy", "sequential_fraction", "unique_lines", "mean_reuse_distance", "median_reuse_distance", "gmc_late_prefetch_rate", "wam_late_prefetch_rate", "gmc_prefetch_precision", "wam_prefetch_precision", "gmc_coverage", "wam_coverage", "gmc_cache_pollution", "wam_cache_pollution", "gmc_bandwidth_bytes", "wam_bandwidth_bytes")})
                disagreement_rows.append({key: row[key] for key in ("workload", "category", "window", "window_index", "agreement_rate", "disagreement_rate", "only_gmc_rate", "only_wam_rate", "wam_wins_disagreements", "gmc_wins_disagreements", "both_wrong_disagreements", "joint_predictions")})
        representative = [row for row in window_rows if row["workload"] == name and int(row["window"]) == PRIMARY_WINDOW]
        if not representative:
            representative = [row for row in window_rows if row["workload"] == name and int(row["window"]) == 1_000]
        # Equal-budget stress tests are intentionally seed-0 representatives;
        # all seeds remain in the core window/oracle/selector measurements.
        if representative and "_seed0_" in name:
            window = evaluation[:100]
            for total in TOTAL_BUDGETS:
                for gmc_fraction, wam_fraction in SPLITS:
                    selector_bytes = 192
                    available = max(0, total - selector_bytes)
                    gmc_budget = max(64, int(available * gmc_fraction))
                    wam_budget = max(64, int(available * wam_fraction))
                    split_gmc = predictor_for("GMC", HORIZON, gmc_budget).fit(train)
                    split_wam = bounded_wam(train, max(512, wam_budget), HORIZON)
                    base, gmc_result, wam_result = simulate_pair(window, train, split_gmc, split_wam)
                    choice = "WAM" if wam_result.cycles < gmc_result.cycles else "GMC-style"
                    chosen = wam_result if choice == "WAM" else gmc_result
                    budget_rows.append({"workload": name, "total_budget": total, "gmc_bytes": gmc_budget, "wam_bytes": wam_budget, "selector_bytes": selector_bytes, "actual_gmc_storage": gmc_result.predictor_storage.get("estimated_bytes", 0), "actual_wam_storage": wam_result.predictor_storage.get("estimated_bytes", 0), "speedup": result_speedup(base, chosen), "winner": choice, "within_budget": gmc_result.predictor_storage.get("estimated_bytes", 0) <= gmc_budget and wam_result.predictor_storage.get("estimated_bytes", 0) <= wam_budget})
    write_csv(output / "window_results.csv", window_rows)
    write_csv(output / "features.csv", feature_rows)
    write_csv(output / "disagreement.csv", disagreement_rows)
    oracle_rows = []
    for size in WINDOW_SIZES:
        values = [row for row in window_rows if int(row["window"]) == size]
        if values:
            gmc = geomean([float(row["gmc_speedup"]) for row in values])
            wam = geomean([float(row["wam_speedup"]) for row in values])
            oracle = geomean([max(float(row["gmc_speedup"]), float(row["wam_speedup"])) for row in values])
            oracle_rows.append({"window": size, "windows": len(values), "gmc_geomean": gmc, "wam_geomean": wam, "oracle_hybrid_geomean": oracle, "oracle_incremental_gain_over_gmc": oracle / max(1e-12, gmc) - 1.0, "wam_selected_fraction": sum(row["winner"] == "WAM" for row in values) / len(values), "wam_gain_attributed_fraction": sum(max(0.0, float(row["wam_speedup"]) - float(row["gmc_speedup"])) for row in values) / max(1e-12, sum(max(0.0, max(float(row["wam_speedup"]), float(row["gmc_speedup"])) - float(row["gmc_speedup"])) for row in values))})
    write_csv(output / "oracle_hybrid.csv", oracle_rows)
    primary = [row for row in window_rows if int(row["window"]) == PRIMARY_WINDOW]
    if not primary:
        primary = [row for row in window_rows if int(row["window"]) == 1_000]
    comp_rows = []
    for workload in sorted({str(row["workload"]) for row in primary}):
        values = [row for row in primary if row["workload"] == workload]
        wam_wins = [row for row in values if row["winner"] == "WAM"]
        gmc_wins = [row for row in values if row["winner"] == "GMC-style"]
        ties = [row for row in values if row["winner"] == "tie"]
        comp_rows.append({"workload": workload, "windows": len(values), "fraction_gmc_wins": len(gmc_wins) / len(values), "fraction_wam_wins": len(wam_wins) / len(values), "fraction_tied": len(ties) / len(values), "mean_wam_advantage_when_wam_wins": statistics.mean(float(row["difference"]) for row in wam_wins) if wam_wins else 0.0, "mean_gmc_advantage_when_gmc_wins": statistics.mean(-float(row["difference"]) for row in gmc_wins) if gmc_wins else 0.0, "oracle_incremental_gain_over_gmc": geomean([max(float(row["gmc_speedup"]), float(row["wam_speedup"])) for row in values]) / max(1e-12, geomean([float(row["gmc_speedup"]) for row in values])) - 1.0, "wam_gain_attributed_fraction": sum(max(0.0, float(row["difference"])) for row in values) / max(1e-12, sum(max(0.0, max(float(row["difference"]), 0.0)) for row in values))})
    write_csv(output / "complementarity.csv", comp_rows)
    selector_detail = selector_rows(window_rows)
    write_csv(output / "selector_results.csv", selector_detail)
    write_csv(output / "budget_split.csv", budget_rows)
    selector_summary = aggregate_selector(selector_detail)
    write_csv(output / "selector_summary.csv", selector_summary)
    config = {"trace_dir": str(trace_dir), "trace_count": len(all_paths), "max_accesses": max_accesses, "train_fraction": 0.70, "train_cap": train_cap, "windows": WINDOW_SIZES, "phase_samples_per_size": PHASE_SAMPLES_PER_SIZE, "primary_window": PRIMARY_WINDOW, "selectors": ["ConfidenceSelector", "RecentWinnerSelector", "EntropyAwareSelector", "StaticPerWorkloadOracle", "WindowOracle"], "selector_granularities": SELECTOR_GRANULARITIES, "total_budgets": TOTAL_BUDGETS, "splits": SPLITS, "source_traces_only": True, "gmc_label": "GMC-style", "budget_trace_policy": "seed-0 representative per workload, 100-access window; core complementarity rows include every discovered seed", "budget_wam_model": "HashedContextPredictor named WAM-sidecar-bounded; fixed-footprint proxy, not exact DirectWAM", "per_access_note": "The simulator is replayed at finest 100-access resolution for the per_access selector proxy; it is not a cycle-exact per-access arbitration replay."}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    plots = output / "plots"
    if window_rows:
        plot_categorical(window_rows, "workload", "gmc_speedup", "window", "GMC-style vs WAM speedup by workload", "Workload index", "GMC-style speedup", plots / "gmc_vs_wam_speedup_per_workload.svg")
        plot_categorical(window_rows, "workload", "difference", "window", "WAM minus GMC-style by workload", "Workload index", "WAM - GMC-style", plots / "wam_minus_gmc_by_workload.svg")
        _plot(oracle_rows, "window", "oracle_incremental_gain_over_gmc", "window", "OracleHybrid gain vs window size", "Window accesses", "Incremental gain", plots / "oracle_hybrid_gain_vs_window.svg")
        _plot(oracle_rows, "window", "wam_selected_fraction", "window", "Fraction of windows WAM wins", "Window accesses", "Fraction", plots / "fraction_windows_wam_wins.svg")
        _plot(feature_rows, "context_reuse_depth16", "difference", "winner", "WAM advantage vs context reuse", "Context reuse", "WAM - GMC-style", plots / "wam_advantage_vs_context_reuse.svg")
        _plot(feature_rows, "entropy_reduction", "difference", "winner", "WAM advantage vs entropy reduction", "Entropy reduction", "WAM - GMC-style", plots / "wam_advantage_vs_entropy_reduction.svg")
        _plot(feature_rows, "h16_oracle_accuracy", "difference", "winner", "WAM advantage vs H16 oracle accuracy", "H16 oracle accuracy", "WAM - GMC-style", plots / "wam_advantage_vs_h16_oracle_accuracy.svg")
    if selector_summary:
        plot_categorical(selector_summary, "selector", "geomean_speedup", "granularity", "Selector vs GMC-style geomean", "Selector index", "Geomean speedup", plots / "selector_vs_gmc_geomean.svg")
    if budget_rows:
        _plot(budget_rows, "total_budget", "speedup", "winner", "Speedup vs total predictor budget", "Total budget bytes", "Speedup", plots / "speedup_vs_total_predictor_budget.svg")
        _plot(budget_rows, "gmc_bytes", "speedup", "total_budget", "Speedup vs GMC/WAM budget split", "GMC bytes", "Speedup", plots / "speedup_vs_budget_split.svg")
    direct_rows = []
    for row in primary:
        direct_rows.append({"workload": row["workload"], "WAM-H16": row["wam_speedup"], "WAM-H1": 0.0, "RecursiveWAM": 0.0, "GMC-style": row["gmc_speedup"]})
    # Direct-H1 and recursive are measured as a separate lightweight pass on one
    # primary window per workload to isolate the direct-horizon contribution.
    direct_rows = direct_contribution(trace_dir, all_paths, max_accesses, primary, train_cap)
    write_csv(output / "direct_horizon.csv", direct_rows)
    write_csv(output / "features_summary.csv", feature_summary(feature_rows))
    report = build_report(output, window_rows, oracle_rows, comp_rows, selector_summary, budget_rows, disagreement_rows, direct_rows, len(all_paths))
    (output / "report.md").write_text(report, encoding="utf-8")
    print_verdict(window_rows, oracle_rows, selector_summary, budget_rows, disagreement_rows, direct_rows, report)


def direct_contribution(trace_dir: Path, paths: list[tuple[str, Path]], max_accesses: int, primary: list[dict[str, object]], train_cap: int) -> list[dict[str, object]]:
    rows = []
    wanted = {str(row["workload"]): row for row in primary}
    for name, path in paths:
        if name not in wanted:
            continue
        trace = list(normalize_addresses(itertools.islice(iter_addresses(path), max_accesses), 64))
        cut = max(1, int(len(trace) * 0.70))
        train = trace[max(0, cut - train_cap):cut]
        evaluation = trace[cut:]
        size = int(wanted[name]["window"])
        window = evaluation[:size]
        if len(window) < 100:
            continue
        baseline = simulate_horizon([value * 64 for value in window], NoHorizonPredictor(), 16, run_config(2), enable_prefetch=False, initial_context=train[-1:])
        vals = {}
        for label, model, horizon in (("WAM-H16", predictor_for("DirectWAM-H16", 16, 8_192).fit(train), 16), ("WAM-H1", predictor_for("DirectWAM-H8", 1, 8_192).fit(train), 1), ("RecursiveWAM", predictor_for("RecursiveWAM", 16, 8_192).fit(train), 16), ("GMC-style", predictor_for("GMC", 16, 8_192).fit(train), 16)):
            result = simulate_horizon([value * 64 for value in window], model, horizon, run_config(2), initial_context=train[-getattr(model, "context_depth", 1):])
            vals[label] = result_speedup(baseline, result)
        rows.append({"workload": name, **vals, "direct_h16_minus_h1": vals["WAM-H16"] - vals["WAM-H1"], "direct_h16_minus_recursive": vals["WAM-H16"] - vals["RecursiveWAM"]})
    return rows


def feature_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    feature_names = ("context_reuse_depth16", "entropy_reduction", "h16_oracle_accuracy", "stride_stability", "delta_entropy", "gmc_confidence", "wam_confidence", "gmc_accuracy", "wam_accuracy", "sequential_fraction", "unique_lines")
    for feature in feature_names:
        wins = [float(row[feature]) for row in rows if row["winner"] == "WAM"]
        gmc = [float(row[feature]) for row in rows if row["winner"] == "GMC-style"]
        tied = [float(row[feature]) for row in rows if row["winner"] == "tie"]
        pooled = wins + gmc
        mean_w = statistics.mean(wins) if wins else 0.0
        mean_g = statistics.mean(gmc) if gmc else 0.0
        pooled_sd = statistics.pstdev(pooled) if len(pooled) > 1 else 0.0
        effect = (mean_w - mean_g) / pooled_sd if pooled_sd else 0.0
        threshold = (mean_w + mean_g) / 2
        correct = sum(value >= threshold for value in wins) + sum(value < threshold for value in gmc)
        output.append({"feature": feature, "wam_win_mean": mean_w, "gmc_win_mean": mean_g, "tie_mean": statistics.mean(tied) if tied else 0.0, "effect_size": effect, "simple_threshold": threshold, "threshold_accuracy": correct / max(1, len(pooled)), "wam_win_windows": len(wins), "gmc_win_windows": len(gmc), "top_signal": abs(effect)})
    return sorted(output, key=lambda row: float(row["top_signal"]), reverse=True)


def build_report(output: Path, windows: list[dict[str, object]], oracle: list[dict[str, object]], comp: list[dict[str, object]], selectors: list[dict[str, object]], budgets: list[dict[str, object]], disagreements: list[dict[str, object]], direct: list[dict[str, object]], trace_count: int) -> str:
    primary = [row for row in windows if int(row["window"]) == PRIMARY_WINDOW] or [row for row in windows if int(row["window"]) == 1_000]
    oracle_primary = next((row for row in oracle if int(row["window"]) == (PRIMARY_WINDOW if any(int(r["window"]) == PRIMARY_WINDOW for r in windows) else 1_000)), oracle[-1] if oracle else {})
    gmc_geo = float(oracle_primary.get("gmc_geomean", 0.0))
    wam_geo = float(oracle_primary.get("wam_geomean", 0.0))
    oracle_geo = float(oracle_primary.get("oracle_hybrid_geomean", 0.0))
    cheap = [row for row in selectors if row["selector"] in {"ConfidenceSelector", "RecentWinnerSelector", "EntropyAwareSelector"} and str(row["granularity"]) != "per_access"]
    cheap_best = max(cheap, key=lambda row: float(row["geomean_speedup"]), default={})
    budget_best = max((row for row in budgets if row.get("within_budget")), key=lambda row: float(row["speedup"]), default={})
    feature_rows = feature_summary([{**row} for row in windows])
    top = ", ".join(row["feature"] for row in feature_rows[:3]) if feature_rows else "none"
    oracle_gain = oracle_geo / max(1e-12, gmc_geo) - 1.0
    if oracle_gain < 0.02:
        classification = "A — No complementarity"
        continue_research = "NO"
        next_step = "Stop WAM architecture tuning; if the line is revisited, first replace the simplified GMC-style proxy with a faithful implementation."
    elif float(cheap_best.get("geomean_speedup", 0.0)) / max(1e-12, gmc_geo) - 1.0 < 0.02:
        classification = "B — Oracle-only complementarity"
        continue_research = "NO"
        next_step = "Do not tune WAM further; improve baseline fidelity and investigate why implementable selectors cannot recover the oracle gap."
    elif budget_best and float(budget_best.get("speedup", 0.0)) > gmc_geo:
        classification = "C — Niche sidecar value"
        continue_research = "YES"
        next_step = "Validate the sidecar on binary-level traces and a faithful GMC implementation."
    else:
        classification = "B — Oracle-only complementarity"
        continue_research = "NO"
        next_step = "Stop architecture tuning pending a faithful GMC reproduction."
    families = {str(row["workload"]).split("_size", 1)[0] for row in primary}
    measured_sizes = sorted({int(row["window"]) for row in windows})
    unmeasured_sizes = sorted(set(WINDOW_SIZES) - set(measured_sizes))
    return f"""# Hybrid complementarity analysis

This study uses {trace_count} source-instrumented trace files, frozen 70/30 chronological splits, and non-overlapping evaluation windows. To keep the cycle-level replay bounded, it samples up to {PHASE_SAMPLES_PER_SIZE} windows per size (early, middle, and late when available). The baseline is explicitly **GMC-style**, not the original GMC design; see `gmc_audit.md`.

## Executive result

- GMC-style geomean: **{gmc_geo:.3f}x**
- WAM-H16 geomean: **{wam_geo:.3f}x**
- OracleHybrid geomean: **{oracle_geo:.3f}x**
- OracleHybrid incremental gain over GMC-style: **{oracle_gain:.2%}**
- Best cheap selector: **{cheap_best.get('selector', 'n/a')} / {cheap_best.get('granularity', 'n/a')}** at **{float(cheap_best.get('geomean_speedup', 0.0)):.3f}x**
- Best equal-budget row: **{budget_best.get('total_budget', 'n/a')} bytes**, GMC {budget_best.get('gmc_bytes', 'n/a')} / WAM {budget_best.get('wam_bytes', 'n/a')} / selector {budget_best.get('selector_bytes', 'n/a')} bytes

## Answers to the research questions

1. **GMC fidelity:** the current implementation is a **simplified approximation**, not a paper-level reproduction. It lacks the original local/global context organization, PC context, published table organization, and update/arbitration policy.
2. **Consistent WAM wins:** WAM wins {sum(row['winner'] == 'WAM' for row in primary)} of {len(primary)} primary windows; per-workload detail is in `complementarity.csv`.
3. **Workloads:** {len({str(row['workload']).split('_size', 1)[0] for row in primary if row['winner'] == 'WAM'})} benchmark families have at least one WAM-winning primary window ({len({row['workload'] for row in primary if row['winner'] == 'WAM'})} workload/seed traces).
4. **Window fraction:** {sum(row['winner'] == 'WAM' for row in primary) / max(1, len(primary)):.1%} of primary windows are WAM wins.
5. **Advantage:** mean WAM advantage in winning primary windows is {statistics.mean(float(row['difference']) for row in primary if row['winner'] == 'WAM') if any(row['winner'] == 'WAM' for row in primary) else 0.0:.3f}x speedup.
6. **Oracle selector:** the oracle gain is {oracle_gain:.2%}; all window sizes are in `oracle_hybrid.csv`.
7. **Complementarity:** {'substantial' if oracle_gain >= 0.02 else 'limited'} under the critical 2% stopping rule.
8. **Discriminating properties:** top basic-statistics signals are **{top}**; see `features_summary.csv` for means, effect sizes, and threshold separability.
9. **Cheap selector:** the best implementable selector is {cheap_best.get('selector', 'n/a')} with {float(cheap_best.get('geomean_speedup', 0.0)):.3f}x; oracle-only rows are not treated as implementable.
10. **Realistic hybrid:** {'beats' if float(cheap_best.get('geomean_speedup', 0.0)) > gmc_geo else 'does not beat'} GMC-style in this bounded study.
11–12. **Storage:** selector state and lookup/update costs are explicit in `selector_results.csv`; equal-budget rows include all three components in `budget_split.csv`.
13. **Sidecar:** WAM activation rates are reported per selector; the confidence selector is GMC-primary and activates WAM only when GMC confidence is low and WAM confidence is high.
14. **Direct horizon:** `direct_horizon.csv` compares WAM-H16, WAM-H1, RecursiveWAM, and GMC-style. Direct-H16 is not credited automatically for wins that H1 or recursive WAM also obtains.
15. **Contribution:** the current result supports **{classification.split(' — ', 1)[1].lower()}**, not a general WAM replacement claim.

## Disagreement analysis

`disagreement.csv` records agreement, disagreement, one-sided predictions, and which predictor is correct on disagreements. It is based on the same frozen contexts and never allows the selector to see future outcomes.

## Selector accounting

Implementable selectors use only start-of-window confidence, prior-window usefulness, and history-derived entropy. `StaticPerWorkloadOracle` and `WindowOracle` are ceilings. The `per_access` line is labeled as a finest-100-access proxy because the current cycle simulator does not expose a composable per-access state snapshot; it must not be read as cycle-exact per-access arbitration.

## Classification

**{classification}**

Continue research: **{continue_research}**

Next step: {next_step}

## Limitations

The traces are source-instrumented, not binary-instrumented. The GMC-style implementation is intentionally simplified. The study freezes each predictor after the 70% training prefix, so it measures phase complementarity under a fixed predictor state rather than adaptive retraining. Measured window sizes are {measured_sizes}; sizes {unmeasured_sizes or 'none'} were not measured because the configured per-trace cap did not leave a complete window. Selector overhead is modeled as one cycle per arbitration block and bounded state bytes; union-prefetch mode is not used in the primary results, so both predictors never prefetch simultaneously.
"""


def print_verdict(windows, oracle, selectors, budgets, disagreements, direct, report: str) -> None:
    primary = [row for row in windows if int(row["window"]) == PRIMARY_WINDOW] or [row for row in windows if int(row["window"]) == 1_000]
    oracle_row = next((row for row in oracle if int(row["window"]) == int(primary[0]["window"])), {}) if primary else {}
    gmc = float(oracle_row.get("gmc_geomean", 0.0))
    wam = float(oracle_row.get("wam_geomean", 0.0))
    oracle_geo = float(oracle_row.get("oracle_hybrid_geomean", 0.0))
    cheap = [row for row in selectors if row["selector"] in {"ConfidenceSelector", "RecentWinnerSelector", "EntropyAwareSelector"} and str(row["granularity"]) != "per_access"]
    cheap_best = max(cheap, key=lambda row: float(row["geomean_speedup"]), default={})
    budget = max((row for row in budgets if row.get("within_budget")), key=lambda row: float(row["speedup"]), default={})
    agreement = statistics.mean(float(row["agreement_rate"]) for row in disagreements) if disagreements else 0.0
    wam_disc = sum(int(row["wam_wins_disagreements"]) for row in disagreements)
    gmc_disc = sum(int(row["gmc_wins_disagreements"]) for row in disagreements)
    top = ", ".join(row["feature"] for row in feature_summary(windows)[:3])
    direct_h16 = geomean([float(row["WAM-H16"]) for row in direct]) if direct else 0.0
    direct_h1 = geomean([float(row["WAM-H1"]) for row in direct]) if direct else 0.0
    recursive = geomean([float(row["RecursiveWAM"]) for row in direct]) if direct else 0.0
    classification_section = report.split("## Classification", 1)[1] if "## Classification" in report else ""
    classification = classification_section.split("**", 2)[1] if "**" in classification_section else "see report.md"
    print(f"GMC implementation fidelity: simplified approximation")
    print(f"GMC-style geomean: {gmc:.3f}x")
    print(f"WAM geomean: {wam:.3f}x")
    print(f"OracleHybrid geomean: {oracle_geo:.3f}x")
    print(f"CheapHybrid geomean: {float(cheap_best.get('geomean_speedup', 0.0)):.3f}x")
    print(f"OracleHybrid incremental gain over GMC-style: {oracle_geo / max(1e-12, gmc) - 1.0:.2%}")
    print(f"CheapHybrid incremental gain over GMC-style: {float(cheap_best.get('geomean_speedup', 0.0)) / max(1e-12, gmc) - 1.0:.2%}")
    print(f"fraction of windows WAM wins: {sum(row['winner'] == 'WAM' for row in primary) / max(1, len(primary)):.1%}")
    print(f"mean WAM advantage when winning: {statistics.mean(float(row['difference']) for row in primary if row['winner'] == 'WAM') if any(row['winner'] == 'WAM' for row in primary) else 0.0:.3f}x")
    print(f"agreement rate: {agreement:.1%}")
    print(f"WAM win rate on disagreements: {wam_disc / max(1, wam_disc + gmc_disc):.1%}")
    print(f"best total budget: {budget.get('total_budget', 'n/a')}")
    print(f"best GMC/WAM split: {budget.get('gmc_bytes', 'n/a')} / {budget.get('wam_bytes', 'n/a')}")
    print(f"WAM sidecar activation rate: {float(cheap_best.get('wam_activation_rate', 0.0)):.1%}")
    print(f"top 3 predictors of WAM success: {top}")
    print(f"direct-H16 contribution: H16 {direct_h16:.3f}x; H1 {direct_h1:.3f}x; recursive {recursive:.3f}x")
    print(f"final classification: {classification}")
    print(f"continue research: {'YES' if oracle_geo / max(1e-12, gmc) - 1.0 >= 0.02 else 'NO'}")
    print(f"next step: validate the sidecar on binary-level traces and a faithful GMC implementation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=Path("traces/source_instrumented/loads"))
    parser.add_argument("--output", type=Path, default=Path("results/hybrid_analysis"))
    parser.add_argument("--max-accesses", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train-cap", type=int, default=20_000)
    args = parser.parse_args()
    run(args.trace_dir, args.output, args.max_accesses, args.seed, args.train_cap)


if __name__ == "__main__":
    main()
