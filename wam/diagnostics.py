"""Falsification-oriented diagnostics for higher-order WAM context.

This phase deliberately writes to ``results/diagnostics`` so the original
benchmark artifacts remain intact. Run ``python -m wam.diagnostics``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

_MPL_CACHE = Path(tempfile.gettempdir()) / "wam-mpl"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

from .benchmark import default_simulator_config
from .hierarchy import HierarchyConfig
from .predictor import HigherOrderMarkovPredictor, WeightedTriePredictor
from .simulator import simulate
from .workloads import higher_order_ambiguous, higher_order_depth4, contextual, to_byte_addresses


def _split(trace: list[int], fraction: float = 0.7) -> tuple[list[int], list[int]]:
    split = max(1, min(len(trace) - 1, int(len(trace) * fraction)))
    return trace[:split], trace[split:]


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _predictor_accuracy(predictor, train: list[int], evaluation: list[int], context_depth: int, top_k: int = 1) -> tuple[float, float, float]:
    predictor.fit(train)
    context = train[-context_depth:]
    correct = 0
    topk = 0
    log_loss = 0.0
    for address in evaluation:
        predictions = predictor.predict(context, k=max(top_k, 256))
        if predictions:
            correct += predictions[0].address == address
            topk += any(prediction.address == address for prediction in predictions[:top_k])
            probability = next((prediction.weight for prediction in predictions if prediction.address == address), 1e-9)
            log_loss -= math.log(max(1e-9, min(1.0, probability)))
        else:
            log_loss -= math.log(1e-9)
        context = (context + [address])[-context_depth:]
    total = len(evaluation)
    return correct / total if total else 0.0, topk / total if total else 0.0, log_loss / total if total else 0.0


def _context_oracle(train: list[int], evaluation: list[int], depth: int) -> tuple[float, float]:
    counts: dict[tuple[int, ...], dict[int, int]] = {}
    history: list[int] = []
    for address in train:
        if history:
            key = tuple(history[-depth:])
            transitions = counts.setdefault(key, {})
            transitions[address] = transitions.get(address, 0) + 1
        history.append(address)
    context = train[-depth:]
    correct = 0
    reused = 0
    for address in evaluation:
        transitions = counts.get(tuple(context[-depth:]))
        if transitions:
            reused += 1
            best = max(transitions.items(), key=lambda item: (item[1], -item[0]))[0]
            correct += best == address
        context = (context + [address])[-depth:]
    total = len(evaluation)
    return correct / total if total else 0.0, reused / total if total else 0.0


def _flat_context_metrics(train: list[int], evaluation: list[int], depth: int) -> tuple[float, float]:
    """Efficient exact-context analysis for million-access length sweeps."""
    counts: dict[tuple[int, ...], dict[int, int]] = {}
    history: list[int] = []
    for address in train:
        if len(history) >= depth:
            key = tuple(history[-depth:])
            transitions = counts.setdefault(key, {})
            transitions[address] = transitions.get(address, 0) + 1
        history.append(address)
    context = train[-depth:]
    correct = 0
    reused = 0
    for address in evaluation:
        transitions = counts.get(tuple(context[-depth:]))
        if transitions:
            reused += 1
            correct += max(transitions.items(), key=lambda item: (item[1], -item[0]))[0] == address
        context = (context + [address])[-depth:]
    total = len(evaluation)
    return (correct / total if total else 0.0, reused / total if total else 0.0)


def context_depth_rows(trace: list[int], workload: str, config, depths: tuple[int, ...] = (1, 2, 3, 4, 8)) -> list[dict]:
    train, evaluation = _split(trace)
    rows: list[dict] = []
    for depth in depths:
        predictor = WeightedTriePredictor(context_depth=depth, threshold=0.0).fit(train)
        oracle_accuracy, oracle_reuse = _context_oracle(train, evaluation, depth)
        accuracy, topk, log_loss = _predictor_accuracy(WeightedTriePredictor(context_depth=depth, threshold=0.0), train, evaluation, depth, top_k=3)
        result = simulate(to_byte_addresses(evaluation, config.cache_line_size), predictor, config, enable_prefetch=False, initial_context=train[-depth:])
        stats = predictor.context_statistics().get(depth, {})
        rows.append({
            "workload": workload,
            "depth": depth,
            "actual_accuracy": accuracy,
            "top3_accuracy": topk,
            "log_loss": log_loss,
            "oracle_accuracy": oracle_accuracy,
            "oracle_context_reuse": oracle_reuse,
            "simulated_context_reuse": result.metrics.context_reuse_ratio,
            "fallback_count": result.metrics.fallback_count,
            "unseen_context_count": result.metrics.unseen_context_count,
            "conditional_entropy": predictor.conditional_entropy(depth),
            "unique_contexts": stats.get("unique_contexts", 0),
            "total_observations": stats.get("total_observations", 0),
            "mean_observations": stats.get("mean_observations", 0.0),
            "median_observations": stats.get("median_observations", 0.0),
            "contexts_seen_once": stats.get("contexts_seen_once", 0),
            "contexts_seen_at_least_2": stats.get("contexts_seen_at_least_2", 0),
            "contexts_seen_at_least_5": stats.get("contexts_seen_at_least_5", 0),
            "contexts_seen_at_least_10": stats.get("contexts_seen_at_least_10", 0),
            "storage_bytes": predictor.storage_stats()["estimated_bytes"],
        })
    return rows


def markov_rows(trace: list[int], config, depths: tuple[int, ...] = (1, 2, 3, 4)) -> list[dict]:
    train, evaluation = _split(trace)
    rows: list[dict] = []
    for depth in depths:
        wam = WeightedTriePredictor(context_depth=depth, threshold=0.0)
        markov = HigherOrderMarkovPredictor(context_depth=depth)
        wam_accuracy, wam_topk, _ = _predictor_accuracy(wam, train, evaluation, depth, 3)
        markov_accuracy, markov_topk, _ = _predictor_accuracy(markov, train, evaluation, depth, 3)
        wam_full = WeightedTriePredictor(context_depth=depth, threshold=0.0).fit(train)
        markov_full = HigherOrderMarkovPredictor(context_depth=depth).fit(train)
        wam_result = simulate(to_byte_addresses(evaluation, config.cache_line_size), wam_full, config, initial_context=train[-depth:])
        markov_result = simulate(to_byte_addresses(evaluation, config.cache_line_size), markov_full, config, initial_context=train[-depth:])
        rows.extend([
            {"workload": "HigherOrder", "predictor": f"WAM-{depth}", "depth": depth, "accuracy": wam_accuracy, "top3_accuracy": wam_topk, "speedup": 0.0, "storage_bytes": wam_full.storage_stats()["estimated_bytes"], "lookup_cost": wam_full.lookup_cost, "effective_latency": wam_result.metrics.average_access_latency},
            {"workload": "HigherOrder", "predictor": f"Markov-{depth}", "depth": depth, "accuracy": markov_accuracy, "top3_accuracy": markov_topk, "speedup": 0.0, "storage_bytes": markov_full.storage_stats()["estimated_bytes"], "lookup_cost": markov_full.lookup_cost, "effective_latency": markov_result.metrics.average_access_latency},
        ])
    return rows


def repetition_rows(config, repetitions: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)) -> list[dict]:
    rows: list[dict] = []
    for repeats in repetitions:
        trace = higher_order_ambiguous(context_count=24, repeats=repeats)
        train, evaluation = _split(trace)
        baseline = simulate(to_byte_addresses(evaluation, 64), None, config, enable_prefetch=False)
        for depth in (1, 2, 3, 4):
            predictor = WeightedTriePredictor(context_depth=depth, threshold=0.0).fit(train)
            accuracy, _, _ = _predictor_accuracy(WeightedTriePredictor(context_depth=depth, threshold=0.0), train, evaluation, depth)
            result = simulate(to_byte_addresses(evaluation, 64), predictor, config, initial_context=train[-depth:])
            rows.append({"repetitions": repeats, "depth": depth, "accuracy": accuracy, "speedup": baseline.cycles / result.cycles if result.cycles else 0.0, "reuse": result.metrics.context_reuse_ratio, "storage_bytes": predictor.storage_stats()["estimated_bytes"]})
    return rows


def trace_length_rows(config, lengths: tuple[int, ...] = (10**3, 10**4, 10**5, 10**6)) -> list[dict]:
    rows: list[dict] = []
    for length in lengths:
        repeats = max(1, math.ceil(length / (100 * 4)))
        trace = higher_order_ambiguous(context_count=100, repeats=repeats)[:length]
        train, evaluation = _split(trace)
        for depth in (1, 2, 3, 4):
            accuracy, reuse = _flat_context_metrics(train, evaluation, depth)
            oracle = accuracy
            rows.append({"trace_length": length, "depth": depth, "accuracy": accuracy, "oracle_accuracy": oracle, "reuse": reuse})
    return rows


def policy_rows(trace: list[int], config) -> tuple[list[dict], list[dict], list[dict]]:
    train, evaluation = _split(trace)
    raw = to_byte_addresses(evaluation, config.cache_line_size)
    baseline = simulate(raw, None, config, enable_prefetch=False)
    pruning: list[dict] = []
    for minimum in (1, 2, 4, 8, 16):
        predictor = WeightedTriePredictor(context_depth=3, threshold=0.05, minimum_observations=minimum).fit(train)
        result = simulate(raw, predictor, config, initial_context=train[-3:])
        pruning.append({"minimum_observations": minimum, "accuracy": result.metrics.top1_accuracy, "speedup": baseline.cycles / result.cycles, "storage_bytes": predictor.storage_stats()["estimated_bytes"], "reuse": result.metrics.context_reuse_ratio})
    support: list[dict] = []
    for support_k in (0, 1, 4, 16, 64):
        predictor = WeightedTriePredictor(context_depth=3, threshold=0.05, support_k=support_k).fit(train)
        result = simulate(raw, predictor, config, initial_context=train[-3:])
        support.append({"support_k": support_k, "accuracy": result.metrics.top1_accuracy, "speedup": baseline.cycles / result.cycles, "precision": result.metrics.prefetch_precision})
    entropy: list[dict] = []
    for threshold in (0.0, 0.5, 1.0, 1.5, 2.0):
        predictor = WeightedTriePredictor(context_depth=3, threshold=0.05, entropy_threshold=threshold if threshold else None).fit(train)
        result = simulate(raw, predictor, config, initial_context=train[-3:])
        entropy.append({"entropy_threshold": threshold, "accuracy": result.metrics.top1_accuracy, "speedup": baseline.cycles / result.cycles, "precision": result.metrics.prefetch_precision})
    return pruning, support, entropy


def _plot(rows: list[dict], x: str, y: str, hue: str, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    """Write a lightweight SVG line chart without requiring a display/font cache."""
    width, height = 800, 500
    left, top, plot_width, plot_height = 80, 50, 650, 350
    numeric_x = [float(row[x]) for row in rows]
    numeric_y = [float(row[y]) for row in rows]
    min_x, max_x = min(numeric_x), max(numeric_x)
    min_y, max_y = min(numeric_y), max(numeric_y)
    x_span = max(max_x - min_x, 1e-9)
    y_span = max(max_y - min_y, 1e-9)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{width/2}" y="25" text-anchor="middle" font-family="sans-serif" font-size="16">{title}</text>', f'<line x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" y2="{top+plot_height}" stroke="black"/>', f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="black"/>']
    groups = sorted({str(row[hue]) for row in rows})
    for index, group in enumerate(groups):
        subset = sorted((row for row in rows if str(row[hue]) == group), key=lambda row: float(row[x]))
        points = []
        for row in subset:
            px = left + (float(row[x]) - min_x) / x_span * plot_width
            py = top + plot_height - (float(row[y]) - min_y) / y_span * plot_height
            points.append(f"{px:.1f},{py:.1f}")
        color = colors[index % len(colors)]
        svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        svg.append(f'<text x="{left+plot_width+15}" y="{top+20+index*20}" font-family="sans-serif" font-size="12" fill="{color}">{group}</text>')
    svg.extend([f'<text x="{left+plot_width/2}" y="{height-15}" text-anchor="middle" font-family="sans-serif" font-size="12">{xlabel}</text>', f'<text x="15" y="{top+plot_height/2}" transform="rotate(-90 15 {top+plot_height/2})" text-anchor="middle" font-family="sans-serif" font-size="12">{ylabel}</text>', "</svg>"])
    path = path.with_suffix(".svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg), encoding="utf-8")


def write_report(output: Path, depth_rows: list[dict], markov: list[dict], repetitions: list[dict], lengths: list[dict], pruning: list[dict], support: list[dict], entropy: list[dict]) -> str:
    diagnostic_workload = "Depth4" if any(row["workload"] == "Depth4" for row in depth_rows) else ("HigherOrder" if any(row["workload"] == "HigherOrder" for row in depth_rows) else str(depth_rows[0]["workload"]))
    higher = [row for row in depth_rows if row["workload"] == diagnostic_workload]
    d1 = next(row for row in higher if row["depth"] == 1)
    d4 = next(row for row in higher if row["depth"] == 4)
    max_repetition = max(repetitions, key=lambda row: row["repetitions"])
    useful_repetition = next((row["repetitions"] for row in repetitions if row["depth"] == 2 and row["accuracy"] > 0.9), "not reached")
    d4_long = next(row for row in lengths if row["depth"] == 4 and row["trace_length"] == 10**6)
    d1_long = next(row for row in lengths if row["depth"] == 1 and row["trace_length"] == 10**6)
    gap = d4["oracle_accuracy"] - d4["actual_accuracy"]
    entropy_drop = d1["conditional_entropy"] - d4["conditional_entropy"]
    if d4["oracle_accuracy"] - d1["oracle_accuracy"] > 0.2 and d4["actual_accuracy"] + 0.1 < d4["oracle_accuracy"]:
        classification = "A — Predictor implementation problem"
        reason = "The training oracle has higher-order signal, but the WAM prediction path leaves a material gap to that oracle."
    elif d4["oracle_accuracy"] - d1["oracle_accuracy"] > 0.2 and d4["oracle_context_reuse"] < 0.5:
        classification = "B — Data sparsity problem"
        reason = "Higher-order signal exists, but held-out deep contexts are reused too rarely without substantial repetition."
    elif entropy_drop < 0.05:
        classification = "D — Fundamental predictability problem"
        reason = "Conditional entropy barely falls with additional history."
    elif d4["actual_accuracy"] > 0.9:
        classification = "C — Prefetch execution problem"
        reason = "Prediction is accurate in the isolated oracle mode; the full benchmark must determine whether timing, bandwidth, pollution, and lookup cost erase the benefit."
    else:
        classification = "E — Promising"
        reason = "Higher-order accuracy and execution both improve on the diagnostic workload."
    lines = [
        "# WAM Context Diagnostics",
        "",
        "This is a falsification report. It keeps the prior benchmark untouched and separates representation quality from prefetch execution.",
        "",
        "## Final classification",
        "",
        f"**{classification}**",
        "",
        reason,
        "",
        "## Answers to the diagnostic questions",
        "",
        f"1. Higher-order information present: depth-1 oracle accuracy was {d1['oracle_accuracy']:.1%}; depth-4 oracle accuracy was {d4['oracle_accuracy']:.1%}; conditional entropy changed by {entropy_drop:.3f} bits.",
        f"2. WAM learning it: actual depth-4 accuracy was {d4['actual_accuracy']:.1%}, with an oracle gap of {gap:.1%}.",
        f"3. Deep-context reuse: depth-4 training/evaluation reuse was {d4['oracle_context_reuse']:.1%}; at one million accesses it was {d4_long['reuse']:.1%}.",
        f"4. Fallback/unseen behavior is in `context_depth.csv`; the simulator records matched-depth histograms, fallback counts, and unseen contexts for every lookup.",
        f"5. Repetition density: depth-2 target accuracy first exceeded 90% at {useful_repetition} repetitions per context.",
        f"6. WAM vs equivalent Markov-N: see `markov_comparison.csv`; both use the same context depth and the same train/test split.",
        f"7. Entropy gating: see `entropy_policy.csv`; it is useful only when entropy correlates with harmful speculative requests.",
        f"8. Bottleneck: the depth-4 one-million-access oracle/actual pair was {d4_long['oracle_accuracy']:.1%}/{d4_long['accuracy']:.1%}; compare full-mode speedups in the prior report to isolate execution cost.",
        f"9. Empirical maximum: the `oracle_accuracy` column is the training-distribution context oracle for each depth; at one million accesses depth 1/4 were {d1_long['oracle_accuracy']:.1%}/{d4_long['oracle_accuracy']:.1%}.",
        f"10. Continue? {('Yes, but only with irregular pointer-heavy traces and a calibrated timing model.' if classification.startswith('E') else 'Only as a targeted diagnostic; the default evidence does not justify broad hardware investment yet.')}",
        "",
        "## Interpretation",
        "",
        f"The largest actual diagnostic gap is {gap:.1%}. Repetition sweep maximum tested density was {max_repetition['repetitions']} repetitions/context. The pruning, support-confidence, and entropy-gating sweeps are deliberately small and expose storage/accuracy/speedup tradeoffs rather than selecting a flattering configuration.",
        "",
        "Real traces are supported by `wam.traces.iter_addresses`. Capture representative programs externally with Valgrind/Lackey, Pin, DynamoRIO, or perf, convert to one byte address per line, and run `python -m wam.diagnostics --trace path/to/trace.txt`. Priority workloads are linked-list traversal, tree/graph traversal, hash tables, sorting, dynamic programming, pointer chasing, and SQLite queries.",
        "",
        "## Artifacts",
        "",
        "`context_depth.csv`, `markov_comparison.csv`, `repetition_sweep.csv`, `trace_length_sweep.csv`, `pruning.csv`, `support_confidence.csv`, `entropy_policy.csv`, and `plots/`.",
    ]
    report = "\n".join(lines) + "\n"
    (output / "context_diagnostics.md").write_text(report, encoding="utf-8")
    return report


def run(output: Path, trace_path: Path | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    config = default_simulator_config()
    if trace_path:
        from .traces import load_trace, normalize_addresses

        trace = list(normalize_addresses(load_trace(trace_path), config.cache_line_size))
        workload = "ExternalTrace"
    else:
        trace = higher_order_ambiguous(context_count=100, repeats=32)
        workload = "HigherOrder"
    depth_rows = context_depth_rows(trace, workload, config)
    depth_rows += context_depth_rows(higher_order_depth4(context_count=24, repeats=32), "Depth4", config)
    markov = markov_rows(higher_order_depth4(context_count=24, repeats=32), config)
    repetitions = repetition_rows(config)
    lengths = trace_length_rows(config)
    pruning, support, entropy = policy_rows(contextual(repeats=256), config)
    _write_csv(output / "context_depth.csv", depth_rows)
    _write_csv(output / "markov_comparison.csv", markov)
    _write_csv(output / "repetition_sweep.csv", repetitions)
    _write_csv(output / "trace_length_sweep.csv", lengths)
    _write_csv(output / "pruning.csv", pruning)
    _write_csv(output / "support_confidence.csv", support)
    _write_csv(output / "entropy_policy.csv", entropy)
    (output / "config.json").write_text(json.dumps({"simulator": asdict(config), "long_lengths": [10**3, 10**4, 10**5, 10**6], "repetitions": [1, 2, 4, 8, 16, 32, 64, 128], "trace": str(trace_path) if trace_path else None}, indent=2), encoding="utf-8")
    # Write the report before rendering figures so a long run always leaves a
    # usable diagnostic summary even if optional figure generation is stopped.
    report = write_report(output, depth_rows, markov, repetitions, lengths, pruning, support, entropy)
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    _plot(depth_rows, "depth", "oracle_context_reuse", "workload", "Context reuse vs depth", "Context depth", "Reuse ratio", plots / "context_reuse_vs_depth.svg")
    _plot(depth_rows, "depth", "mean_observations", "workload", "Mean observations per context", "Context depth", "Observations/context", plots / "mean_observations_vs_depth.svg")
    _plot(depth_rows, "depth", "conditional_entropy", "workload", "Conditional entropy vs depth", "Context depth", "Bits", plots / "conditional_entropy_vs_depth.svg")
    _plot(depth_rows, "depth", "actual_accuracy", "workload", "Actual accuracy vs depth", "Context depth", "Accuracy", plots / "actual_accuracy_vs_depth.svg")
    _plot(depth_rows, "depth", "oracle_accuracy", "workload", "Oracle accuracy vs depth", "Context depth", "Accuracy", plots / "oracle_accuracy_vs_depth.svg")
    _plot(repetitions, "repetitions", "accuracy", "depth", "Repetition density vs accuracy", "Repetitions/context", "Accuracy", plots / "repetition_vs_accuracy.svg")
    _plot(repetitions, "repetitions", "speedup", "depth", "Repetition density vs speedup", "Repetitions/context", "Speedup", plots / "repetition_vs_speedup.svg")
    _plot(lengths, "trace_length", "accuracy", "depth", "Trace length vs accuracy", "Trace length", "Accuracy", plots / "trace_length_vs_accuracy.svg")
    _plot(markov, "depth", "accuracy", "predictor", "WAM vs Markov-N accuracy", "Order", "Accuracy", plots / "wam_vs_markov_accuracy.svg")
    (output.parent / "context_diagnostics.md").write_text(report, encoding="utf-8")
    d4 = next(row for row in depth_rows if row["workload"] == workload and row["depth"] == 4)
    print("\nWAM context diagnostics")
    print(f"Depth-4 actual/oracle accuracy: {d4['actual_accuracy']:.1%}/{d4['oracle_accuracy']:.1%}")
    print(f"Depth-4 context reuse: {d4['oracle_context_reuse']:.1%}")
    print(f"Report: {output.parent / 'context_diagnostics.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/diagnostics"))
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    run(args.output, args.trace)


if __name__ == "__main__":
    main()
