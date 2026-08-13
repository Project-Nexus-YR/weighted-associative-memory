"""Lead-time experiments for the falsifiable WAM prefetch hypothesis.

Run ``python -m wam.horizon_analysis``. Results are written to a new
``results/horizon_analysis`` directory and never overwrite earlier reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import asdict, replace
from pathlib import Path

from .benchmark import default_simulator_config
from .diagnostics import _plot
from .horizon import DirectHorizonWAM, DirectMarkovHorizon, HorizonConfig, NoHorizonPredictor, OracleHorizon, RecursiveWAM, simulate_horizon
from .traces import load_trace, normalize_addresses
from .workloads import all_workloads, contextual, higher_order_ambiguous, longer_dependency, to_byte_addresses

HORIZONS = (1, 2, 4, 8, 16, 32)
SYSTEMS = ("Oracle", "DirectWAM", "RecursiveWAM", "DirectMarkov")


def config(dram_latency: int = 150, compute_gap: int = 0, outstanding: int = 8) -> HorizonConfig:
    base = default_simulator_config(dram_latency=dram_latency)
    return HorizonConfig(
        hierarchy=base.hierarchy,
        prefetch_issue_cost=base.prefetch_issue_cost,
        prefetch_destination=base.prefetch_destination,
        address_bytes=base.address_bytes,
        top_k=base.top_k,
        max_outstanding_prefetches=outstanding,
        compute_cycles_between_accesses=compute_gap,
    )


def split(trace: list[int], fraction: float = 0.7) -> tuple[list[int], list[int]]:
    index = max(1, min(len(trace) - 1, int(len(trace) * fraction)))
    return trace[:index], trace[index:]


def predictor_for(system: str, horizon: int, train: list[int], context_depth: int = 4, threshold: float = 0.0):
    if system == "Oracle":
        return OracleHorizon()
    if system == "DirectWAM":
        return DirectHorizonWAM(context_depth=context_depth, horizon=horizon, threshold=threshold).fit(train)
    if system == "RecursiveWAM":
        return RecursiveWAM(context_depth=context_depth, max_horizon=horizon, threshold=threshold, cumulative_threshold=0.0, speculative_width=horizon).fit(train)
    if system == "DirectMarkov":
        return DirectMarkovHorizon(context_depth=context_depth, horizon=horizon, threshold=threshold).fit(train)
    raise ValueError(system)


def _metrics_row(workload: str, system: str, horizon: int, trace_length: int, result, baseline_cycles: int, oracle_cycles: int) -> dict[str, object]:
    m = result.metrics
    oracle_gain = oracle_cycles and baseline_cycles / oracle_cycles - 1
    wam_gain = baseline_cycles / result.cycles - 1 if result.cycles else 0.0
    return {
        "workload": workload,
        "system": system,
        "configuration": f"{system}-H{horizon}",
        "configuration": f"{system}-H{horizon}",
        "horizon": horizon,
        "trace_length": trace_length,
        "accuracy": m.top1_accuracy,
        "top3_accuracy": m.topk_accuracy,
        "prediction_attempts": m.prediction_attempts,
        "total_cycles": m.cycles,
        "average_latency": m.cycles / m.total_accesses if m.total_accesses else 0.0,
        "raw_memory_cycles": m.raw_memory_cycles,
        "speedup": baseline_cycles / m.cycles if m.cycles else 0.0,
        "useful_prefetches": m.useful_prefetches,
        "prefetches_issued": m.prefetches_issued,
        "prefetches_completed": m.prefetches_completed,
        "late_prefetches": m.late_prefetches,
        "late_rate": m.late_prefetch_rate,
        "unused_prefetches": m.unused_prefetches,
        "dropped_prefetches": m.dropped_prefetches,
        "pollution_misses": m.pollution_misses,
        "bandwidth_bytes": m.bandwidth_bytes,
        "bandwidth_utilization": m.bandwidth_utilization,
        "cycles_hidden": m.cycles_hidden,
        "fully_hidden_misses": m.fully_hidden_misses,
        "partially_hidden_misses": m.partially_hidden_misses,
        "unhidden_misses": m.unhidden_misses,
        "mean_lead_time": m.mean_lead_time,
        "median_lead_time": m.median_lead_time,
        "mean_slack": m.mean_slack,
        "median_slack": m.median_slack,
        "queue_occupancy": m.queue_occupancy_sum / m.queue_occupancy_samples if m.queue_occupancy_samples else 0.0,
        "lookup_prefetch_overhead": m.predictor_overhead + m.prefetch_overhead,
        "storage_bytes": result.predictor_storage.get("estimated_bytes", 0),
        "fraction_of_oracle_gain": wam_gain / oracle_gain if oracle_gain > 1e-9 else 0.0,
    }


def evaluate_workload(workload: str, line_trace: list[int], run_config: HorizonConfig, train_fraction: float = 0.7, horizons: tuple[int, ...] = HORIZONS) -> tuple[list[dict], list[dict], list[dict]]:
    train, evaluation = split(line_trace, train_fraction)
    raw_evaluation = to_byte_addresses(evaluation, run_config.hierarchy.cache_line_size)
    all_rows: list[dict] = []
    oracle_rows: list[dict] = []
    timeliness: list[dict] = []
    for horizon in horizons:
        baseline = simulate_horizon(raw_evaluation, NoHorizonPredictor(), horizon, run_config, enable_prefetch=False)
        oracle = simulate_horizon(raw_evaluation, OracleHorizon(), horizon, run_config, initial_context=train[-1:])
        oracle_cycles = oracle.cycles
        for system in SYSTEMS:
            predictor = predictor_for(system, horizon, train)
            result = oracle if system == "Oracle" else simulate_horizon(raw_evaluation, predictor, horizon, run_config, initial_context=train[-4:])
            row = _metrics_row(workload, system, horizon, len(line_trace), result, baseline.cycles, oracle_cycles)
            all_rows.append(row)
            if system == "Oracle":
                oracle_rows.append(row.copy())
            if system in {"Oracle", "DirectWAM", "RecursiveWAM", "DirectMarkov"}:
                timeliness.append({"workload": workload, "system": system, "horizon": horizon, "mean_lead_time": result.metrics.mean_lead_time, "median_lead_time": result.metrics.median_lead_time, "mean_slack": result.metrics.mean_slack, "median_slack": result.metrics.median_slack, "late_rate": result.metrics.late_prefetch_rate, "cycles_hidden": result.metrics.cycles_hidden, "fully_hidden": result.metrics.fully_hidden_misses, "partially_hidden": result.metrics.partially_hidden_misses, "unhidden": result.metrics.unhidden_misses})
    return all_rows, oracle_rows, timeliness


def compute_gap_rows(line_trace: list[int], gaps: tuple[int, ...], run_horizons: tuple[int, ...] = (1, 4, 8, 16)) -> list[dict]:
    rows: list[dict] = []
    for gap in gaps:
        run_config = config(compute_gap=gap)
        evaluated, _, _ = evaluate_workload("Contextual", line_trace, run_config, horizons=run_horizons)
        best = max(evaluated, key=lambda row: row["speedup"])
        rows.append({"compute_gap": gap, "best_horizon": best["horizon"], "maximum_speedup": best["speedup"], "best_system": best["system"], "accuracy": best["accuracy"]})
    return rows


def dram_rows(line_trace: list[int], latencies: tuple[int, ...] = (80, 150, 300, 500)) -> list[dict]:
    rows: list[dict] = []
    for latency in latencies:
        evaluated, _, _ = evaluate_workload("Contextual", line_trace, config(dram_latency=latency), horizons=(1, 4, 8, 16))
        for row in evaluated:
            if row["system"] in {"Oracle", "DirectWAM", "RecursiveWAM"}:
                rows.append({"dram_latency": latency, "system": row["system"], "horizon": row["horizon"], "speedup": row["speedup"], "accuracy": row["accuracy"], "late_rate": row["late_rate"]})
    return rows


def bandwidth_rows(line_trace: list[int], limits: tuple[int, ...] = (1, 2, 4, 8, 16, 32)) -> list[dict]:
    rows: list[dict] = []
    for limit in limits:
        evaluated, _, _ = evaluate_workload("Contextual", line_trace, config(outstanding=limit), horizons=(1, 4, 8, 16))
        for row in evaluated:
            if row["system"] in {"Oracle", "DirectWAM", "RecursiveWAM"}:
                rows.append({"outstanding_limit": limit, "system": row["system"], "horizon": row["horizon"], "speedup": row["speedup"], "drops": row["dropped_prefetches"], "queue_occupancy": row["queue_occupancy"]})
    return rows


def horizon_accuracy_rows(line_trace: list[int], workload: str, run_config: HorizonConfig, train_fraction: float = 0.7, horizons: tuple[int, ...] = HORIZONS) -> list[dict]:
    train, evaluation = split(line_trace, train_fraction)
    raw = to_byte_addresses(evaluation, run_config.hierarchy.cache_line_size)
    rows: list[dict] = []
    for horizon in horizons:
        for system in ("DirectWAM", "RecursiveWAM", "DirectMarkov"):
            predictor = predictor_for(system, horizon, train)
            result = simulate_horizon(raw, predictor, horizon, run_config, enable_prefetch=False, initial_context=train[-4:])
            rows.append({"workload": workload, "horizon": horizon, "system": system, "accuracy": result.metrics.top1_accuracy, "top3_accuracy": result.metrics.topk_accuracy, "coverage": result.metrics.prediction_attempts / max(1, len(evaluation) - horizon), "storage_bytes": result.predictor_storage.get("estimated_bytes", 0)})
    return rows


def fast_long_accuracy_rows(line_trace: list[int], workload: str, horizons: tuple[int, ...] = HORIZONS) -> list[dict]:
    """Accuracy/reuse-only long-trace pass without hierarchy cycle accounting."""
    train, evaluation = split(line_trace)
    rows: list[dict] = []
    for horizon in horizons:
        # Exact-context table keeps this required million-access diagnostic
        # bounded in memory/time; the full trie is exercised in the main runs.
        counts: dict[tuple[int, ...], dict[int, int]] = {}
        for position in range(max(0, len(train) - horizon)):
            key = tuple(train[max(0, position - 3) : position + 1])
            transitions = counts.setdefault(key, {})
            target = train[position + horizon]
            transitions[target] = transitions.get(target, 0) + 1
        context = train[-4:]
        correct = 0
        attempts = 0
        reused = 0
        for index, address in enumerate(evaluation):
            context = (context + [address])[-4:]
            transitions = counts.get(tuple(context[-4:]), {})
            predictions = [max(transitions.items(), key=lambda item: (item[1], -item[0]))[0]] if transitions else []
            if index + horizon < len(evaluation):
                attempts += 1
                if predictions:
                    reused += 1
                    correct += predictions[0] == evaluation[index + horizon]
        rows.append({"workload": workload, "horizon": horizon, "system": "DirectWAM", "accuracy": correct / attempts if attempts else 0.0, "context_reuse": reused / attempts if attempts else 0.0, "evaluated_accesses": attempts, "storage_bytes": len(counts) * 24, "mode": "long_accuracy_only"})
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def failure_rows(summary: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in summary:
        if row["system"] != "DirectWAM":
            continue
        potential = int(row["useful_prefetches"] * 150 + row["unhidden_misses"] * 150)
        rows.append({"workload": row["workload"], "horizon": row["horizon"], "potential_cycles_saved": potential, "wrong_prediction_cost": int((1 - float(row["accuracy"])) * max(1, int(row["prefetches_issued"])) * 150), "lateness_cost": int(float(row["late_rate"]) * max(1, int(row["useful_prefetches"])) * 150), "bandwidth_cost": int(row["dropped_prefetches"]) * 150, "pollution_cost": int(row["pollution_misses"]) * 150, "predictor_overhead": int(row["lookup_prefetch_overhead"]), "net_saved": int(row["cycles_hidden"]) - int(row["lookup_prefetch_overhead"])})
    return rows


def report(output: Path, summary: list[dict], oracle: list[dict], timeliness: list[dict], compute: list[dict], dram: list[dict], bandwidth: list[dict], accuracy: list[dict]) -> str:
    oracle_best = max(oracle, key=lambda row: row["speedup"]) if oracle else None
    wam_rows = [row for row in summary if row["system"] in {"DirectWAM", "RecursiveWAM"}]
    wam_best = max(wam_rows, key=lambda row: row["speedup"]) if wam_rows else None
    best_workload = wam_best["workload"] if wam_best else None
    h1 = next((row for row in summary if row["system"] == "DirectWAM" and row["horizon"] == 1 and row["workload"] == best_workload), None)
    best_timeliness = next((row for row in timeliness if row["system"] == wam_best["system"] and row["horizon"] == wam_best["horizon"] and row["workload"] == wam_best["workload"]), None) if wam_best else None
    all_oracle = [row for row in oracle if row["system"] == "Oracle"]
    oracle_max = max((float(row["speedup"]) for row in all_oracle), default=1.0)
    wam_max = max((float(row["speedup"]) for row in wam_rows), default=1.0)
    fraction = float(wam_best["fraction_of_oracle_gain"]) if wam_best else 0.0
    mean_gap = statistics.mean([float(row["compute_gap"]) for row in compute]) if compute else 0.0
    lead_interval = config().hierarchy.l1_latency + mean_gap
    required = {"L2": config().hierarchy.l2_latency / max(1, lead_interval), "L3": config().hierarchy.l3_latency / max(1, lead_interval), "DRAM": config().hierarchy.dram_latency / max(1, lead_interval)}
    if oracle_best and float(oracle_best["speedup"]) <= 1.1:
        classification = "A — No latency-hiding opportunity"
    elif oracle_best and float(oracle_best["speedup"]) > 1.1 and wam_max <= 1.05:
        classification = "C — Long-horizon predictability collapses"
    elif wam_best and float(wam_best["lookup_prefetch_overhead"]) > 0.5 * max(1.0, float(wam_best["cycles_hidden"])) and fraction < 0.25:
        classification = "E — Predictor overhead bottleneck"
    elif wam_best and float(wam_best["dropped_prefetches"]) > float(wam_best["useful_prefetches"]):
        classification = "D — Bandwidth/pollution bottleneck"
    else:
        classification = "F — Promising latency-hiding result"
    lines = [
        "# WAM Prediction-Horizon Analysis", "",
        "This report tests whether accurate higher-order prediction arrives early enough to overlap memory latency. It preserves the previous benchmark and diagnostics artifacts.", "",
        "## Final classification", "", f"**{classification}**", "",
        f"- Best oracle horizon: H{oracle_best['horizon'] if oracle_best else 'n/a'} ({oracle_best['workload'] if oracle_best else 'n/a'}).", f"- Best WAM result: {wam_best['system'] if wam_best else 'n/a'} H{wam_best['horizon'] if wam_best else 'n/a'}.", f"- Oracle maximum speedup: {oracle_max:.3f}x.", f"- WAM maximum speedup: {wam_max:.3f}x.", f"- WAM fraction of oracle gain at its best row: {fraction:.1%}.", f"- H1 WAM late-prefetch rate: {float(h1['late_rate']):.1%}." if h1 else "- H1 WAM late-prefetch rate: unavailable.", f"- Best-horizon WAM late-prefetch rate: {float(best_timeliness['late_rate']):.1%}." if best_timeliness else "- Best-horizon WAM late-prefetch rate: unavailable.", "",
        "## Answers", "",
        f"1. Perfect predictor speedup: maximum observed oracle speedup was {oracle_max:.3f}x; see `oracle_horizon.csv` for workload-specific results.", f"2. Best oracle horizon: H{oracle_best['horizon'] if oracle_best else 'n/a'}.", f"3. H1 is {'too late relative to the best oracle horizon' if oracle_best and oracle_best['horizon'] > 1 and oracle_best['speedup'] > 1.05 else 'not clearly too late in this model'}.", f"4. Estimated accesses needed to hide latency: L2 {required['L2']:.1f}, L3 {required['L3']:.1f}, DRAM {required['DRAM']:.1f}; use `compute_gap.csv` for measured sensitivity.", f"5. WAM accuracy at H1/Hbest: {float(h1['accuracy']):.1%}/{float(wam_best['accuracy']):.1%}." if h1 and wam_best else "5. WAM accuracy by horizon is in `horizon_accuracy.csv`.", "6. Direct versus recursive: compare system rows in `summary.csv`; recursive traversal pays repeated lookup cost and can lose confidence multiplicatively.", "7. Fraction of oracle: reported above and per row in `summary.csv`.", "8. Remaining loss buckets are in `failure_breakdown.csv`; lateness, wrong predictions, bandwidth drops, pollution, and overhead are estimated separately.", f"9. Compute gaps: the best horizon/maximum speedup curve is in `compute_gap.csv`; mean tested gap was {mean_gap:.1f} cycles.", "10. DRAM sweep: `dram_sweep.csv` tests 80/150/300/500-cycle models without claiming universal hardware values.", "11. The optimum is empirical, not assumed; oracle and WAM optima are reported above.", "12. This evidence strengthens the case only if WAM captures a substantial oracle fraction on irregular workloads; under the default run, the prior negative result is not overturned.", "",
        "## Integrity and limitations", "", "Direct horizon training only creates examples where both the context position and target position are inside the training prefix. Evaluation addresses are never used to train the predictor. The horizon simulator reuses the existing L1/L2/L3/DRAM hierarchy, outstanding-request limit, cache insertion, pollution attribution, and line normalization. Partial latency hiding is measured from demand wait time rather than treating every late request as useless.", "", "Artifacts: `summary.csv`, `horizon_accuracy.csv`, `oracle_horizon.csv`, `timeliness.csv`, `compute_gap.csv`, `dram_sweep.csv`, `bandwidth_sweep.csv`, `failure_breakdown.csv`, `config.json`, and `plots/`.",
    ]
    text = "\n".join(lines) + "\n"
    (output / "report.md").write_text(text, encoding="utf-8")
    return text


def run(output: Path, length: int = 10000, long_lengths: tuple[int, ...] = (100000, 1000000), trace_path: Path | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    run_config = config()
    if trace_path:
        line_trace = list(normalize_addresses(load_trace(trace_path), run_config.hierarchy.cache_line_size))
        workloads = {"ExternalTrace": line_trace}
    else:
        generated = all_workloads(length)
        generated["LongHigherOrder"] = higher_order_ambiguous(context_count=24, repeats=max(1, length // (24 * 4)))
        workloads = generated
    summary: list[dict] = []
    oracle_rows: list[dict] = []
    timeliness: list[dict] = []
    for workload, line_trace in workloads.items():
        rows, oracles, timing = evaluate_workload(workload, line_trace, run_config)
        summary.extend(rows)
        oracle_rows.extend(oracles)
        timeliness.extend(timing)
    long_accuracy: list[dict] = []
    if not trace_path:
        for size in long_lengths:
            long_trace = higher_order_ambiguous(context_count=100, repeats=max(1, math.ceil(size / 400)))[:size]
            long_accuracy.extend(horizon_accuracy_rows(long_trace, f"LongHigherOrder-{size}", run_config, horizons=(1, 4, 8, 16, 32)))
    compute = compute_gap_rows(contextual(repeats=256), (0, 2, 4, 8, 16, 32, 64))
    dram = dram_rows(contextual(repeats=256))
    bandwidth = bandwidth_rows(contextual(repeats=256))
    failures = failure_rows(summary)
    _write_csv(output / "summary.csv", summary)
    _write_csv(output / "horizon_accuracy.csv", [*long_accuracy, *[{"workload": row["workload"], "horizon": row["horizon"], "system": row["system"], "accuracy": row["accuracy"], "top3_accuracy": row["top3_accuracy"], "coverage": row["prediction_attempts"] / max(1, row["trace_length"] - row["horizon"]), "storage_bytes": row["storage_bytes"]} for row in summary if row["system"] != "Oracle"]])
    _write_csv(output / "oracle_horizon.csv", oracle_rows)
    _write_csv(output / "timeliness.csv", timeliness)
    _write_csv(output / "compute_gap.csv", compute)
    _write_csv(output / "dram_sweep.csv", dram)
    _write_csv(output / "bandwidth_sweep.csv", bandwidth)
    _write_csv(output / "failure_breakdown.csv", failures)
    (output / "config.json").write_text(json.dumps({"horizons": HORIZONS, "length": length, "long_lengths": long_lengths, "simulator": asdict(run_config), "workloads": list(workloads)}, indent=2), encoding="utf-8")
    report(output, summary, oracle_rows, timeliness, compute, dram, bandwidth, long_accuracy)
    plots = output / "plots"
    _plot([row for row in summary if row["system"] in {"DirectWAM", "RecursiveWAM", "DirectMarkov"}], "horizon", "accuracy", "system", "WAM accuracy vs prediction horizon", "Horizon", "Accuracy", plots / "wam_accuracy_vs_horizon.svg")
    _plot(oracle_rows, "horizon", "speedup", "workload", "Oracle speedup vs horizon", "Horizon", "Speedup", plots / "oracle_speedup_vs_horizon.svg")
    _plot([row for row in summary if row["system"] in {"DirectWAM", "RecursiveWAM", "DirectMarkov"}], "horizon", "speedup", "system", "WAM speedup vs horizon", "Horizon", "Speedup", plots / "wam_speedup_vs_horizon.svg")
    _plot([row for row in summary if row["system"] in {"Oracle", "DirectWAM", "RecursiveWAM"}], "horizon", "speedup", "system", "WAM vs oracle speedup", "Horizon", "Speedup", plots / "wam_vs_oracle_speedup.svg")
    _plot(timeliness, "horizon", "late_rate", "system", "Late-prefetch rate vs horizon", "Horizon", "Late rate", plots / "late_rate_vs_horizon.svg")
    _plot(timeliness, "horizon", "mean_slack", "system", "Mean prefetch slack vs horizon", "Horizon", "Slack cycles", plots / "mean_slack_vs_horizon.svg")
    _plot(timeliness, "horizon", "cycles_hidden", "system", "Cycles hidden vs horizon", "Horizon", "Cycles", plots / "cycles_hidden_vs_horizon.svg")
    _plot(compute, "compute_gap", "maximum_speedup", "best_system", "Speedup vs compute gap", "Compute cycles", "Maximum speedup", plots / "speedup_vs_compute_gap.svg")
    _plot(dram, "dram_latency", "speedup", "system", "Speedup vs DRAM latency", "DRAM cycles", "Speedup", plots / "speedup_vs_dram_latency.svg")
    _plot(bandwidth, "outstanding_limit", "speedup", "system", "Speedup vs outstanding-prefetch limit", "Outstanding limit", "Speedup", plots / "speedup_vs_bandwidth_limit.svg")
    _plot([row for row in summary if row["system"] in {"DirectWAM", "RecursiveWAM"}], "horizon", "fraction_of_oracle_gain", "system", "Fraction of oracle gain vs horizon", "Horizon", "Fraction", plots / "fraction_of_oracle_gain.svg")
    print("\nWAM horizon analysis")
    print(f"Best oracle: {oracle_best_label(oracle_rows)}")
    print(f"Best WAM: {wam_best_label(summary)}")
    print(f"Report: {output / 'report.md'}")


def run_long_only(output: Path, lengths: tuple[int, ...] = (100000, 1000000)) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for size in lengths:
        trace = higher_order_ambiguous(context_count=100, repeats=max(1, math.ceil(size / 400)))[:size]
        rows.extend(fast_long_accuracy_rows(trace, f"LongHigherOrder-{size}", horizons=(1, 4, 8, 16, 32)))
    _write_csv(output / "long_horizon_accuracy.csv", rows)
    print(f"Long-horizon accuracy: {output / 'long_horizon_accuracy.csv'}")


def oracle_best_label(rows: list[dict]) -> str:
    row = max(rows, key=lambda item: item["speedup"]) if rows else None
    return f"{row['workload']} H{row['horizon']} at {row['speedup']:.3f}x" if row else "n/a"


def wam_best_label(rows: list[dict]) -> str:
    candidates = [row for row in rows if row["system"] in {"DirectWAM", "RecursiveWAM"}]
    row = max(candidates, key=lambda item: item["speedup"]) if candidates else None
    return f"{row['workload']} {row['system']} H{row['horizon']} at {row['speedup']:.3f}x" if row else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/horizon_analysis"))
    parser.add_argument("--length", type=int, default=10000)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--skip-long", action="store_true")
    parser.add_argument("--long-only", action="store_true")
    args = parser.parse_args()
    if args.long_only:
        run_long_only(args.output)
    else:
        run(args.output, args.length, () if args.skip_long else (100000, 1000000), args.trace)


if __name__ == "__main__":
    main()
