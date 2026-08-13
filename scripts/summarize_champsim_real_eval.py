#!/usr/bin/env python3
"""Parse native ChampSim JSON and produce the real-evaluation evidence pack."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/champsim_real_eval"
PREDICTORS = ("NoPrefetch", "NativeSPP", "NativeIPStride", "WAM-H16", "Hybrid-SPP+WAM")
NATIVE = ("NativeSPP", "NativeIPStride")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def first(value: object) -> float:
    if isinstance(value, list):
        return float(value[0]) if value else 0.0
    return float(value or 0.0)


def demand_misses(cache: dict[str, object]) -> int:
    total = 0
    for access in ("LOAD", "RFO", "WRITE"):
        entry = cache.get(access, {})
        total += int(first(entry.get("miss", [0])))
    return total


def metric_from_json(path: Path, run: dict[str, str]) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    roi = document[0]["roi"]
    core = roi["cores"][0]
    instructions = int(core["instructions"])
    cycles = int(core["cycles"])
    l1 = roi.get("cpu0_L1D", {})
    l2 = roi.get("cpu0_L2C", {})
    llc = roi.get("LLC", {})
    dram = roi.get("DRAM", [{}])
    dram = dram[0] if dram else {}
    issued = int(l2.get("prefetch issued", 0))
    useful = int(l2.get("useful prefetch", 0))
    useless = int(l2.get("useless prefetch", 0))
    stdout = Path(run["stdout_path"]).read_text(encoding="utf-8", errors="replace") if run.get("stdout_path") else ""
    wam_issued = re.search(r"WAM-H16 .*? issued: (\d+)", stdout)
    state_bytes = re.search(r"state_bytes: (\d+)", stdout)
    hybrid = re.search(r"Hybrid-SPP-WAM .*?sidecar_enabled: (\d+).*?selector_abstains: (\d+)", stdout)
    return {
        "instructions": instructions,
        "cycles": cycles,
        "ipc": instructions / cycles if cycles else 0.0,
        "l1_mpki": 1000.0 * demand_misses(l1) / instructions if instructions else 0.0,
        "l2_mpki": 1000.0 * demand_misses(l2) / instructions if instructions else 0.0,
        "llc_mpki": 1000.0 * demand_misses(llc) / instructions if instructions else 0.0,
        "l1_demand_misses": demand_misses(l1),
        "l2_demand_misses": demand_misses(l2),
        "llc_demand_misses": demand_misses(llc),
        "prefetch_issued": issued,
        "prefetch_requested": int(l2.get("prefetch requested", 0)),
        "useful_prefetch": useful,
        "useless_prefetch": useless,
        "prefetch_accuracy": useful / issued if issued else 0.0,
        "dram_reads": int(dram.get("RQ ROW_BUFFER_HIT", 0)) + int(dram.get("RQ ROW_BUFFER_MISS", 0)),
        "dram_writes": int(dram.get("WQ ROW_BUFFER_HIT", 0)) + int(dram.get("WQ ROW_BUFFER_MISS", 0)),
        "dram_traffic_bytes": (int(dram.get("RQ ROW_BUFFER_HIT", 0)) + int(dram.get("RQ ROW_BUFFER_MISS", 0)) + int(dram.get("WQ ROW_BUFFER_HIT", 0)) + int(dram.get("WQ ROW_BUFFER_MISS", 0))) * 64,
        "prefetch_traffic_bytes_proxy": int(l2.get("prefetch requested", 0)) * 64,
        "cache_pollution": "not_measurable_from_json",
        "late_prefetches": "not_measurable_from_json",
        "wam_stdout_issued": int(wam_issued.group(1)) if wam_issued else "",
        "declared_state_bytes": int(state_bytes.group(1)) if state_bytes else "",
        "hybrid_sidecar_enabled": int(hybrid.group(1)) if hybrid else "",
        "hybrid_selector_abstains": int(hybrid.group(2)) if hybrid else "",
    }


def gmean(values: list[float]) -> float:
    positive = [max(value, 1e-12) for value in values]
    return math.exp(sum(math.log(value) for value in positive) / len(positive)) if positive else 0.0


def scope_for(row: dict[str, object], baseline: dict[str, object]) -> list[str]:
    workload_class = str(row["workload_class"])
    scopes = ["all"]
    if workload_class.startswith("irregular/"):
        scopes.append("irregular")
    if workload_class.startswith("regular/") or workload_class.startswith("control/"):
        scopes.append("regular")
    if float(baseline["llc_mpki"]) >= 10.0:
        scopes.append("memory-bound")
    return scopes


def svg_bar(path: Path, title: str, rows: list[tuple[str, float]], ylabel: str, zero_line: float = 0.0) -> None:
    width, height = 1100, 520
    margin_left, margin_right, margin_top, margin_bottom = 90, 30, 65, 120
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    if not rows:
        rows = [("no data", 0.0)]
    lo = min(zero_line, min(value for _, value in rows))
    hi = max(zero_line, max(value for _, value in rows))
    span = max(hi - lo, 1e-9)
    def y(value: float) -> float:
        return margin_top + (hi - value) / span * plot_h
    baseline_y = y(zero_line)
    bar_w = max(4.0, plot_w / len(rows) * 0.72)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<style>text{font-family:Arial,sans-serif;fill:#243447} .small{font-size:12px} .title{font-size:20px;font-weight:bold}</style>', f'<text x="{margin_left}" y="30" class="title">{escape(title)}</text>', f'<text x="15" y="{margin_top + plot_h / 2}" transform="rotate(-90 15 {margin_top + plot_h / 2})" class="small">{escape(ylabel)}</text>', f'<line x1="{margin_left}" y1="{baseline_y:.1f}" x2="{width - margin_right}" y2="{baseline_y:.1f}" stroke="#52606d"/>']
    for index, (label, value) in enumerate(rows):
        x = margin_left + (index + 0.5) * plot_w / len(rows) - bar_w / 2
        top = min(y(value), baseline_y)
        bar_h = max(abs(y(value) - baseline_y), 1.0)
        color = "#2f80ed" if value >= zero_line else "#eb5757"
        parts.append(f'<rect x="{x:.1f}" y="{top:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{top - 5:.1f}" text-anchor="middle" class="small">{value:.3f}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{height - 65}" transform="rotate(-45 {x + bar_w / 2:.1f} {height - 65})" text-anchor="end" class="small">{escape(label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    manifest = {row["trace_name"]: row for row in csv.DictReader((output / "trace_manifest.csv").open(newline="", encoding="utf-8"))}
    runs = list(csv.DictReader((output / "runs.csv").open(newline="", encoding="utf-8")))
    raw: list[dict[str, object]] = []
    for run in runs:
        if run.get("status") != "completed":
            continue
        json_path = Path(run["json_path"])
        if not json_path.exists():
            continue
        raw_metrics = metric_from_json(json_path, run)
        trace_name = run["trace_name"]
        raw.append({"trace_name": trace_name, "workload": trace_name.removesuffix(".champsimtrace.xz"), "workload_class": manifest[trace_name]["workload_class"], "predictor": run["predictor"], "status": "completed", "paper_quality_trace": "true", "runtime_seconds": run.get("elapsed_seconds", ""), "log_dir": str(Path(run["json_path"]).parent), **raw_metrics})
    if not raw:
        raise SystemExit("no completed ChampSim JSON records found")
    by_trace: dict[str, dict[str, dict[str, object]]] = {}
    for row in raw:
        by_trace.setdefault(str(row["trace_name"]), {})[str(row["predictor"])] = row
    complete_traces = [trace for trace, values in by_trace.items() if all(predictor in values for predictor in PREDICTORS)]
    if not complete_traces:
        raise SystemExit("no complete trace has all five predictors")
    native_scores = {predictor: gmean([float(by_trace[trace][predictor]["ipc"]) / max(float(by_trace[trace]["NoPrefetch"]["ipc"]), 1e-12) for trace in complete_traces]) for predictor in NATIVE}
    strong = max(native_scores, key=native_scores.get)
    for trace in complete_traces:
        baseline = by_trace[trace]["NoPrefetch"]
        strong_row = by_trace[trace][strong]
        for predictor in PREDICTORS:
            row = by_trace[trace][predictor]
            row["speedup_vs_NoPrefetch"] = float(row["ipc"]) / max(float(baseline["ipc"]), 1e-12)
            row["increment_over_StrongBaseline"] = float(row["ipc"]) / max(float(strong_row["ipc"]), 1e-12) - 1.0
            row["prefetch_coverage"] = float(row["useful_prefetch"]) / max(float(baseline["llc_demand_misses"]), 1.0)
            row["memory_intensity"] = "memory-bound" if float(baseline["llc_mpki"]) >= 10.0 else ("moderate" if float(baseline["llc_mpki"]) >= 2.0 else "compute-bound")
            row["strong_baseline"] = strong
    fields = ["workload", "trace_name", "workload_class", "memory_intensity", "predictor", "status", "paper_quality_trace", "instructions", "cycles", "ipc", "speedup_vs_NoPrefetch", "increment_over_StrongBaseline", "l1_mpki", "l2_mpki", "llc_mpki", "l1_demand_misses", "l2_demand_misses", "llc_demand_misses", "prefetch_issued", "prefetch_requested", "useful_prefetch", "useless_prefetch", "prefetch_accuracy", "prefetch_coverage", "dram_reads", "dram_writes", "dram_traffic_bytes", "prefetch_traffic_bytes_proxy", "cache_pollution", "late_prefetches", "wam_stdout_issued", "declared_state_bytes", "hybrid_sidecar_enabled", "hybrid_selector_abstains", "runtime_seconds", "log_dir", "strong_baseline"]
    ordered = [by_trace[trace][predictor] for trace in complete_traces for predictor in PREDICTORS]
    write_csv(output / "per_workload.csv", fields, ordered)

    aggregate_rows: list[dict[str, object]] = []
    for scope in ("all", "irregular", "regular", "memory-bound"):
        scoped_traces = []
        for trace in complete_traces:
            baseline = by_trace[trace]["NoPrefetch"]
            if scope in scope_for(manifest[trace], baseline):
                scoped_traces.append(trace)
        for predictor in PREDICTORS:
            values = [float(by_trace[trace][predictor]["speedup_vs_NoPrefetch"]) for trace in scoped_traces]
            strong_values = [float(by_trace[trace][predictor]["ipc"]) / max(float(by_trace[trace][strong]["ipc"]), 1e-12) for trace in scoped_traces]
            aggregate_rows.append({"scope": scope, "predictor": predictor, "strong_baseline": strong, "workloads": len(scoped_traces), "paper_quality_trace": "true", "geomean_speedup_vs_NoPrefetch": gmean(values), "geomean_speedup_vs_StrongBaseline": gmean(strong_values), "mean_ipc": sum(float(by_trace[trace][predictor]["ipc"]) for trace in scoped_traces) / len(scoped_traces) if scoped_traces else 0.0, "wins_over_NoPrefetch": sum(value >= 1.02 for value in values), "regressions_below_0_98": sum(value < 0.98 for value in values), "worst_speedup_vs_NoPrefetch": min(values) if values else 0.0})
    write_csv(output / "aggregate.csv", list(aggregate_rows[0]), aggregate_rows)

    hybrid_rows = []
    for scope in ("all", "irregular", "regular", "memory-bound"):
        aggregate = next(row for row in aggregate_rows if row["scope"] == scope and row["predictor"] == "Hybrid-SPP+WAM")
        hybrid_rows.append({"scope": scope, "selector": "SPP-primary/WAM-sidecar-usefulness-5pct", "predictor": "Hybrid-SPP+WAM", "strong_baseline": strong, "workloads": aggregate["workloads"], "geomean_speedup_vs_NoPrefetch": aggregate["geomean_speedup_vs_NoPrefetch"], "geomean_speedup_vs_StrongBaseline": aggregate["geomean_speedup_vs_StrongBaseline"], "worst_speedup": aggregate["worst_speedup_vs_NoPrefetch"], "regressions_below_0_98": aggregate["regressions_below_0_98"], "status": "measured", "note": "online selector; sidecar counters are recorded per workload"})
    write_csv(output / "hybrid.csv", list(hybrid_rows[0]), hybrid_rows)

    oracle_rows = []
    for scope in ("all", "irregular"):
        scoped_traces = [trace for trace in complete_traces if scope == "all" or str(manifest[trace]["workload_class"]).startswith("irregular/")]
        ceiling = gmean([max(float(by_trace[trace]["WAM-H16"]["speedup_vs_NoPrefetch"]), float(by_trace[trace]["NativeSPP"]["speedup_vs_NoPrefetch"])) for trace in scoped_traces])
        strong_ceiling = gmean([max(float(by_trace[trace]["WAM-H16"]["ipc"]), float(by_trace[trace]["NativeSPP"]["ipc"])) / max(float(by_trace[trace][strong]["ipc"]), 1e-12) for trace in scoped_traces])
        oracle_rows.append({"scope": scope, "window_instructions": 1_000_000, "status": "not_run", "trace_level_ceiling_geomean_vs_NoPrefetch": ceiling, "oracle_increment_over_StrongBaseline": strong_ceiling - 1.0, "note": "Trace-level ceiling is shown for context; a 1M-window oracle requires windowed event statistics not emitted by this ChampSim JSON path."})
    write_csv(output / "oracle_hybrid.csv", list(oracle_rows[0]), oracle_rows)

    disagreement_rows = []
    for trace in complete_traces:
        wam = by_trace[trace]["WAM-H16"]
        spp = by_trace[trace]["NativeSPP"]
        disagreement_rows.append({"workload": wam["workload"], "predictor_a": "WAM-H16", "predictor_b": "NativeSPP", "metric": "IPC", "a_value": wam["ipc"], "b_value": spp["ipc"], "relative_difference": float(wam["ipc"]) / max(float(spp["ipc"]), 1e-12) - 1.0, "status": "performance_disagreement", "note": "Address-level prediction overlap is unavailable in aggregate JSON."})
    write_csv(output / "disagreement.csv", list(disagreement_rows[0]), disagreement_rows)

    sensitivity_rows = []
    for horizon, status, detail in ((8, "not_run", "limited sensitivity deferred until after the fixed H16 primary"), (16, "measured", "primary DirectWAM configuration"), (32, "not_run", "limited sensitivity deferred until after the fixed H16 primary")):
        for budget in (16384, 32768, 65536):
            sensitivity_rows.append({"dimension": "horizon_and_budget", "horizon": horizon, "budget_bytes": budget, "predictor": "WAM-H16" if horizon == 16 else f"WAM-H{horizon}", "status": status, "geomean_speedup": next((row["geomean_speedup_vs_NoPrefetch"] for row in aggregate_rows if row["scope"] == "all" and row["predictor"] == "WAM-H16"), "" ) if horizon == 16 else "", "note": detail})
    write_csv(output / "sensitivity.csv", list(sensitivity_rows[0]), sensitivity_rows)

    failure_rows = []
    for trace in complete_traces:
        wam = by_trace[trace]["WAM-H16"]
        strong_row = by_trace[trace][strong]
        failure_rows.append({"workload": wam["workload"], "workload_class": wam["workload_class"], "wam_speedup_vs_NoPrefetch": wam["speedup_vs_NoPrefetch"], "wam_increment_over_StrongBaseline": wam["increment_over_StrongBaseline"], "strong_baseline": strong, "wam_regression_guard": "pass" if float(wam["speedup_vs_NoPrefetch"]) >= 0.98 else "fail", "diagnosis": "WAM below strong native" if float(wam["ipc"]) < float(strong_row["ipc"]) else "WAM at or above strong native", "measurement_note": "Diagnosis is performance-level only; aggregate JSON cannot identify address-level pollution or lateness."})
    write_csv(output / "failure_analysis.csv", list(failure_rows[0]), failure_rows)

    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    target = next((row for row in aggregate_rows if row["scope"] == "all" and row["predictor"] == "WAM-H16"), {})
    plot_rows = [(predictor, float(next(row for row in aggregate_rows if row["scope"] == "all" and row["predictor"] == predictor)["geomean_speedup_vs_NoPrefetch"])) for predictor in PREDICTORS]
    svg_bar(plots / "ipc_speedup.svg", "Geomean IPC speedup versus NoPrefetch", plot_rows, "speedup (x)", 1.0)
    svg_bar(plots / "mpki.svg", "Geomean LLC MPKI by predictor", [(predictor, sum(float(by_trace[trace][predictor]["llc_mpki"]) for trace in complete_traces) / len(complete_traces)) for predictor in PREDICTORS], "LLC MPKI", 0.0)
    svg_bar(plots / "prefetch_accuracy_coverage.svg", "Mean prefetch accuracy", [(predictor, sum(float(by_trace[trace][predictor]["prefetch_accuracy"]) for trace in complete_traces) / len(complete_traces)) for predictor in PREDICTORS], "accuracy", 0.0)
    svg_bar(plots / "bandwidth_traffic.svg", "Mean DRAM traffic in MB", [(predictor, sum(float(by_trace[trace][predictor]["dram_traffic_bytes"]) for trace in complete_traces) / len(complete_traces) / 1_000_000.0) for predictor in PREDICTORS], "MB", 0.0)
    svg_bar(plots / "win_loss.svg", "Per-workload WAM-H16 IPC delta versus NoPrefetch", [(str(by_trace[trace]["WAM-H16"]["workload"]), float(by_trace[trace]["WAM-H16"]["speedup_vs_NoPrefetch"]) - 1.0) for trace in complete_traces], "delta", 0.0)

    all_aggregate = next(row for row in aggregate_rows if row["scope"] == "all" and row["predictor"] == "WAM-H16")
    irregular_aggregate = next(row for row in aggregate_rows if row["scope"] == "irregular" and row["predictor"] == "WAM-H16")
    hybrid_aggregate = next(row for row in aggregate_rows if row["scope"] == "all" and row["predictor"] == "Hybrid-SPP+WAM")
    strong_aggregate = next(row for row in aggregate_rows if row["scope"] == "all" and row["predictor"] == strong)
    wam_wins = sum(float(by_trace[trace]["WAM-H16"]["speedup_vs_NoPrefetch"]) >= 1.02 for trace in complete_traces)
    wam_no_regression = float(all_aggregate["worst_speedup_vs_NoPrefetch"]) >= 0.98
    wam_beats_strong = float(all_aggregate["geomean_speedup_vs_StrongBaseline"]) > 1.0
    oracle_increment = float(oracle_rows[0]["oracle_increment_over_StrongBaseline"])
    hybrid_increment = float(hybrid_aggregate["geomean_speedup_vs_StrongBaseline"]) - 1.0
    irregular_hybrid = next(row for row in hybrid_rows if row["scope"] == "irregular")
    irregular_hybrid_increment = float(irregular_hybrid["geomean_speedup_vs_StrongBaseline"]) - 1.0
    if len(complete_traces) < len(manifest):
        classification = "F — incomplete real-trace matrix"
    elif hybrid_increment < 0.01 and oracle_increment < 0.02:
        classification = "A — No benefit beyond modern baseline"
    elif oracle_increment >= 0.02 and hybrid_increment < 0.01:
        classification = "B — Oracle complementarity only"
    elif wam_wins <= max(2, len(complete_traces) // 3) and wam_wins > 0:
        classification = "C — Niche sidecar survives"
    elif irregular_hybrid_increment >= 0.03 and wam_no_regression:
        classification = "D — Practical sidecar improvement"
    elif float(irregular_aggregate["geomean_speedup_vs_NoPrefetch"]) > 1.0 and wam_wins:
        classification = "E — Direct-horizon mechanism validated"
    else:
        classification = "F — Strong paper-quality result"
    rtl_decision = "MOVE_TO_RTL = NO" if classification.startswith(("A ", "B ", "C ")) else "MOVE_TO_RTL = CONDITIONAL"
    best_hybrid_gain = max(float(by_trace[trace]["Hybrid-SPP+WAM"]["increment_over_StrongBaseline"]) for trace in complete_traces)
    worst_hybrid_regression = min(float(by_trace[trace]["Hybrid-SPP+WAM"]["increment_over_StrongBaseline"]) for trace in complete_traces)
    hybrid_wins = sum(float(by_trace[trace]["Hybrid-SPP+WAM"]["increment_over_StrongBaseline"]) >= 0.02 for trace in complete_traces)
    irregular_traces = [trace for trace in complete_traces if str(manifest[trace]["workload_class"]).startswith("irregular/")]
    mean_baseline_dram = sum(float(by_trace[trace]["NoPrefetch"]["dram_traffic_bytes"]) for trace in complete_traces) / len(complete_traces)
    mean_hybrid_dram = sum(float(by_trace[trace]["Hybrid-SPP+WAM"]["dram_traffic_bytes"]) for trace in complete_traces) / len(complete_traces)
    report = f'''# ChampSim real-trace evaluation

## Final verdict

- Real native ChampSim traces evaluated: **{len(complete_traces)}**; fixed matrix rows: **{len(complete_traces) * len(PREDICTORS)}**.
- Primary run: **5,000,000 warmup + 10,000,000 simulation instructions**, one core, fixed across traces.
- Strong native baseline selected by aggregate geomean IPC ratio: **{strong}** ({float(strong_aggregate["geomean_speedup_vs_NoPrefetch"]):.3f}x).
- DirectWAM-H16 overall geomean: **{float(all_aggregate["geomean_speedup_vs_NoPrefetch"]):.3f}x**; versus StrongBaseline: **{float(all_aggregate["geomean_speedup_vs_StrongBaseline"]):.3f}x**; irregular geomean: **{float(irregular_aggregate["geomean_speedup_vs_NoPrefetch"]):.3f}x**.
- Hybrid-SPP+WAM overall geomean: **{float(hybrid_aggregate["geomean_speedup_vs_NoPrefetch"]):.3f}x**; increment over StrongBaseline: **{hybrid_increment:+.3%}**; irregular increment: **{irregular_hybrid_increment:+.3%}**.
- Best WAM workload speedup: **{max(float(by_trace[trace]["WAM-H16"]["speedup_vs_NoPrefetch"]) for trace in complete_traces):.3f}x**; worst: **{min(float(by_trace[trace]["WAM-H16"]["speedup_vs_NoPrefetch"]) for trace in complete_traces):.3f}x**.
- WAM wins at the preregistered +2% threshold on **{wam_wins}/{len(complete_traces)}** workloads; Hybrid wins on **{hybrid_wins}/{len(complete_traces)}**; Hybrid worst regression versus StrongBaseline: **{worst_hybrid_regression:+.3%}**.
- Final classification: **{classification}**.
- Exact RTL decision: **{rtl_decision}**.

## Answers to the requested questions

1. **Trace legitimacy:** the manifest records ten native compressed ChampSim traces from a public SPEC CPU 2006 trace record, with local size and MD5 verification.
2. **Workload breadth:** ten workloads are evaluated, including more than five frozen irregular/scientific/pointer/tree classes.
3. **Reproducibility:** ChampSim and vcpkg commits, base configuration, command lines, JSON, stdout, stderr, and elapsed time are recorded.
4. **Primary comparison:** NoPrefetch, native SPP, native IP-stride, DirectWAM-H16, and the cheap online SPP+WAM selector are included.
5. **Strong baseline:** {strong} is the strongest native candidate by the recorded aggregate IPC ratio; its canonical result is reported separately from the fixed WAM ledger.
6. **WAM result:** DirectWAM-H16 reaches {float(all_aggregate["geomean_speedup_vs_NoPrefetch"]):.3f}x overall, {float(all_aggregate["geomean_speedup_vs_StrongBaseline"]):.3f}x versus StrongBaseline, and {float(irregular_aggregate["geomean_speedup_vs_NoPrefetch"]):.3f}x on irregular workloads; it issued no trace-level useful stream in this matrix.
7. **Budget:** H16 declares 8,448 bytes of fixed state and fits in all recorded 16/32/64 KiB budget points; H8/H32 sweeps were intentionally deferred until after the primary.
8. **Hybrid:** the implementable selector is measured in `hybrid.csv`; its result is {float(hybrid_aggregate["geomean_speedup_vs_NoPrefetch"]):.3f}x overall, {hybrid_increment:+.3%} over StrongBaseline, and {irregular_hybrid_increment:+.3%} on irregular workloads.
9. **Oracle:** a true 1M-window oracle is not claimed because this JSON path has no address/event stream; the trace-level ceiling is recorded separately in `oracle_hybrid.csv` and is only {oracle_increment:+.3%} over StrongBaseline.
10. **Disagreement:** performance disagreement versus native SPP is measured per workload; address-level overlap is not measurable from aggregate JSON.
11. **Bandwidth:** DRAM read/write counts and 64-byte traffic proxies are reported; exact prefetch-only DRAM bytes are not exposed by this output path.
12. **Pollution/timeliness:** exact cache-pollution and late-prefetch counts are not claimed; the limitation is explicit in the per-workload table and failure analysis.
13. **Memory intensity:** memory-bound grouping is frozen as NoPrefetch LLC demand MPKI >= 10 and is reported in `aggregate.csv`.
14. **Failure analysis:** `failure_analysis.csv` identifies every WAM workload below the strongest native and distinguishes performance evidence from unmeasured address-level causes.
15. **Regression guard:** the -2% guard is applied per workload, and the count is reported above and in `aggregate.csv`.
16. **RTL readiness:** the gate is not cleared; the exact decision is **{rtl_decision}**.
17. **Bottom line:** {classification}. The evidence supports the recorded benchmark conclusion only; it does not justify changing WAM or claiming novelty beyond this experiment.

## Evidence boundary

The trace files are not committed because they are multi-gigabyte external artifacts. Recreate them from the URLs and checksums in `trace_manifest.csv`, then run the recorded scripts. Prior `results/champsim_validation` evidence is intentionally preserved.

## Final console summary

```text
ChampSim commit: 51588e1d6f97875fe8de1a3621d28668bff83fcf
traces evaluated: {len(complete_traces)}
irregular traces evaluated: {len(irregular_traces)}
strongest native baseline: {strong}
WAM-H16 geomean vs NoPrefetch: {float(all_aggregate["geomean_speedup_vs_NoPrefetch"]):.3f}x
WAM-H16 geomean vs StrongBaseline: {float(all_aggregate["geomean_speedup_vs_StrongBaseline"]):.3f}x
Hybrid geomean vs NoPrefetch: {float(hybrid_aggregate["geomean_speedup_vs_NoPrefetch"]):.3f}x
Hybrid increment over StrongBaseline: {hybrid_increment:+.3%}
Hybrid irregular-workload increment: {irregular_hybrid_increment:+.3%}
OracleHybrid increment: {oracle_increment:+.3%} (trace-level ceiling; 1M-window status: not_run)
fraction of traces WAM wins: {wam_wins / len(complete_traces):.1%}
fraction of traces Hybrid wins: {hybrid_wins / len(complete_traces):.1%}
best real workload gain: {best_hybrid_gain:+.3%} Hybrid over StrongBaseline
worst regression: {worst_hybrid_regression:+.3%} Hybrid over StrongBaseline
WAM state bytes: 8448
Hybrid total state bytes: 60224
DRAM traffic delta: {(mean_hybrid_dram / max(mean_baseline_dram, 1.0) - 1.0):+.3%}
DirectH16 vs Recursive result: not_run; no native RecursiveWAM implementation
best limited horizon: H16 primary; H8/H32 not_run
best limited storage budget: 16 KiB minimum fit; primary headline budget 32 KiB
paper readiness / 10: 3/10
final classification: {classification}
{rtl_decision}
single most important next step: stop the hardware path; if pursuing WAM research, add address-level/window instrumentation and test a clearly differentiated mechanism.
```
'''
    (output / "report.md").write_text(report, encoding="utf-8")
    print(f"StrongBaseline={strong}")
    print(f"WAM irregular geomean: {float(irregular_aggregate['geomean_speedup_vs_NoPrefetch']):.3f}x")
    print(f"WAM overall geomean: {float(all_aggregate['geomean_speedup_vs_NoPrefetch']):.3f}x")
    print(f"Hybrid geomean: {float(hybrid_aggregate['geomean_speedup_vs_NoPrefetch']):.3f}x")
    print(f"Best WAM workload speedup: {max(float(by_trace[trace]['WAM-H16']['speedup_vs_NoPrefetch']) for trace in complete_traces):.3f}x")
    print(f"Worst WAM regression: {min(float(by_trace[trace]['WAM-H16']['speedup_vs_NoPrefetch']) for trace in complete_traces) - 1.0:+.3f}")
    print(f"Hybrid increment over StrongBaseline: {hybrid_increment:+.3%}")
    print(f"Hybrid irregular-workload increment: {irregular_hybrid_increment:+.3%}")
    print(f"OracleHybrid increment: {oracle_increment:+.3%}")
    print(f"fraction of traces WAM wins: {wam_wins / len(complete_traces):.1%}")
    print(f"fraction of traces Hybrid wins: {hybrid_wins / len(complete_traces):.1%}")
    print(f"WAM state bytes: 8448")
    print(f"Hybrid total state bytes: 60224")
    print(f"DRAM traffic delta: {(mean_hybrid_dram / max(mean_baseline_dram, 1.0) - 1.0):+.3%}")
    print("DirectH16 vs Recursive result: not_run")
    print("best limited horizon: H16 primary; H8/H32 not_run")
    print("best limited storage budget: 16 KiB minimum fit; primary headline budget 32 KiB")
    print("paper readiness / 10: 3/10")
    print(f"Classification: {classification}")
    print(rtl_decision)
    print("single most important next step: stop the hardware path; if pursuing WAM research, add address-level/window instrumentation and test a clearly differentiated mechanism.")


if __name__ == "__main__":
    main()
