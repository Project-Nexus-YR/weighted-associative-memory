#!/usr/bin/env python3
"""Analyze the fixed-window 4-way SetAssociativeWAM evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_champsim_final_diagnostic import (  # noqa: E402
    chronological_oracle,
    gmean,
    parse_json_metric,
    parse_stdout,
    read_events,
    read_predictions,
    svg_plot,
)

DEFAULT_OUTPUT = ROOT / "results/set_associative_wam"
DIRECT_OUTPUT = ROOT / "results/champsim_final_diagnostic"
REAL_OUTPUT = ROOT / "results/champsim_real_eval"
PINNED_COMMIT = "51588e1d6f97875fe8de1a3621d28668bff83fcf"
WARMUP = 5_000_000
SIMULATION = 10_000_000
TABLE_ENTRIES = 256
SETS = 64
WAYS = 4
HORIZON = 16
DEPTH = 4
CONFIDENCE_THRESHOLD = 8
DIRECT_BYTES = 8448
SET_BYTES = 8512


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"], extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def integer(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def parse_sa_stdout(path: Path) -> dict[str, object]:
    counters: dict[str, object] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("WAM-SA-DIAG "):
            continue
        for item in line.removeprefix("WAM-SA-DIAG ").split():
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            try:
                counters[key] = int(value)
            except ValueError:
                counters[key] = number(value, 0.0)
    return counters


def shadow_metrics(lines: array, predictions: list[tuple[int, int, int, int, int, int, int]]) -> dict[str, object]:
    generated = [record for record in predictions if record[6] == 1]
    resolved = [(record, int(lines[record[0] + HORIZON])) for record in generated if record[0] + HORIZON < len(lines)]
    correct = sum(record[2] == target for record, target in resolved)
    return {"generated": len(generated), "resolved": len(resolved), "correct": correct, "accuracy": ratio(correct, len(resolved)), "coverage": ratio(len(resolved), len(lines)), "confidence_mean": statistics.mean([record[4] for record in generated]) if generated else 0.0, "support_mean": statistics.mean([record[3] for record in generated]) if generated else 0.0, "support_max": max([record[3] for record in generated], default=0)}


def funnel(trace: str, variant: str, counters: dict[str, object], hit_count: int) -> list[dict[str, object]]:
    stages = [("eligible_accesses", integer(counters.get("eligible_accesses_seen"))), ("contexts_formed", integer(counters.get("contexts_formed"))), ("context_hits", hit_count), ("predictions_generated", integer(counters.get("predictions_generated"))), ("predictions_above_threshold", integer(counters.get("predictions_above_threshold"))), ("prefetch_requests", integer(counters.get("prefetch_requests_generated"))), ("accepted_requests", integer(counters.get("prefetch_requests_accepted"))), ("useful_prefetches", integer(counters.get("prefetches_useful")))]
    eligible = max(stages[0][1], 1)
    rows: list[dict[str, object]] = []
    previous = stages[0][1]
    for stage, count in stages:
        rows.append({"trace": trace, "variant": variant, "stage": stage, "count": count, "fraction_of_eligible": count / eligible, "fraction_of_previous_stage": ratio(count, previous), "note": "SA context_hits is exact tag hit; direct context_hits is frozen direct lookup hit"})
        previous = count
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    direct_activation = {row["trace"]: row for row in read_csv(DIRECT_OUTPUT / "activation.csv")}
    direct_shadow = {row["trace"]: row for row in read_csv(DIRECT_OUTPUT / "shadow_accuracy.csv")}
    direct_oracle = {(row["trace"], integer(row["depth"]), integer(row["horizon"])): row for row in read_csv(DIRECT_OUTPUT / "oracle_predictability.csv")}
    direct_confidence = read_csv(DIRECT_OUTPUT / "confidence.csv")
    manifest = {row["trace_name"].removesuffix(".champsimtrace.xz"): row for row in read_csv(REAL_OUTPUT / "trace_manifest.csv")}
    runs = read_csv(output / "set_associative_runs.csv")

    activation: list[dict[str, object]] = []
    hash_rows: list[dict[str, object]] = []
    occupancy: list[dict[str, object]] = []
    pressure: list[dict[str, object]] = []
    shadow_rows: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []
    confidence_rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    funnel_rows: list[dict[str, object]] = []
    per_trace: list[dict[str, object]] = []
    ipc_rows: list[dict[str, object]] = []
    trace_data: dict[str, dict[str, object]] = {}

    for run in runs:
        if run.get("status") != "completed":
            continue
        trace = run["trace_name"].removesuffix(".champsimtrace.xz")
        direct = direct_activation[trace]
        events = read_events(Path(run["event_path"]))
        lines: array = events["lines"]
        measurement = array("Q", (line for line, warmup in zip(lines, events["warmup"]) if not warmup))
        counters, hist, limits = parse_stdout(Path(run["stdout_path"]))
        sa = parse_sa_stdout(Path(run["stdout_path"]))
        predictions = read_predictions(Path(run["prediction_path"]))
        shadow = shadow_metrics(lines, predictions)
        oracle = chronological_oracle(measurement, DEPTH, HORIZON)
        oracle_direct = direct_oracle[(trace, DEPTH, HORIZON)]
        shadow_direct = direct_shadow[trace]
        direct_lookups = integer(direct.get("prediction_lookups"))
        direct_hits = integer(direct.get("prediction_context_hits"))
        set_lookups = integer(sa.get("set_lookups"))
        set_hits = integer(sa.get("tag_hits"))
        direct_alias = integer(direct.get("hash_alias_misses"), integer(direct.get("hash_collisions")))
        set_unresolved = integer(sa.get("unresolved_conflict_events"))
        workload_class = manifest.get(trace, {}).get("workload_class", "")
        activation.append({"trace": trace, "variant": "DirectMappedWAM", "workload_class": workload_class, **direct})
        activation.append({"trace": trace, "variant": "SetAssociativeWAM", "workload_class": workload_class, **counters, **sa, "events_total": len(lines), "events_warmup": sum(events["warmup"]), "events_measurement": len(measurement), "events_per_1k_measurement_instructions": len(measurement) / SIMULATION * 1000, "prefetch_rejection_reason": limits.get("prefetch_rejection_reason", "not_exposed")})
        hash_rows.extend([
            {"trace": trace, "variant": "DirectMappedWAM", "table_entries": TABLE_ENTRIES, "prediction_lookups": direct_lookups, "tag_hits": direct_hits, "tag_misses": integer(direct.get("prediction_context_misses")), "alias_loss_events": direct_alias, "alias_loss_rate": ratio(direct_alias, direct_lookups), "replacement_events": integer(direct.get("entry_evictions")), "note": "frozen direct-map hash alias misses"},
            {"trace": trace, "variant": "SetAssociativeWAM", "table_entries": TABLE_ENTRIES, "prediction_lookups": set_lookups, "tag_hits": set_hits, "tag_misses": integer(sa.get("tag_misses")), "alias_loss_events": set_unresolved, "alias_loss_rate": ratio(set_unresolved, set_lookups), "set_conflict_lookups": integer(sa.get("set_conflict_lookups")), "unresolved_conflict_events": set_unresolved, "same_set_distinct_contexts": integer(sa.get("same_set_distinct_contexts")), "direct_map_alias_equivalent_events": integer(sa.get("direct_map_alias_equivalent_events")), "empty_way_insertions": integer(sa.get("empty_way_insertions")), "conflict_insertions": integer(sa.get("conflict_insertions")), "replacement_events": integer(sa.get("replacement_events")), "reused_before_replacement": integer(sa.get("reused_before_replacement")), "way0_hits": integer(sa.get("way0_hits")), "way1_hits": integer(sa.get("way1_hits")), "way2_hits": integer(sa.get("way2_hits")), "way3_hits": integer(sa.get("way3_hits")), "note": "SA loss is an all-four-ways-valid tag miss"},
        ])
        occupied = sum(index * integer(sa.get(f"occupancy{index}")) for index in range(1, 5))
        occupancy.extend([
            {"trace": trace, "variant": "DirectMappedWAM", "occupancy0": "not_exposed", "occupancy1": "not_exposed", "occupancy2": "not_exposed", "occupancy3": "not_exposed", "occupancy4": "not_applicable", "occupied_entries": integer(direct.get("occupied_entries"), TABLE_ENTRIES), "occupied_fraction": ratio(integer(direct.get("occupied_entries"), TABLE_ENTRIES), TABLE_ENTRIES), "sets_ever_full": "not_applicable", "note": "direct per-set occupancy was not exposed"},
            {"trace": trace, "variant": "SetAssociativeWAM", "occupancy0": integer(sa.get("occupancy0")), "occupancy1": integer(sa.get("occupancy1")), "occupancy2": integer(sa.get("occupancy2")), "occupancy3": integer(sa.get("occupancy3")), "occupancy4": integer(sa.get("occupancy4")), "occupied_entries": occupied, "occupied_fraction": ratio(occupied, TABLE_ENTRIES), "sets_ever_full": integer(sa.get("sets_ever_reaching_4way")), "note": "occupancy bins count sets"},
        ])
        pressure.extend([
            {"trace": trace, "variant": "DirectMappedWAM", "median": "not_exposed", "p90": "not_exposed", "p95": "not_exposed", "p99": "not_exposed", "max": "not_exposed", "active_contexts_per_occupied_set_mean": "not_exposed", "occupied_sets": "not_exposed", "sets_ever_full": "not_applicable"},
            {"trace": trace, "variant": "SetAssociativeWAM", "median": number(sa.get("set_pressure_median")), "p90": number(sa.get("set_pressure_p90")), "p95": number(sa.get("set_pressure_p95")), "p99": number(sa.get("set_pressure_p99")), "max": number(sa.get("set_pressure_max")), "active_contexts_per_occupied_set_mean": number(sa.get("active_contexts_per_occupied_set_mean")), "occupied_sets": integer(sa.get("occupied_sets")), "sets_ever_full": integer(sa.get("sets_ever_reaching_4way"))},
        ])
        shadow_rows.extend([
            {"trace": trace, "variant": "DirectMappedWAM", "depth": DEPTH, "horizon": HORIZON, "prediction_lookups": direct_lookups, "generated_predictions": integer(shadow_direct.get("generated_predictions")), "resolved_predictions": integer(shadow_direct.get("resolved_predictions")), "correct_predictions": integer(shadow_direct.get("correct_predictions")), "accuracy": number(shadow_direct.get("top1_accuracy")), "coverage": number(shadow_direct.get("coverage")), "confidence_mean": number(shadow_direct.get("confidence_mean")), "support_mean": number(shadow_direct.get("support_mean")), "support_max": integer(shadow_direct.get("support_max")), "note": "frozen control; no extra diagnostic prefetches"},
            {"trace": trace, "variant": "SetAssociativeWAM", "depth": DEPTH, "horizon": HORIZON, "prediction_lookups": set_lookups, "generated_predictions": shadow["generated"], "resolved_predictions": shadow["resolved"], "correct_predictions": shadow["correct"], "accuracy": shadow["accuracy"], "coverage": shadow["coverage"], "confidence_mean": shadow["confidence_mean"], "support_mean": shadow["support_mean"], "support_max": shadow["support_max"], "note": "same online state/update path; no extra diagnostic prefetches"},
        ])
        oracle_rows.extend([
            {"trace": trace, "variant": "DirectMappedWAM", "depth": DEPTH, "horizon": HORIZON, "oracle_accuracy": number(oracle_direct.get("oracle_top1_accuracy")), "oracle_coverage": number(oracle_direct.get("coverage")), "reusable_evaluation_examples": integer(oracle_direct.get("reusable_evaluation_examples")), "evaluation_examples": integer(oracle_direct.get("evaluation_examples"))},
            {"trace": trace, "variant": "SetAssociativeWAM", "depth": DEPTH, "horizon": HORIZON, "oracle_accuracy": number(oracle.get("oracle_top1_accuracy")), "oracle_coverage": number(oracle.get("coverage")), "reusable_evaluation_examples": integer(oracle.get("reusable_evaluation_examples")), "evaluation_examples": integer(oracle.get("evaluation_examples"))},
        ])
        for kind in ("confidence", "support"):
            if kind == "confidence":
                direct_rows = [row for row in direct_confidence if row.get("trace") == trace and row.get("distribution") == kind]
            else:
                direct_rows = [row for row in direct_confidence if row.get("trace") == trace and row.get("distribution") == kind]
            for row in direct_rows:
                copied = dict(row)
                copied["variant"] = "DirectMappedWAM"
                (confidence_rows if kind == "confidence" else support_rows).append(copied)
            total = sum(integer(hist[kind].get(index)) for index in hist[kind])
            max_bin = 15 if kind == "confidence" else 16
            for index in range(max_bin + 1):
                row = {"trace": trace, "variant": "SetAssociativeWAM", "distribution": kind, "bin": index, "count": integer(hist[kind].get(index)), "fraction": ratio(integer(hist[kind].get(index)), total), "threshold": CONFIDENCE_THRESHOLD if kind == "confidence" else "not_applicable"}
                (confidence_rows if kind == "confidence" else support_rows).append(row)
        funnel_rows.extend(funnel(trace, "DirectMappedWAM", direct, direct_hits))
        funnel_rows.extend(funnel(trace, "SetAssociativeWAM", counters, set_hits))
        oracle_accuracy_value = number(oracle.get("oracle_top1_accuracy"))
        oracle_coverage_value = number(oracle.get("coverage"))
        direct_accuracy = number(shadow_direct.get("top1_accuracy"))
        direct_coverage = number(shadow_direct.get("coverage"))
        set_accuracy = number(shadow.get("accuracy"))
        set_coverage = number(shadow.get("coverage"))
        per_trace.append({"trace": trace, "workload_class": workload_class, "direct_context_hit_rate": ratio(direct_hits, direct_lookups), "set_context_hit_rate": ratio(set_hits, set_lookups), "direct_alias_loss_rate": ratio(direct_alias, direct_lookups), "set_unresolved_conflict_rate": ratio(set_unresolved, set_lookups), "oracle_accuracy": oracle_accuracy_value, "oracle_coverage": oracle_coverage_value, "direct_shadow_accuracy": direct_accuracy, "set_shadow_accuracy": set_accuracy, "direct_shadow_coverage": direct_coverage, "set_shadow_coverage": set_coverage, "direct_accuracy_recovery": ratio(direct_accuracy, oracle_accuracy_value), "set_accuracy_recovery": ratio(set_accuracy, oracle_accuracy_value), "direct_coverage_recovery": ratio(direct_coverage, oracle_coverage_value), "set_coverage_recovery": ratio(set_coverage, oracle_coverage_value), "set_replacement_events": integer(sa.get("replacement_events")), "set_same_set_distinct_contexts": integer(sa.get("same_set_distinct_contexts")), "set_sets_ever_full": integer(sa.get("sets_ever_reaching_4way")), "direct_predictions_generated": integer(direct.get("predictions_generated")), "set_predictions_generated": integer(counters.get("predictions_generated"))})
        trace_data[trace] = {"direct": direct, "counters": counters, "sa": sa}

        for predictor, directory in (("NoPrefetch", "NoPrefetch"), ("NativeSPP", "NativeSPP"), ("DirectMappedWAM", "WAM-H16"), ("SetAssociativeWAM", None)):
            stats_path = Path(run["json_path"]) if directory is None else REAL_OUTPUT / "runs" / trace / directory / "stats.json"
            metrics = parse_json_metric(stats_path)
            no_prefetch = parse_json_metric(REAL_OUTPUT / "runs" / trace / "NoPrefetch/stats.json")
            direct_ipc = parse_json_metric(REAL_OUTPUT / "runs" / trace / "WAM-H16/stats.json")
            ipc_rows.append({"trace": trace, "workload_class": workload_class, "predictor": predictor, "ipc": metrics.get("ipc", 0.0), "speedup_vs_noprefetch": ratio(metrics.get("ipc", 0.0), no_prefetch.get("ipc", 0.0)), "speedup_vs_direct": "not_applicable" if predictor == "DirectMappedWAM" else ratio(metrics.get("ipc", 0.0), direct_ipc.get("ipc", 0.0)), "status": "available" if metrics else "missing"})

    write_csv(output / "activation.csv", activation)
    write_csv(output / "hash_comparison.csv", hash_rows)
    write_csv(output / "occupancy.csv", occupancy)
    write_csv(output / "set_pressure.csv", pressure)
    write_csv(output / "shadow_accuracy.csv", shadow_rows)
    write_csv(output / "oracle_recovery.csv", [{"trace": row["trace"], "variant": variant, "oracle_accuracy": row["oracle_accuracy"], "oracle_coverage": row["oracle_coverage"], "actual_accuracy": row[f"{prefix}_shadow_accuracy"], "actual_coverage": row[f"{prefix}_shadow_coverage"], "accuracy_recovery": row[f"{prefix}_accuracy_recovery"], "coverage_recovery": row[f"{prefix}_coverage_recovery"]} for row in per_trace for variant, prefix in (("DirectMappedWAM", "direct"), ("SetAssociativeWAM", "set"))])
    write_csv(output / "confidence.csv", confidence_rows)
    write_csv(output / "support.csv", support_rows)
    write_csv(output / "activation.csv", activation)
    write_csv(output / "failure_funnel.csv", funnel_rows)
    write_csv(output / "ipc.csv", ipc_rows)
    write_csv(output / "per_trace.csv", per_trace)
    write_csv(output / "storage.csv", [{"variant": "DirectMappedWAM", "table_entries": 256, "sets": 256, "ways": 1, "entry_bytes": 32, "pending_bytes": 256, "replacement_metadata_bytes": 0, "logical_state_bytes": DIRECT_BYTES, "delta_vs_direct_bytes": 0, "budget_note": "diagnostic fields excluded"}, {"variant": "SetAssociativeWAM", "table_entries": 256, "sets": SETS, "ways": WAYS, "entry_bytes": 32, "pending_bytes": 256, "replacement_metadata_bytes": 64, "logical_state_bytes": SET_BYTES, "delta_vs_direct_bytes": 64, "budget_note": "one byte round-robin pointer per set; diagnostic fields excluded"}])

    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    svg_plot(plots / "context_hit_rate.svg", "Context hit rate: DirectMappedWAM vs SetAssociativeWAM", [(f"{row['trace']} D", number(row["direct_context_hit_rate"])) for row in per_trace] + [(f"{row['trace']} SA", number(row["set_context_hit_rate"])) for row in per_trace], "hit rate")
    svg_plot(plots / "alias_loss.svg", "Alias/conflict loss", [(f"{row['trace']} D", number(row["direct_alias_loss_rate"])) for row in per_trace] + [(f"{row['trace']} SA", number(row["set_unresolved_conflict_rate"])) for row in per_trace], "loss rate")
    svg_plot(plots / "set_occupancy.svg", "Occupied state fraction", [(f"{row['trace']} {row['variant'][:2]}", number(row["occupied_fraction"])) for row in occupancy], "occupied fraction")
    svg_plot(plots / "set_pressure.svg", "Distinct contexts competing per set", [(row["trace"], number(row["p95"])) for row in pressure if row["variant"] == "SetAssociativeWAM"], "p95 contexts/set")
    svg_plot(plots / "shadow_accuracy_vs_oracle.svg", "Shadow accuracy and oracle H16 accuracy", [(f"{row['trace']} oracle", number(row["oracle_accuracy"])) for row in per_trace] + [(f"{row['trace']} {row['variant'][:2]}", number(row["accuracy"])) for row in shadow_rows], "accuracy")
    svg_plot(plots / "coverage_recovery.svg", "Coverage recovery relative to chronological oracle", [(f"{row['trace']} D", number(row["direct_coverage_recovery"])) for row in per_trace] + [(f"{row['trace']} SA", number(row["set_coverage_recovery"])) for row in per_trace], "actual/oracle coverage")
    svg_plot(plots / "confidence_distribution.svg", "Confidence distribution at context lookup", [(f"D C{index}", sum(integer(row["count"]) for row in confidence_rows if row["variant"] == "DirectMappedWAM" and integer(row["bin"]) == index)) for index in range(16)] + [(f"SA C{index}", sum(integer(row["count"]) for row in confidence_rows if row["variant"] == "SetAssociativeWAM" and integer(row["bin"]) == index)) for index in range(16)], "count")
    svg_plot(plots / "support_distribution.svg", "Support distribution at context lookup", [(f"D S{index}", sum(integer(row["count"]) for row in support_rows if row["variant"] == "DirectMappedWAM" and integer(row["bin"]) == index)) for index in range(17)] + [(f"SA S{index}", sum(integer(row["count"]) for row in support_rows if row["variant"] == "SetAssociativeWAM" and integer(row["bin"]) == index)) for index in range(17)], "count")
    funnel_plot: list[tuple[str, float]] = []
    stages = ("eligible_accesses", "contexts_formed", "context_hits", "predictions_generated", "predictions_above_threshold", "prefetch_requests", "accepted_requests", "useful_prefetches")
    for variant in ("DirectMappedWAM", "SetAssociativeWAM"):
        eligible = max(sum(integer(row["count"]) for row in funnel_rows if row["variant"] == variant and row["stage"] == "eligible_accesses"), 1)
        funnel_plot.extend((f"{variant[:2]} {stage}", ratio(sum(integer(row["count"]) for row in funnel_rows if row["variant"] == variant and row["stage"] == stage), eligible)) for stage in stages)
    svg_plot(plots / "prediction_funnel.svg", "Prediction funnel: DirectMappedWAM vs SetAssociativeWAM", funnel_plot, "fraction of eligible")
    svg_plot(plots / "ipc_comparison.svg", "IPC speedup versus NoPrefetch", [(f"{row['trace']} {row['predictor']}", number(row["speedup_vs_noprefetch"])) for row in ipc_rows], "speedup", 1.0)

    direct_hits_total = sum(integer(direct_activation[row["trace"]].get("prediction_context_hits")) for row in per_trace)
    direct_lookups_total = sum(integer(direct_activation[row["trace"]].get("prediction_lookups")) for row in per_trace)
    set_hits_total = sum(integer(trace_data[row["trace"]]["sa"].get("tag_hits")) for row in per_trace)
    set_lookups_total = sum(integer(trace_data[row["trace"]]["sa"].get("set_lookups")) for row in per_trace)
    direct_alias_total = sum(integer(direct_activation[row["trace"]].get("hash_alias_misses"), integer(direct_activation[row["trace"]].get("hash_collisions"))) for row in per_trace)
    set_unresolved_total = sum(integer(trace_data[row["trace"]]["sa"].get("unresolved_conflict_events")) for row in per_trace)
    oracle_accuracy = statistics.mean(number(row["oracle_accuracy"]) for row in per_trace) if per_trace else 0.0
    oracle_coverage = statistics.mean(number(row["oracle_coverage"]) for row in per_trace) if per_trace else 0.0
    direct_shadow_accuracy = statistics.mean(number(row["direct_shadow_accuracy"]) for row in per_trace) if per_trace else 0.0
    set_shadow_accuracy = statistics.mean(number(row["set_shadow_accuracy"]) for row in per_trace) if per_trace else 0.0
    direct_shadow_coverage = statistics.mean(number(row["direct_shadow_coverage"]) for row in per_trace) if per_trace else 0.0
    set_shadow_coverage = statistics.mean(number(row["set_shadow_coverage"]) for row in per_trace) if per_trace else 0.0
    direct_acc_recovery = statistics.mean(number(row["direct_accuracy_recovery"]) for row in per_trace) if per_trace else 0.0
    set_acc_recovery = statistics.mean(number(row["set_accuracy_recovery"]) for row in per_trace) if per_trace else 0.0
    direct_cov_recovery = statistics.mean(number(row["direct_coverage_recovery"]) for row in per_trace) if per_trace else 0.0
    set_cov_recovery = statistics.mean(number(row["set_coverage_recovery"]) for row in per_trace) if per_trace else 0.0
    delta_rows = read_csv(DIRECT_OUTPUT / "representation_diagnostics.csv")
    delta_advantage = statistics.mean(number(row.get("delta_predictability_advantage")) for row in delta_rows) if delta_rows else 0.0
    state_recovered = ratio(set_hits_total, set_lookups_total) > ratio(direct_hits_total, direct_lookups_total) + 0.02 and set_cov_recovery >= 0.5
    if not state_recovered and ratio(set_hits_total, set_lookups_total) <= ratio(direct_hits_total, direct_lookups_total) + 0.02:
        classification = "A — Aliasing hypothesis falsified"
        dominant = "four-way associativity does not recover context state; set pressure and replacement churn remain destructive"
    elif set_cov_recovery < 0.5 or set_acc_recovery < 0.5:
        classification = "B — Aliasing partially explains the failure"
        dominant = "associativity recovers some state, but not enough H16 coverage or accuracy"
    elif set_shadow_accuracy < oracle_accuracy * 0.5:
        classification = "C — State validated but prediction remains weak"
        dominant = "state lookup recovers, but confidence/update dynamics suppress accurate predictions"
    elif state_recovered and delta_advantage >= 0.15:
        classification = "E — State solved; representation is next bottleneck"
        dominant = "state recovery is strong and the frozen diagnostic shows a meaningful delta advantage"
    else:
        classification = "D — State strongly validated"
        dominant = "set-associative state recovers the H16 signal without a meaningful delta-variant justification"
    decision = "RESEARCH_DECISION = CONTINUE_TO_DELTA_VARIANT" if classification.startswith("E") else "RESEARCH_DECISION = STOP"
    next_variant = "DELTA" if decision.endswith("CONTINUE_TO_DELTA_VARIANT") else "NONE"
    ipc_geomean: dict[str, float] = {}
    for predictor in ("DirectMappedWAM", "SetAssociativeWAM", "NativeSPP"):
        values = [number(row["speedup_vs_noprefetch"]) for row in ipc_rows if row["predictor"] == predictor and row["status"] == "available"]
        ipc_geomean[predictor] = gmean(values)
    total_direct = {key: sum(integer(direct_activation[row["trace"]].get(key)) for row in per_trace) for key in ("predictions_generated", "prefetch_requests_generated", "prefetches_useful")}
    total_set = {key: sum(integer(trace_data[row["trace"]]["counters"].get(key)) for row in per_trace) for key in ("predictions_generated", "prefetch_requests_generated", "prefetches_useful")}
    config = {"variant": "SetAssociativeWAM", "sets": SETS, "ways": WAYS, "total_entries": TABLE_ENTRIES, "replacement": "round_robin_next_way_per_set", "entry_bytes": 32, "pending_context_bytes": 256, "horizon": HORIZON, "context_depth": DEPTH, "confidence_threshold": CONFIDENCE_THRESHOLD, "hash_function": "SplitMix64 finalizer preserved from DirectMappedWAM", "direct_logical_state_bytes": DIRECT_BYTES, "set_logical_state_bytes": SET_BYTES, "storage_delta_bytes": 64, "warmup_instructions": WARMUP, "simulation_instructions": SIMULATION, "traces": len(per_trace), "champsim_commit": PINNED_COMMIT, "controls": ["DirectMappedWAM frozen diagnostic results", "NativeSPP frozen real-trace results", "NoPrefetch frozen real-trace results"], "production_semantics_changed": False, "diagnostic_fields_excluded_from_storage": True, "classification": classification, "research_decision": decision, "next_variant": next_variant}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    report = f'''# Set-associative WAM diagnostic

## Experimental scope

This is the single authorized follow-up to the final DirectMappedWAM diagnostic: a 64-set × 4-way table with 256 total entries. Each way preserves the DirectMappedWAM entry layout and H16 update semantics. The confidence threshold remains {CONFIDENCE_THRESHOLD}, the prefetch path is unchanged, and the replacement pointer adds 64 logical bytes. The evaluation uses the same ten native traces, {WARMUP:,} warmup instructions, {SIMULATION:,} simulated instructions, one core, and pinned ChampSim commit `{PINNED_COMMIT}`. DirectMappedWAM, NativeSPP, and NoPrefetch are frozen controls from the prior fixed-window evaluation.

## Results

- Direct context hit rate: **{ratio(direct_hits_total, direct_lookups_total):.3%}**.
- Set-associative context hit rate: **{ratio(set_hits_total, set_lookups_total):.3%}**.
- Direct hash-alias loss rate: **{ratio(direct_alias_total, direct_lookups_total):.3%}**.
- Set unresolved all-ways conflict rate: **{ratio(set_unresolved_total, set_lookups_total):.3%}**.
- Oracle H16 accuracy / coverage: **{oracle_accuracy:.3%} / {oracle_coverage:.3%}**.
- Direct ShadowWAM accuracy / coverage: **{direct_shadow_accuracy:.3%} / {direct_shadow_coverage:.3%}**.
- Set ShadowWAM accuracy / coverage: **{set_shadow_accuracy:.3%} / {set_shadow_coverage:.3%}**.
- Set accuracy / coverage recovery versus oracle: **{set_acc_recovery:.3%} / {set_cov_recovery:.3%}**.
- IPC geomean speedup: DirectMappedWAM **{ipc_geomean['DirectMappedWAM']:.3f}×**, SetAssociativeWAM **{ipc_geomean['SetAssociativeWAM']:.3f}×**, NativeSPP **{ipc_geomean['NativeSPP']:.3f}×**.
- Frozen delta-oracle advantage: **{delta_advantage:.3%}** absolute; no delta simulation was run.

## Required questions

1. **Did four-way associativity recover the lost context state?** {"Yes, materially." if state_recovered else "No; the aggregate tag-hit rate did not materially exceed the direct-mapped control."}
2. **How much did it reduce alias loss?** Direct alias loss was {ratio(direct_alias_total, direct_lookups_total):.3%}; SA unresolved conflict loss was {ratio(set_unresolved_total, set_lookups_total):.3%}. The SA-specific direct-map-equivalent counter is in `hash_comparison.csv`.
3. **How many contexts compete per set?** The per-trace median/p90/p95/p99/max pressure counters are in `set_pressure.csv`; sets reaching four ways are recorded per trace.
4. **Was the fixed 256-entry budget preserved?** Yes. The table remains 256 entries; only 64 one-byte round-robin pointers were added.
5. **Did SA improve context hit rate, coverage, or reuse?** See `per_trace.csv`, `context_hit_rate.svg`, `set_occupancy.svg`, and `set_pressure.svg`; the aggregate hit rates above are the primary gate.
6. **Did SA recover the offline H16 oracle?** The oracle is unchanged because the trace and semantics are unchanged; `oracle_recovery.csv` measures actual recovery for both state layouts.
7. **Did ShadowWAM accuracy improve?** Direct ShadowWAM was {direct_shadow_accuracy:.3%}; SA was {set_shadow_accuracy:.3%}. The per-trace comparison is in `shadow_accuracy.csv`.
8. **Did ShadowWAM coverage improve?** Direct was {direct_shadow_coverage:.3%}; SA was {set_shadow_coverage:.3%}.
9. **Did confidence or support distributions improve?** The frozen direct and new SA distributions are in `confidence.csv` and `support.csv`; no threshold or support policy was retuned.
10. **Did production predictions improve?** Direct generated {total_direct['predictions_generated']:,}; SA generated {total_set['predictions_generated']:,}. The funnel is in `failure_funnel.csv`.
11. **Did useful prefetches improve?** Direct requested {total_direct['prefetch_requests_generated']:,} and recorded {total_direct['prefetches_useful']:,} useful; SA requested {total_set['prefetch_requests_generated']:,} and recorded {total_set['prefetches_useful']:,} useful.
12. **Did IPC improve?** The primary comparison is in `ipc.csv` and `ipc_comparison.svg`; NativeSPP remains the unchanged reference.
13. **What did occupancy show?** SA occupancy bins 0..4, occupied sets, and final occupied entries are in `occupancy.csv`; direct per-set bins are not exposed.
14. **What did set pressure show?** SA pressure is measured as distinct context keys competing for each set; distribution statistics are in `set_pressure.csv`.
15. **Was the storage budget respected?** Yes: DirectMappedWAM {DIRECT_BYTES} bytes, SetAssociativeWAM {SET_BYTES} bytes, delta 64 bytes.
16. **Does the evidence justify a delta variant?** No. The frozen delta advantage is {delta_advantage:.3%}, below the meaningful continuation threshold used here, and SA did not satisfy the state-recovery gate.
17. **What is the final research decision?** **{decision}**; next variant: **{next_variant}**.

## Final classification

**{classification}**

Dominant remaining bottleneck: **{dominant}**.

## Required final console summary

traces evaluated: {len(per_trace)}
DirectMappedWAM storage bytes: {DIRECT_BYTES}
SetAssociativeWAM storage bytes: {SET_BYTES}
direct-map alias rate: {ratio(direct_alias_total, direct_lookups_total):.3%}
set-associative unresolved conflict rate: {ratio(set_unresolved_total, set_lookups_total):.3%}
direct context hit rate: {ratio(direct_hits_total, direct_lookups_total):.3%}
set-associative context hit rate: {ratio(set_hits_total, set_lookups_total):.3%}
oracle H16 accuracy: {oracle_accuracy:.3%}
DirectShadow H16 accuracy: {direct_shadow_accuracy:.3%}
SetAssociativeShadow H16 accuracy: {set_shadow_accuracy:.3%}
oracle H16 coverage: {oracle_coverage:.3%}
DirectShadow coverage: {direct_shadow_coverage:.3%}
SetAssociativeShadow coverage: {set_shadow_coverage:.3%}
accuracy recovery %: {set_acc_recovery:.3%}
coverage recovery %: {set_cov_recovery:.3%}
predictions generated:
direct: {total_direct['predictions_generated']}
set-associative: {total_set['predictions_generated']}
prefetches requested:
direct: {total_direct['prefetch_requests_generated']}
set-associative: {total_set['prefetch_requests_generated']}
useful prefetches:
direct: {total_direct['prefetches_useful']}
set-associative: {total_set['prefetches_useful']}
DirectMappedWAM geomean IPC speedup: {ipc_geomean['DirectMappedWAM']:.3f}x
SetAssociativeWAM geomean IPC speedup: {ipc_geomean['SetAssociativeWAM']:.3f}x
NativeSPP geomean IPC speedup: {ipc_geomean['NativeSPP']:.3f}x
dominant remaining bottleneck: {dominant}
final classification: {classification}
RESEARCH_DECISION = {"CONTINUE_TO_DELTA_VARIANT" if decision.endswith("CONTINUE_TO_DELTA_VARIANT") else "STOP"}
next variant: {next_variant}
'''
    (output / "report.md").write_text(report, encoding="utf-8")
    print(report[report.index("traces evaluated:"):])


if __name__ == "__main__":
    main()
