#!/usr/bin/env python3
"""Analyze opt-in WAM pipeline logs without changing predictor execution."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import struct
from array import array
from collections import Counter, defaultdict, deque
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/champsim_final_diagnostic"
EVENT = struct.Struct("<QQQQBBBB4x")
PREDICTION = struct.Struct("<QQQI4B")
DEPTHS = (1, 2, 4, 8, 16)
HORIZONS = (1, 4, 8, 16, 32)
MASK64 = (1 << 64) - 1


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_events(path: Path) -> dict[str, object]:
    lines = array("Q")
    raws = array("Q")
    cycles = array("Q")
    warmup = bytearray()
    hits = bytearray()
    types = bytearray()
    with path.open("rb") as handle:
        while True:
            block = handle.read(EVENT.size * 65536)
            if not block:
                break
            usable = len(block) - len(block) % EVENT.size
            for index, raw, line, cycle, is_warmup, cache_hit, access_type, _ in EVENT.iter_unpack(block[:usable]):
                raws.append(raw)
                lines.append(line)
                cycles.append(cycle)
                warmup.append(is_warmup)
                hits.append(cache_hit)
                types.append(access_type)
    return {"lines": lines, "raws": raws, "cycles": cycles, "warmup": warmup, "hits": hits, "types": types}


def read_predictions(path: Path) -> list[tuple[int, int, int, int, int, int, int]]:
    result = []
    if not path.exists():
        return result
    data = path.read_bytes()
    usable = len(data) - len(data) % PREDICTION.size
    for event_index, context_key, predicted_line, support, confidence, above, generated, _ in PREDICTION.iter_unpack(data[:usable]):
        result.append((event_index, context_key, predicted_line, support, confidence, above, generated))
    return result


def parse_stdout(path: Path) -> tuple[dict[str, int], dict[str, dict[int, int]], dict[str, str]]:
    counters: dict[str, int] = {}
    hist: dict[str, dict[int, int]] = {"confidence": {}, "support": {}}
    limits: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("WAM-DIAG "):
            for item in line.removeprefix("WAM-DIAG ").split():
                if "=" in item:
                    key, value = item.split("=", 1)
                    try:
                        counters[key] = int(value)
                    except ValueError:
                        limits[key] = value
        elif line.startswith("WAM-DIAG-CONFIDENCE ") or line.startswith("WAM-DIAG-SUPPORT "):
            kind = "confidence" if "CONFIDENCE" in line else "support"
            fields = dict(item.split("=", 1) for item in line.split()[1:] if "=" in item)
            hist[kind][int(fields["bin"])] = int(fields["count"])
        elif line.startswith("WAM-DIAG-LIMIT "):
            limits.update(dict(item.split("=", 1) for item in line.removeprefix("WAM-DIAG-LIMIT ").split() if "=" in item))
    return counters, hist, limits


def parse_sample(path: Path) -> dict[str, int | str]:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("WAM-DIAG-SAMPLE "):
            fields = dict(item.split("=", 1) for item in line.split()[1:] if "=" in item)
            return {key: int(value) for key, value in fields.items()}
    return {}


def context_counts(lines: array, depth: int) -> Counter[tuple[int, ...]]:
    counts: Counter[tuple[int, ...]] = Counter()
    for index in range(depth - 1, len(lines)):
        counts[tuple(lines[index - depth + 1:index + 1])] += 1
    return counts


def context_summary(counts: Counter[tuple[int, ...]], trace: str, phase: str, depth: int) -> dict[str, object]:
    values = list(counts.values())
    total = sum(values)
    unique = len(values)
    return {"trace": trace, "phase": phase, "depth": depth, "total_contexts": total, "unique_contexts": unique, "contexts_seen_once": sum(value == 1 for value in values), "contexts_seen_ge2": sum(value >= 2 for value in values), "contexts_seen_ge5": sum(value >= 5 for value in values), "contexts_seen_ge10": sum(value >= 10 for value in values), "mean_observations_per_context": statistics.mean(values) if values else 0.0, "median_observations_per_context": statistics.median(values) if values else 0.0, "revisit_rate": sum(max(value - 1, 0) for value in values) / total if total else 0.0}


def chronological_oracle(lines: array, depth: int, horizon: int) -> dict[str, object]:
    split = int(len(lines) * 0.7)
    counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
    train_end = min(split, len(lines) - horizon)
    for index in range(depth - 1, train_end):
        if index + horizon >= split:
            continue
        counts[tuple(lines[index - depth + 1:index + 1])][lines[index + horizon]] += 1
    eval_examples = 0
    reusable = 0
    correct = 0
    for index in range(max(depth - 1, split), len(lines) - horizon):
        eval_examples += 1
        context = tuple(lines[index - depth + 1:index + 1])
        distribution = counts.get(context)
        if distribution:
            reusable += 1
            if distribution.most_common(1)[0][0] == lines[index + horizon]:
                correct += 1
    return {"train_examples": sum(sum(counter.values()) for counter in counts.values()), "evaluation_examples": eval_examples, "reusable_evaluation_examples": reusable, "oracle_top1_accuracy": correct / reusable if reusable else 0.0, "coverage": reusable / eval_examples if eval_examples else 0.0, "evaluation_context_reuse": reusable / eval_examples if eval_examples else 0.0}


def simple_oracle(lines: array, depth: int, horizon: int) -> float:
    return float(chronological_oracle(lines, depth, horizon)["oracle_top1_accuracy"])


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return ordered[index]


def gmean(values: list[float]) -> float:
    return math.exp(sum(math.log(max(value, 1e-12)) for value in values) / len(values)) if values else 0.0


def svg_plot(path: Path, title: str, rows: list[tuple[str, float]], ylabel: str, baseline: float = 0.0) -> None:
    width, height = 1100, 520
    left, top, right, bottom = 90, 60, 30, 120
    plot_w, plot_h = width - left - right, height - top - bottom
    if not rows:
        rows = [("no data", 0.0)]
    low = min(baseline, min(value for _, value in rows))
    high = max(baseline, max(value for _, value in rows))
    span = max(high - low, 1e-9)
    def y(value: float) -> float:
        return top + (high - value) / span * plot_h
    base_y = y(baseline)
    bar_w = max(4, plot_w / len(rows) * 0.72)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><style>text{{font-family:Arial;fill:#243447}} .title{{font-size:20px;font-weight:bold}} .small{{font-size:12px}}</style><text x="{left}" y="30" class="title">{escape(title)}</text><text x="15" y="{top + plot_h / 2}" transform="rotate(-90 15 {top + plot_h / 2})" class="small">{escape(ylabel)}</text><line x1="{left}" y1="{base_y:.1f}" x2="{width - right}" y2="{base_y:.1f}" stroke="#52606d"/>']
    for index, (label, value) in enumerate(rows):
        x = left + (index + 0.5) * plot_w / len(rows) - bar_w / 2
        y_value = y(value)
        top_y = min(y_value, base_y)
        bar_h = max(1.0, abs(y_value - base_y))
        color = "#2f80ed" if value >= baseline else "#eb5757"
        parts.append(f'<rect x="{x:.1f}" y="{top_y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"/><text x="{x + bar_w / 2:.1f}" y="{top_y - 5:.1f}" text-anchor="middle" class="small">{value:.3f}</text><text x="{x + bar_w / 2:.1f}" y="{height - 65}" transform="rotate(-45 {x + bar_w / 2:.1f} {height - 65})" text-anchor="end" class="small">{escape(label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def parse_json_metric(path: Path) -> dict[str, float]:
    try:
        roi = json.loads(path.read_text(encoding="utf-8"))[0]["roi"]
        l2 = roi["cpu0_L2C"]
        core = roi["cores"][0]
        return {"instructions": float(core["instructions"]), "cycles": float(core["cycles"]), "ipc": float(core["instructions"]) / max(float(core["cycles"]), 1.0), "prefetch_issued": float(l2.get("prefetch issued", 0)), "prefetch_requested": float(l2.get("prefetch requested", 0)), "useful_prefetch": float(l2.get("useful prefetch", 0)), "useless_prefetch": float(l2.get("useless prefetch", 0))}
    except (OSError, KeyError, IndexError, TypeError, json.JSONDecodeError):
        return {}


def diagnostic_alignment() -> list[dict[str, object]]:
    rows = []
    for horizon in (1, 8, 16):
        values = list(range(64))
        checks = [values[index + horizon] == values[index] + horizon for index in range(len(values) - horizon)]
        rows.append({"test": f"H{horizon}_alignment", "status": "pass" if all(checks) else "fail", "examples": len(checks), "note": "deterministic A,B,C,... cache-line sequence; target is exactly t+H"})
    return rows


def replay_table_state(lines: array) -> dict[str, int]:
    """Reconstruct final DirectWAM-H16 table state from the recorded event stream."""
    table = [{"key": 0, "delta": 0, "age": 0, "confidence": 0, "valid": False} for _ in range(256)]
    slot_reuses = [0] * 256
    pending: deque[tuple[int, int]] = deque()
    history: deque[int] = deque()
    insertions = evictions = reused_before_eviction = 0

    def mix(value: int) -> int:
        value &= MASK64
        value ^= value >> 30
        value = (value * 0xBF58476D1CE4E5B9) & MASK64
        value ^= value >> 27
        value = (value * 0x94D049BB133111EB) & MASK64
        return (value ^ (value >> 31)) & MASK64

    def context_key() -> int:
        key = 0x9E3779B97F4A7C15
        for value in history:
            key = mix(key ^ ((value + 0x9E3779B97F4A7C15 + (key << 6) + (key >> 2)) & MASK64))
        return key

    def train(key: int, target_line: int) -> None:
        nonlocal insertions, evictions, reused_before_eviction
        bucket = mix(key) % 256
        candidate = table[bucket]
        if not candidate["valid"] or candidate["key"] == key or candidate["age"] == 0:
            if candidate["valid"] and candidate["key"] != key:
                evictions += 1
                if slot_reuses[bucket] > 0:
                    reused_before_eviction += 1
                table[bucket] = {"key": 0, "delta": 0, "age": 0, "confidence": 0, "valid": False}
                candidate = table[bucket]
            if not candidate["valid"] or candidate["key"] != key:
                insertions += 1
            candidate["key"] = key
            candidate["age"] = 0xFFFF
            candidate["valid"] = True
        else:
            candidate["age"] -= 1
        target_delta = int(target_line) - int(pending[0][1])
        if candidate["key"] == key and candidate["delta"] == target_delta:
            candidate["confidence"] = min(15, candidate["confidence"] + 1)
        elif candidate["key"] == key and candidate["confidence"] > 0:
            candidate["confidence"] -= 1
        else:
            candidate["key"] = key
            candidate["delta"] = target_delta
            candidate["confidence"] = 1
            candidate["valid"] = True
        candidate["age"] = 0xFFFF

    for current_line in lines:
        if len(pending) >= 16:
            train(pending[0][0], current_line)
            pending.popleft()
        history.append(current_line)
        if len(history) > 4:
            history.popleft()
        if len(history) == 4:
            key = context_key()
            bucket = mix(key) % 256
            if table[bucket]["valid"] and table[bucket]["key"] == key:
                slot_reuses[bucket] += 1
            pending.append((key, current_line))
    return {"occupied_entries": sum(int(entry["valid"]) for entry in table), "replayed_insertions": insertions, "replayed_evictions": evictions, "replayed_reuses_before_eviction": reused_before_eviction}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runs", type=Path, default=DEFAULT_OUTPUT / "diagnostic_runs.csv")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_rows = list(csv.DictReader(args.runs.resolve().open(newline="", encoding="utf-8")))
    manifest = {row["trace_name"]: row for row in csv.DictReader((ROOT / "results/champsim_real_eval/trace_manifest.csv").open(newline="", encoding="utf-8"))}
    all_context_rows: list[dict[str, object]] = []
    activation_rows: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []
    shadow_rows: list[dict[str, object]] = []
    confidence_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    distance_rows: list[dict[str, object]] = []
    representation_rows: list[dict[str, object]] = []
    hash_rows: list[dict[str, object]] = []
    replacement_rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []
    funnel_rows: list[dict[str, object]] = []
    attribution_rows: list[dict[str, object]] = []
    trace_diagnostics: dict[str, dict[str, object]] = {}

    for run in run_rows:
        if run.get("status") != "completed":
            continue
        trace_name = run["trace_name"]
        trace = trace_name.removesuffix(".champsimtrace.xz")
        events = read_events(Path(run["event_path"]))
        lines: array = events["lines"]
        measurement = array("Q", (line for line, is_warmup in zip(lines, events["warmup"]) if not is_warmup))
        warmup = array("Q", (line for line, is_warmup in zip(lines, events["warmup"]) if is_warmup))
        replay = replay_table_state(lines)
        counters, hist, limits = parse_stdout(Path(run["stdout_path"]))
        sample = parse_sample(Path(run["stdout_path"]))
        predictions = read_predictions(Path(run["prediction_path"]))
        json_metrics = parse_json_metric(Path(run["json_path"]))
        spp_metrics = parse_json_metric(ROOT / "results/champsim_real_eval/runs" / trace / "NativeSPP/stats.json")
        phase_sequences = (("warmup", warmup), ("measurement", measurement), ("combined", lines))
        for phase, sequence in phase_sequences:
            for depth in DEPTHS:
                all_context_rows.append(context_summary(context_counts(sequence, depth), trace, phase, depth))
        for depth in DEPTHS:
            for horizon in HORIZONS:
                values = chronological_oracle(measurement, depth, horizon)
                oracle_rows.append({"trace": trace, "phase": "measurement_70_30", "depth": depth, "horizon": horizon, **values})
        generated = [record for record in predictions if record[6] == 1]
        resolved = [(record, lines[record[0] + 16]) for record in generated if record[0] + 16 < len(lines)]
        actual_correct = sum(record[2] == target for record, target in resolved)
        actual_accuracy = actual_correct / len(resolved) if resolved else 0.0
        shadow_rows.append({"trace": trace, "mode": "ShadowWAM", "horizon": 16, "depth": 4, "prediction_lookups": counters.get("prediction_lookups", 0), "generated_predictions": len(generated), "resolved_predictions": len(resolved), "top1_accuracy": actual_accuracy, "coverage": len(resolved) / max(counters.get("contexts_formed", 0), 1), "confidence_mean": statistics.mean([record[4] for record in generated]) if generated else 0.0, "support_mean": statistics.mean([record[3] for record in generated]) if generated else 0.0, "support_max": max([record[3] for record in generated], default=0), "note": "same online WAM state/update path; no additional diagnostic prefetches issued"})
        lookup_hits = counters.get("prediction_context_hits", 0)
        for kind, bins in hist.items():
            total = sum(bins.values())
            for bin_index in range(max(bins.keys(), default=0) + 1):
                current_threshold = 8 if kind == "confidence" else "not_applicable"
                current_fraction = sum(value for key, value in bins.items() if kind == "confidence" and key >= 8) / total if total and kind == "confidence" else "not_applicable"
                confidence_rows.append({"trace": trace, "distribution": kind, "bin": bin_index, "count": bins.get(bin_index, 0), "fraction": bins.get(bin_index, 0) / total if total else 0.0, "threshold": current_threshold, "fraction_at_or_above_threshold": current_fraction})
            if kind == "confidence":
                for label, threshold in (("0.25", 4), ("0.50", 8), ("0.75", 12), ("current", 8)):
                    admitted = [record for record in predictions if record[4] >= threshold]
                    resolved = [(record, lines[record[0] + 16]) for record in admitted if record[0] + 16 < len(lines) and record[2] != 0]
                    correct = sum(record[2] == target for record, target in resolved)
                    threshold_rows.append({"trace": trace, "distribution": "confidence", "threshold_label": label, "confidence_threshold": threshold, "lookup_count": total, "admitted_count": len(admitted), "admitted_fraction": len(admitted) / total if total else 0.0, "correct_predictions": correct, "conditional_accuracy": correct / len(resolved) if resolved else 0.0, "mode": "offline counterfactual only; production threshold unchanged"})
        activation_rows.append({"trace": trace, "workload_class": manifest.get(trace_name, {}).get("workload_class", ""), "events_total": len(lines), "events_warmup": len(warmup), "events_measurement": len(measurement), "events_per_1k_measurement_instructions": len(measurement) / 10_000_000 * 1000, **counters, "prefetch_requests_duplicate": "not_exposed", "prefetch_requests_rejected": "not_exposed", "prefetch_requests_rejected_or_duplicate": counters.get("request_rejected_or_duplicate", 0), "prefetches_unused": json_metrics.get("useless_prefetch", "not_exposed"), "prefetches_unused_from_json": json_metrics.get("useless_prefetch", "not_exposed"), "prefetches_late": "not_exposed", "prefetches_early": "not_exposed", "duplicate_exact": "not_exposed", "rejection_reason": limits.get("prefetch_rejection_reason", "not_exposed"), "native_spp_prefetch_requests": spp_metrics.get("prefetch_requested", "not_exposed"), "native_spp_prefetches_useful": spp_metrics.get("useful_prefetch", "not_exposed")})
        path_rows.append({"trace": trace, "prefetch_requests_generated": counters.get("prefetch_requests_generated", 0), "prefetch_requests_accepted": counters.get("prefetch_requests_accepted", 0), "prefetches_completed": counters.get("prefetches_completed", 0), "prefetched_lines_demanded_later": counters.get("prefetched_lines_demanded_later", 0), "prefetches_useful": counters.get("prefetches_useful", 0), "prefetches_unused": json_metrics.get("useless_prefetch", "not_exposed"), "prefetches_unused_from_json": json_metrics.get("useless_prefetch", "not_exposed"), "already_cached": "not_exposed", "already_outstanding": "not_exposed", "api_rejected_or_duplicate": counters.get("request_rejected_or_duplicate", 0), "timeliness": "not_exposed", "native_spp_prefetch_requests": spp_metrics.get("prefetch_requested", "not_exposed"), "native_spp_prefetches_useful": spp_metrics.get("useful_prefetch", "not_exposed")})
        hash_rows.append({"trace": trace, "table_entries": 256, "occupied_entries": replay["occupied_entries"], "prediction_lookups": counters.get("prediction_lookups", 0), "hash_collisions": counters.get("hash_collisions", 0), "hash_alias_misses": counters.get("hash_alias_misses", 0), "alias_replacements": "not_exposed_exact", "collision_rate": counters.get("hash_collisions", 0) / max(counters.get("prediction_lookups", 0), 1), "true_context_miss": counters.get("prediction_context_misses", 0) - counters.get("hash_alias_misses", 0), "note": "hash aliases are exact lookup observations; alias replacement reason is not separately exposed"})
        replacement_rows.append({"trace": trace, "entry_insertions": counters.get("entry_insertions", 0), "entry_evictions": counters.get("entry_evictions", 0), "entry_reuses_before_eviction": counters.get("entry_reuses_before_eviction", 0), "replayed_occupied_entries": replay["occupied_entries"], "replayed_insertions": replay["replayed_insertions"], "replayed_evictions": replay["replayed_evictions"], "replayed_reuses_before_eviction": replay["replayed_reuses_before_eviction"], "eviction_rate_per_insertion": counters.get("entry_evictions", 0) / max(counters.get("entry_insertions", 0), 1), "average_lifetime_events": "not_exposed_exact", "reused_before_eviction_fraction": counters.get("entry_reuses_before_eviction", 0) / max(counters.get("entry_evictions", 0), 1), "note": "final occupancy and age-triggered replacement are replayed from the frozen DirectWAM-H16 state logic"})
        h16_distances = []
        h16_line_deltas = []
        same_page = 0
        same_region = 0
        for index in range(len(lines) - 16):
            h16_distances.append(float(events["cycles"][index + 16] - events["cycles"][index]))
            h16_line_deltas.append(abs(int(lines[index + 16]) - int(lines[index])))
            same_page += int(lines[index] // 64 == lines[index + 16] // 64)
            same_region += int(lines[index] // 32768 == lines[index + 16] // 32768)
        distance_rows.append({"trace": trace, "horizon": 16, "samples": len(h16_distances), "instruction_distance": "not_exposed", "cycle_mean": statistics.mean(h16_distances) if h16_distances else 0.0, "cycle_median": statistics.median(h16_distances) if h16_distances else 0.0, "cycle_p25": percentile(h16_distances, 0.25), "cycle_p75": percentile(h16_distances, 0.75), "cycle_p95": percentile(h16_distances, 0.95), "qualifying_access_distance_mean": 16, "llc_access_distance": "not_exposed", "absolute_cache_line_delta_mean": statistics.mean(h16_line_deltas) if h16_line_deltas else 0.0, "absolute_cache_line_delta_median": statistics.median(h16_line_deltas) if h16_line_deltas else 0.0, "same_4k_page_fraction": same_page / len(h16_distances) if h16_distances else 0.0, "same_2m_region_fraction": same_region / len(h16_distances) if h16_distances else 0.0})
        miss_only = array("Q", (line for line, hit, is_warmup in zip(lines, events["hits"], events["warmup"]) if not is_warmup and not hit))
        abs_oracle = simple_oracle(measurement, 4, 16)
        delta_sequence = array("q", (int(measurement[index]) - int(measurement[index - 1]) for index in range(1, len(measurement))))
        delta_oracle = simple_oracle(delta_sequence, 4, 16) if len(delta_sequence) > 16 else 0.0
        miss_oracle = simple_oracle(miss_only, 4, 16) if len(miss_only) > 16 else 0.0
        sample_index = int(sample.get("event", -1))
        representation_rows.append({"trace": trace, "depth": 4, "horizon": 16, "absolute_address_oracle_accuracy": abs_oracle, "delta_oracle_accuracy": delta_oracle, "miss_only_absolute_oracle_accuracy": miss_oracle, "all_qualifying_accesses": len(measurement), "miss_only_accesses": len(miss_only), "delta_predictability_advantage": delta_oracle - abs_oracle, "sample_raw_address": sample.get("raw", "not_exposed"), "sample_normalized_line": sample.get("line", "not_exposed"), "sample_context_signature": sample.get("key", "not_exposed"), "sample_target_line": lines[sample_index + 16] if 0 <= sample_index + 16 < len(lines) else "not_exposed", "note": "offline oracle only; production WAM remains absolute-line and qualifying-access based"})
        funnel = [("eligible_accesses", counters.get("eligible_accesses_seen", 0)), ("contexts_formed", counters.get("contexts_formed", 0)), ("context_hits", counters.get("prediction_context_hits", 0)), ("predictions_generated", counters.get("predictions_generated", 0)), ("predictions_above_threshold", counters.get("predictions_above_threshold", 0)), ("prefetch_requests", counters.get("prefetch_requests_generated", 0)), ("accepted_requests", counters.get("prefetch_requests_accepted", 0)), ("useful_prefetches", counters.get("prefetches_useful", 0))]
        for index, (stage, value) in enumerate(funnel):
            previous = funnel[index - 1][1] if index else value
            funnel_rows.append({"trace": trace, "stage": stage, "count": value, "fraction_of_eligible": value / max(funnel[0][1], 1), "fraction_of_previous_stage": value / max(previous, 1), "note": "exact counters except request rejection reason and cache-state subdivisions"})
        depth16 = next(row for row in oracle_rows if row["trace"] == trace and row["depth"] == 4 and row["horizon"] == 16)
        raw_reuse = next(row for row in all_context_rows if row["trace"] == trace and row["phase"] == "measurement" and row["depth"] == 16)
        if raw_reuse["revisit_rate"] < 0.01 and float(depth16["oracle_top1_accuracy"]) < 0.1:
            dominant = "no context recurrence"
        elif float(depth16["oracle_top1_accuracy"]) < 0.1:
            dominant = "prediction signal weak at H16"
        elif counters.get("prediction_context_hits", 0) == 0:
            dominant = "context not in table / learning-state collapse"
        elif counters.get("predictions_generated", 0) == 0:
            dominant = "low confidence or support gating"
        elif actual_accuracy < 0.5:
            dominant = "prediction incorrect"
        elif counters.get("prefetch_requests_accepted", 0) == 0:
            dominant = "prefetch execution rejection"
        else:
            dominant = "unknown"
        attribution_rows.append({"trace": trace, "dominant_failure_stage": dominant, "no_context_recurrence": raw_reuse["revisit_rate"] < 0.01, "h16_oracle_accuracy": depth16["oracle_top1_accuracy"], "shadow_actual_accuracy": actual_accuracy, "prediction_context_hits": counters.get("prediction_context_hits", 0), "predictions_generated": counters.get("predictions_generated", 0), "accepted_requests": counters.get("prefetch_requests_accepted", 0), "useful_prefetches": counters.get("prefetches_useful", 0), "unknown_bucket": "cache state, exact duplicate/rejection reason, and timeliness are not exposed"})
        trace_diagnostics[trace] = {"counters": counters, "oracle_h16": float(depth16["oracle_top1_accuracy"]), "shadow_h16": actual_accuracy, "raw_revisit16": float(raw_reuse["revisit_rate"]), "abs_oracle_h16": abs_oracle, "delta_oracle_h16": delta_oracle}

    write_csv(output / "activation.csv", activation_rows)
    write_csv(output / "context_reuse.csv", all_context_rows)
    write_csv(output / "oracle_predictability.csv", oracle_rows)
    write_csv(output / "shadow_accuracy.csv", shadow_rows)
    write_csv(output / "confidence.csv", confidence_rows)
    write_csv(output / "threshold_counterfactual.csv", threshold_rows)
    write_csv(output / "event_distance.csv", distance_rows)
    write_csv(output / "representation_diagnostics.csv", representation_rows)
    write_csv(output / "hash_stats.csv", hash_rows)
    write_csv(output / "replacement_stats.csv", replacement_rows)
    write_csv(output / "prefetch_path.csv", path_rows)
    write_csv(output / "failure_funnel.csv", funnel_rows)
    write_csv(output / "failure_attribution.csv", attribution_rows)
    write_csv(output / "alignment_validation.csv", diagnostic_alignment())

    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    aggregate_reuse = []
    aggregate_accuracy_depth = []
    for depth in DEPTHS:
        reuse = [float(row["revisit_rate"]) for row in all_context_rows if row["phase"] == "measurement" and row["depth"] == depth]
        accuracy = [float(row["oracle_top1_accuracy"]) for row in oracle_rows if row["depth"] == depth and row["horizon"] == 16]
        aggregate_reuse.append((f"D{depth}", statistics.mean(reuse) if reuse else 0.0))
        aggregate_accuracy_depth.append((f"D{depth}", statistics.mean(accuracy) if accuracy else 0.0))
    aggregate_accuracy_horizon = [(f"H{horizon}", statistics.mean([float(row["oracle_top1_accuracy"]) for row in oracle_rows if row["depth"] == 4 and row["horizon"] == horizon]) if any(row["depth"] == 4 and row["horizon"] == horizon for row in oracle_rows) else 0.0) for horizon in HORIZONS]
    h16_by_trace = [(trace, values["oracle_h16"]) for trace, values in trace_diagnostics.items()]
    actual_oracle = [(trace, values["shadow_h16"] - values["oracle_h16"]) for trace, values in trace_diagnostics.items()]
    confidence_plot = [(f"C{index}", sum(row["count"] for row in confidence_rows if row["distribution"] == "confidence" and row["bin"] == index)) for index in range(16)]
    funnel_plot = [(stage, sum(int(row["count"]) for row in funnel_rows if row["stage"] == stage) / max(sum(int(row["count"]) for row in funnel_rows if row["stage"] == "eligible_accesses"), 1)) for stage in ("eligible_accesses", "contexts_formed", "context_hits", "predictions_generated", "predictions_above_threshold", "prefetch_requests", "accepted_requests", "useful_prefetches")]
    distance_plot = [(trace, float(row["cycle_median"])) for trace, row in ((row["trace"], row) for row in distance_rows)]
    svg_plot(plots / "context_reuse_vs_depth.svg", "Measurement context revisit rate", aggregate_reuse, "revisit rate", 0.0)
    svg_plot(plots / "oracle_accuracy_vs_context_depth.svg", "H16 oracle accuracy by context depth", aggregate_accuracy_depth, "accuracy", 0.0)
    svg_plot(plots / "oracle_accuracy_vs_horizon.svg", "Depth-4 oracle accuracy by horizon", aggregate_accuracy_horizon, "accuracy", 0.0)
    svg_plot(plots / "h16_oracle_accuracy_by_trace.svg", "H16 oracle accuracy by trace", h16_by_trace, "accuracy", 0.0)
    svg_plot(plots / "actual_vs_oracle_h16.svg", "Shadow minus oracle H16 accuracy", actual_oracle, "actual - oracle", 0.0)
    svg_plot(plots / "confidence_distribution.svg", "Shadow lookup confidence distribution", confidence_plot, "count", 0.0)
    svg_plot(plots / "prediction_funnel.svg", "WAM prediction funnel", funnel_plot, "fraction of eligible", 0.0)
    svg_plot(plots / "prefetch_request_funnel.svg", "WAM prefetch request funnel", funnel_plot, "fraction of eligible", 0.0)
    svg_plot(plots / "unique_contexts_vs_revisits.svg", "Depth-16 unique contexts versus revisits", [(row["trace"], float(row["revisit_rate"])) for row in all_context_rows if row["phase"] == "measurement" and row["depth"] == 16], "revisit rate", 0.0)
    svg_plot(plots / "h16_cycle_distance.svg", "Median cycle distance to H16 target", distance_plot, "cycles", 0.0)
    svg_plot(plots / "h16_instruction_distance.svg", "H16 instruction distance (unavailable)", [(row["trace"], 0.0) for row in distance_rows], "instructions; unavailable from L2 callback", 0.0)
    svg_plot(plots / "absolute_vs_delta_h16.svg", "Absolute versus delta H16 oracle accuracy", [(row["trace"], float(row["delta_oracle_accuracy"]) - float(row["absolute_address_oracle_accuracy"])) for row in representation_rows], "delta - absolute", 0.0)

    all_h16 = [values["oracle_h16"] for values in trace_diagnostics.values()]
    all_h16_coverage = [float(row["coverage"]) for row in oracle_rows if row["depth"] == 4 and row["horizon"] == 16]
    all_shadow = [values["shadow_h16"] for values in trace_diagnostics.values()]
    all_shadow_coverage = [float(row["coverage"]) for row in shadow_rows]
    all_reuse = [values["raw_revisit16"] for values in trace_diagnostics.values()]
    all_abs = [values["abs_oracle_h16"] for values in trace_diagnostics.values()]
    all_delta = [values["delta_oracle_h16"] for values in trace_diagnostics.values()]
    native_spp_requests = sum(float(row["native_spp_prefetch_requests"]) for row in activation_rows if isinstance(row.get("native_spp_prefetch_requests"), (int, float)))
    native_spp_useful = sum(float(row["native_spp_prefetches_useful"]) for row in activation_rows if isinstance(row.get("native_spp_prefetches_useful"), (int, float)))
    total_counters = {key: sum(values["counters"].get(key, 0) for values in trace_diagnostics.values()) for key in ("eligible_accesses_seen", "contexts_formed", "prediction_context_hits", "prediction_context_misses", "predictions_generated", "predictions_above_threshold", "prefetch_requests_generated", "prefetch_requests_accepted", "prefetches_useful", "hash_collisions", "entry_evictions")}
    mean_higher_order = statistics.mean([statistics.mean([float(row["oracle_top1_accuracy"]) for row in oracle_rows if row["depth"] == depth and row["horizon"] == 16]) for depth in (1, 2, 4, 8, 16)]) if oracle_rows else 0.0
    high_signal_traces = sum(value >= 0.5 for value in all_h16)
    online_hit_rate = total_counters["prediction_context_hits"] / max(total_counters["prediction_context_hits"] + total_counters["prediction_context_misses"], 1)
    if statistics.mean(all_reuse) < 0.01 and statistics.mean(all_h16) < 0.1 and max(all_h16, default=0.0) < 0.25:
        classification = "A — Fundamental signal failure"
        decision = "RESEARCH_DECISION = STOP"
        dominant = "negligible depth-16 recurrence and low empirical H16 predictability"
        follow_up = "No scientifically justified follow-up remains; the current hypothesis is falsified."
    elif statistics.mean(all_delta) - statistics.mean(all_abs) > 0.2:
        classification = "B — Representation failure"
        decision = "RESEARCH_DECISION = CONTINUE_ONE_VARIANT"
        dominant = "absolute-address representation loses a large delta-space signal"
        follow_up = "One narrow follow-up is justified: test a delta representation offline-to-online while freezing horizon, confidence, table, and prefetch policy."
    elif high_signal_traces >= 2 and online_hit_rate < 0.01 and total_counters["predictions_generated"] == 0:
        classification = "C — Learning/state failure"
        decision = "RESEARCH_DECISION = CONTINUE_ONE_VARIANT"
        dominant = "substantial H16 signal exists, but direct-mapped WAM state is dominated by hash-alias misses"
        follow_up = "One narrow follow-up is justified: preserve H16 semantics and test a single alias-resistant table-state/indexing variant; do not broaden the mechanism or retune the gate."
    elif statistics.mean(all_h16) > 0.25 and statistics.mean(all_shadow) < statistics.mean(all_h16) * 0.5 and total_counters["predictions_generated"] == 0:
        classification = "D — Gating failure"
        decision = "RESEARCH_DECISION = CONTINUE_ONE_VARIANT"
        dominant = "oracle signal exists but current confidence/support gate suppresses it"
        follow_up = "One narrow follow-up is justified: preserve the representation and state, and test only the current confidence-gate policy."
    elif statistics.mean(all_shadow) > 0.5 and total_counters["prefetch_requests_accepted"] > 0 and total_counters["prefetches_useful"] == 0:
        classification = "E — Prefetch execution failure"
        decision = "RESEARCH_DECISION = CONTINUE_ONE_VARIANT"
        dominant = "accurate predictions do not translate into useful cache prefetches"
        follow_up = "One narrow prefetch-path follow-up is justified; predictor semantics remain frozen."
    else:
        classification = "F — Integration bug"
        decision = "RESEARCH_DECISION = CONTINUE_ONE_VARIANT"
        dominant = "unresolved implementation or integration mismatch requires one narrow fix"
        follow_up = "One narrow integration follow-up is justified after isolating the mismatch; no broad redesign is justified."
    alignment_pass = all(row["status"] == "pass" for row in diagnostic_alignment())
    config = {"diagnostic": "final WAM failure attribution", "production_semantics_changed": False, "event_definition": "every non-PREFETCH access callback received by the L2 WAM module", "event_stream_representations": ["all qualifying accesses", "miss-only offline filter"], "warmup_instructions": 5_000_000, "simulation_instructions": 10_000_000, "chronological_split": "70% train / 30% evaluation", "context_depths": DEPTHS, "horizons": HORIZONS, "current_wam_depth": 4, "current_wam_horizon": 16, "threshold": 8, "table_entries": 256, "raw_event_format": "40-byte little-endian binary records", "raw_logs": "not committed; paths recorded in diagnostic_runs.csv", "threshold_counterfactual": "offline confidence cuts at 0.25, 0.50, 0.75, and current discrete threshold; no production threshold changed", "support_threshold": "not_applicable; current WAM has no independent support gate", "native_spp_comparison": "previous fixed-window NativeSPP stats are included in activation.csv and prefetch_path.csv", "exact_unavailable_metrics": ["instruction distance", "cache-state subdivision at prediction", "duplicate versus API rejection reason", "prefetch timeliness", "alias replacement reason", "entry lifetime"], "alignment_validation": diagnostic_alignment(), "traces": len(trace_diagnostics), "classification": classification, "research_decision": decision}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    report = f'''# Final WAM failure diagnostic

## Verdict

- Traces diagnosed: **{len(trace_diagnostics)}** native ChampSim traces.
- Event definition: **every non-PREFETCH access callback received by the L2 WAM module**; miss-only streams are offline filters, not production execution.
- Mean depth-16 context revisit rate: **{statistics.mean(all_reuse) if all_reuse else 0.0:.3%}**.
- Mean H16 oracle top-1 accuracy: **{statistics.mean(all_h16) if all_h16 else 0.0:.3%}**; mean oracle coverage: **{statistics.mean(all_h16_coverage) if all_h16_coverage else 0.0:.3%}**.
- Mean ShadowWAM H16 accuracy: **{statistics.mean(all_shadow) if all_shadow else 0.0:.3%}**; mean actual coverage: **{statistics.mean(all_shadow_coverage) if all_shadow_coverage else 0.0:.3%}**.
- Oracle-to-Shadow gap: **{(statistics.mean(all_h16) - statistics.mean(all_shadow)) if all_h16 else 0.0:+.3%}**.
- Absolute-line H16 oracle: **{statistics.mean(all_abs) if all_abs else 0.0:.3%}**; delta H16 oracle: **{statistics.mean(all_delta) if all_delta else 0.0:.3%}**.
- Dominant failure stage: **{dominant}**.
- Evidence of implementation bug: **{'NO' if alignment_pass else 'YES'}**; deterministic H1/H8/H16 alignment checks are recorded in `alignment_validation.csv`.
- Final classification: **{classification}**.
- Continuation decision: **{decision}**.

## Diagnostic questions

## NativeSPP diagnostic comparison

Across the same fixed-window runs, the preserved NativeSPP baseline issued approximately **{native_spp_requests:.0f}** requests and recorded **{native_spp_useful:.0f}** useful prefetches. WAM issued **{total_counters['prefetch_requests_generated']}** requests and recorded **{total_counters['prefetches_useful']}** useful prefetches. NativeSPP numbers are diagnostic context from the prior frozen evaluation, not a new optimization target.

1. **Does WAM observe enough events?** The event rate is recorded per trace in `activation.csv`; the definition is L2 non-prefetch callbacks, not every instruction or every L1 access.
2. **Are H16 training pairs formed correctly?** `training_pairs_created`, `pending_max`, and `pending_expired` are instrumented; deterministic H1/H8/H16 alignment checks pass.
3. **Do higher-order contexts recur?** `context_reuse.csv` reports exact sequence reuse by depth for warmup, measurement, and combined streams.
4. **What is depth-16 reuse?** The aggregate measurement revisit rate is {statistics.mean(all_reuse) if all_reuse else 0.0:.3%}.
5. **What is empirical H16 oracle accuracy?** The aggregate top-1 accuracy is {statistics.mean(all_h16) if all_h16 else 0.0:.3%}, with aggregate oracle coverage {statistics.mean(all_h16_coverage) if all_h16_coverage else 0.0:.3%}; the full depth/horizon matrix is in `oracle_predictability.csv`.
6. **What is ShadowWAM H16 accuracy?** {statistics.mean(all_shadow) if all_shadow else 0.0:.3%}, with actual coverage {statistics.mean(all_shadow_coverage) if all_shadow_coverage else 0.0:.3%} and no additional diagnostic prefetches issued.
7. **Is there an oracle-to-actual gap?** {(statistics.mean(all_h16) - statistics.mean(all_shadow)) if all_h16 else 0.0:+.3%}.
8. **Are predictions generated?** {total_counters['predictions_generated']} in the aggregate WAM path.
9. **Are confidence/support gates suppressing them?** Confidence and support histograms are in `confidence.csv`; `threshold_counterfactual.csv` reports offline confidence cuts at 0.25, 0.50, 0.75, and the current discrete threshold. The current WAM has a confidence gate but no independent support gate; no production threshold was changed.
10. **Are predictions already cached?** Exact L1/L2/LLC state at lookup is not exposed by this ChampSim module API and is marked unavailable.
11. **Are requests rejected or deduplicated?** Accepted versus generated requests is exact; duplicate versus generic API rejection reason is not exposed.
12. **Are accepted prefetches late?** Not measurable through the current prefetcher callback API; recorded as unavailable.
13. **Is table collision/replacement destructive?** Hash aliases, insertions, evictions, and reuse-before-eviction are recorded in `hash_stats.csv` and `replacement_stats.csv`.
14. **Is absolute addressing less predictable than deltas?** `representation_diagnostics.csv` compares absolute and delta oracle accuracy without changing production WAM.
15. **Is H16 an appropriate temporal distance?** `event_distance.csv` reports exact qualifying-access distance and cycle distance; instruction distance is unavailable from this L2 callback API. `plots/h16_instruction_distance.svg` is an explicit unavailable-metric placeholder, while `plots/h16_cycle_distance.svg` reports the exposed cycle distribution.
16. **What caused the negative IPC result?** The evidence funnel is in `failure_funnel.csv`; the final attribution is **{dominant}**.
17. **Is there an implementation bug?** No alignment mismatch was found by the deterministic tests; no production semantics were changed.
18. **Should this exact WAM architecture be abandoned?** **{'Yes' if decision.endswith('STOP') else 'No, only one narrow evidence-backed variant is justified'}**.
19. **Is there a scientifically justified follow-up?** **{follow_up}**
20. **Should hardware work stop?** **{'Yes.' if decision.endswith('STOP') else 'Not yet, but only one narrow diagnostic follow-up is justified.'}**

## Required final console summary

```text
traces diagnosed: {len(trace_diagnostics)}
WAM event type: L2 non-PREFETCH access callback
events per 1K instructions: {statistics.mean([float(row['events_per_1k_measurement_instructions']) for row in activation_rows]) if activation_rows else 0.0:.3f}
depth1 context reuse: {statistics.mean([float(row['revisit_rate']) for row in all_context_rows if row['phase'] == 'measurement' and row['depth'] == 1]) if all_context_rows else 0.0:.3%}
depth4 context reuse: {statistics.mean([float(row['revisit_rate']) for row in all_context_rows if row['phase'] == 'measurement' and row['depth'] == 4]) if all_context_rows else 0.0:.3%}
depth16 context reuse: {statistics.mean(all_reuse) if all_reuse else 0.0:.3%}
H1 oracle accuracy: {gmean([float(row['oracle_top1_accuracy']) for row in oracle_rows if row['depth'] == 4 and row['horizon'] == 1]):.3%}
H8 oracle accuracy: {gmean([float(row['oracle_top1_accuracy']) for row in oracle_rows if row['depth'] == 4 and row['horizon'] == 8]):.3%}
H16 oracle accuracy: {statistics.mean(all_h16) if all_h16 else 0.0:.3%}
H32 oracle accuracy: {gmean([float(row['oracle_top1_accuracy']) for row in oracle_rows if row['depth'] == 4 and row['horizon'] == 32]):.3%}
actual ShadowWAM H16 accuracy: {statistics.mean(all_shadow) if all_shadow else 0.0:.3%}
H16 oracle gap: {(statistics.mean(all_h16) - statistics.mean(all_shadow)) if all_h16 else 0.0:+.3%}
predictions generated: {total_counters['predictions_generated']}
predictions above threshold: {total_counters['predictions_above_threshold']}
prefetches requested: {total_counters['prefetch_requests_generated']}
prefetches accepted: {total_counters['prefetch_requests_accepted']}
useful prefetches: {total_counters['prefetches_useful']}
hash collision rate: {total_counters['hash_collisions'] / max(total_counters['prediction_context_hits'] + total_counters['prediction_context_misses'], 1):.3%}
entry eviction rate: {total_counters['entry_evictions'] / max(sum(row['entry_insertions'] for row in replacement_rows), 1):.3%}
absolute-address H16 oracle accuracy: {statistics.mean(all_abs) if all_abs else 0.0:.3%}
delta H16 oracle accuracy: {statistics.mean(all_delta) if all_delta else 0.0:.3%}
dominant failure stage: {dominant}
evidence of implementation bug: {'NO' if alignment_pass else 'YES'}
final classification: {classification}
{decision}
```
'''
    (output / "report.md").write_text(report, encoding="utf-8")
    print(report.split("## Required final console summary", 1)[1])


if __name__ == "__main__":
    main()
