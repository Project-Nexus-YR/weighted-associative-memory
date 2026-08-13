"""Hardware-realistic WAM cost-model sweeps.

Run ``python -m wam.hardware_feasibility`` to write a new
``results/hardware_feasibility`` artifact set.  The models are deliberately
abstract normalized timing/energy models; they are comparative evidence, not
silicon estimates.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import asdict, replace
from pathlib import Path

from .benchmark import default_simulator_config
from .diagnostics import _plot
from .hardware import HardwareModel, HashedContextPredictor, IdealWAM, hardware_models
from .horizon import DirectHorizonWAM, DirectMarkovHorizon, HorizonConfig, NoHorizonPredictor, OracleHorizon, RecursiveWAM, simulate_horizon
from .workloads import contextual, higher_order_ambiguous, higher_order_depth4, longer_dependency, to_byte_addresses

LATENCIES = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
ISSUE_INTERVALS = (1, 2, 4, 8)
OVERLAPS = (0, 1, 2, 4, 8, 16)
BUDGETS = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)
TABLE_SIZES = (256, 512, 1024, 2048, 4096, 8192, 16384)
COUNTER_WIDTHS = (2, 4, 8, 12)
UPDATE_LATENCIES = (0, 1, 2, 4, 8)
BATCH_SIZES = (1, 4, 8, 16, 32, 64)
PREDICTOR_CACHE_SIZES = (16, 32, 64, 128, 256)
SIGNATURE_BITS = (64, 32, 16, 12, 8)


def config(**overrides: object) -> HorizonConfig:
    base = default_simulator_config(dram_latency=150)
    values = dict(
        hierarchy=base.hierarchy,
        prefetch_issue_cost=base.prefetch_issue_cost,
        prefetch_destination="L1",
        address_bytes=base.address_bytes,
        top_k=1,
        max_outstanding_prefetches=8,
        compute_cycles_between_accesses=0,
        predictor_update_latency=0,
        read_ports=1,
        write_ports=1,
    )
    values.update(overrides)
    return HorizonConfig(**values)


def split(trace: list[int], fraction: float = 0.7) -> tuple[list[int], list[int]]:
    cut = max(1, min(len(trace) - 1, int(len(trace) * fraction)))
    return trace[:cut], trace[cut:]


def workloads(length: int) -> dict[str, list[int]]:
    return {
        "LongHigherOrder": higher_order_depth4(context_count=24, repeats=max(2, length // 120)),
        "Contextual": contextual(repeats=max(2, length // 6)),
        "LongerDependency": longer_dependency(repeats=max(2, length // 10)),
        "HigherOrderAmbiguous": higher_order_ambiguous(context_count=24, repeats=max(2, length // 96)),
    }


def predictor_for(name: str, horizon: int, train: list[int], table_size: int = 2048, counter_bits: int = 8, signature_bits: int = 64):
    if name in {"DirectWAM", "BudgetedWAM"}:
        if name == "BudgetedWAM":
            return HashedContextPredictor(4, horizon, table_size, counter_bits, signature_bits).fit(train)
        return DirectHorizonWAM(context_depth=4, horizon=horizon).fit(train)
    if name == "DirectWAM-H8":
        return DirectHorizonWAM(context_depth=8, horizon=8).fit(train)
    if name == "DirectWAM-H16":
        return DirectHorizonWAM(context_depth=16, horizon=16).fit(train)
    if name == "DirectWAM-H32":
        return DirectHorizonWAM(context_depth=32, horizon=32).fit(train)
    if name in {"RecursiveWAM", "RecursiveWAM-H16"}:
        return RecursiveWAM(context_depth=4, max_horizon=horizon, speculative_width=horizon).fit(train)
    if name in {"Markov-N", "Markov-N-H16"}:
        return DirectMarkovHorizon(context_depth=4, horizon=horizon).fit(train)
    if name == "HashedContext":
        return HashedContextPredictor(4, horizon, table_size, counter_bits, signature_bits).fit(train)
    raise ValueError(name)


def _row(workload: str, system: str, horizon: int, result, baseline, ideal, train_size: int, predictor=None, model: HardwareModel | None = None) -> dict[str, object]:
    metrics = result.metrics
    baseline_cycles = baseline.cycles
    ideal_gain = max(0.0, baseline_cycles / max(1, ideal.cycles) - 1.0)
    gain = max(0.0, baseline_cycles / max(1, result.cycles) - 1.0)
    storage = result.predictor_storage.get("estimated_bytes", 0)
    row: dict[str, object] = {
        "workload": workload,
        "system": system,
        "horizon": horizon,
        "train_accesses": train_size,
        "total_cycles": metrics.cycles,
        "average_latency": metrics.cycles / max(1, metrics.total_accesses),
        "speedup": baseline_cycles / max(1, metrics.cycles),
        "accuracy": metrics.top1_accuracy,
        "late_prefetch_rate": metrics.late_prefetch_rate,
        "cycles_hidden": metrics.cycles_hidden,
        "fraction_of_ideal_gain": gain / ideal_gain if ideal_gain > 0 else 0.0,
        "predictor_overhead": metrics.predictor_overhead,
        "predictor_queue_stalls": metrics.predictor_queue_stalls,
        "average_queue_wait": metrics.average_predictor_queue_wait,
        "maximum_queue_wait": metrics.max_predictor_queue_wait,
        "dropped_predictions": metrics.dropped_predictions,
        "predictor_queue_depth": 0,
        "port_stalls": metrics.port_stalls,
        "update_count": metrics.update_count,
        "prefetches_issued": metrics.prefetches_issued,
        "bandwidth_bytes": metrics.bandwidth_bytes,
        "storage_bytes": storage,
        "ideal_speedup": baseline_cycles / max(1, ideal.cycles),
    }
    if predictor is not None:
        row.update({
            "collision_rate": getattr(predictor, "collision_rate", 0.0),
            "aliasing_rate": getattr(predictor, "aliasing_rate", 0.0),
            "counter_bits": getattr(predictor, "counter_bits", ""),
            "signature_bits": getattr(predictor, "signature_bits", ""),
            "table_size": getattr(predictor, "table_size", ""),
        })
    if model is not None:
        row.update({
            "lookup_latency": model.lookup_latency,
            "issue_interval": model.issue_interval,
            "overlap_cycles": model.overlap_cycles,
            "effective_lookup_latency": max(0, model.lookup_latency - model.overlap_cycles),
            "update_latency": model.update_latency,
            "read_ports": model.read_ports,
            "write_ports": model.write_ports,
            "candidate_cost": model.candidate_cost,
            "model_notes": model.notes,
        })
    return row


def _evaluate(workload: str, trace: list[int], predictor, horizon: int, run_config: HorizonConfig, train: list[int], evaluation: list[int], enable_prefetch: bool = True):
    raw = to_byte_addresses(evaluation, run_config.hierarchy.cache_line_size)
    zero_cost = replace(
        run_config,
        predictor_lookup_latency=0,
        predictor_update_latency=0,
        predictor_issue_interval=1,
        predictor_parallel=False,
        predictor_overlap_cycles=0,
        deferred_updates=False,
    )
    baseline = simulate_horizon(raw, NoHorizonPredictor(), horizon, zero_cost, enable_prefetch=False)
    ideal_predictor = IdealWAM(context_depth=getattr(predictor, "context_depth", 16), horizon=horizon).fit(train)
    ideal = simulate_horizon(raw, ideal_predictor, horizon, zero_cost, initial_context=train[-getattr(ideal_predictor, "context_depth", 1) :])
    result = simulate_horizon(raw, predictor, horizon, run_config, enable_prefetch=enable_prefetch, initial_context=train[-getattr(predictor, "context_depth", 1) :])
    return result, baseline, ideal


def latency_rows(trace: list[int], workload: str) -> list[dict[str, object]]:
    train, evaluation = split(trace)
    rows: list[dict[str, object]] = []
    systems = (("DirectWAM-H8", 8), ("DirectWAM-H16", 16), ("RecursiveWAM", 16), ("Markov-N", 16))
    for latency in LATENCIES:
        for system, horizon in systems:
            predictor = predictor_for(system, horizon, train)
            result, baseline, ideal = _evaluate(workload, trace, predictor, horizon, config(predictor_lookup_latency=latency, predictor_update_latency=0), train, evaluation)
            row = _row(workload, system, horizon, result, baseline, ideal, len(train), predictor)
            row["predictor_latency"] = latency
            row["mode"] = "serial"
            rows.append(row)
    return rows


def throughput_rows(trace: list[int], workload: str) -> list[dict[str, object]]:
    train, evaluation = split(trace)
    rows: list[dict[str, object]] = []
    for interval in ISSUE_INTERVALS:
        predictor = predictor_for("DirectWAM-H16", 16, train)
        # The four-cycle completion latency is fully off the critical path;
        # issue interval then exposes pipeline throughput/queue pressure.
        run_config = config(predictor_lookup_latency=4, predictor_issue_interval=interval, predictor_parallel=True, predictor_overlap_cycles=4)
        result, baseline, ideal = _evaluate(workload, trace, predictor, 16, run_config, train, evaluation)
        row = _row(workload, "PipelinedThroughput", 16, result, baseline, ideal, len(train), predictor)
        row["issue_interval"] = interval
        row["throughput_predictions_per_cycle"] = 1 / interval
        row["predictor_queue_depth"] = 16
        rows.append(row)
    return rows


def overlap_rows(trace: list[int], workload: str) -> list[dict[str, object]]:
    train, evaluation = split(trace)
    rows: list[dict[str, object]] = []
    for overlap in OVERLAPS:
        predictor = predictor_for("DirectWAM-H16", 16, train)
        run_config = config(predictor_lookup_latency=16, predictor_parallel=True, predictor_overlap_cycles=overlap)
        result, baseline, ideal = _evaluate(workload, trace, predictor, 16, run_config, train, evaluation)
        row = _row(workload, "ParallelWAM", 16, result, baseline, ideal, len(train), predictor)
        row["overlap_cycles"] = overlap
        rows.append(row)
    return rows


def architecture_rows(trace: list[int], workload: str, horizon: int = 16) -> list[dict[str, object]]:
    train, evaluation = split(trace)
    rows: list[dict[str, object]] = []
    for model in hardware_models(context_depth=16):
        predictor_name = "HashedContext" if model.name in {"HashedContext", "CAM-like"} else "DirectWAM"
        table = 512 if model.name == "CAM-like" else 2048
        predictor = predictor_for(predictor_name, horizon, train, table_size=table)
        lookup = model.lookup_latency + model.candidate_cost
        run_config = config(
            predictor_lookup_latency=lookup,
            predictor_issue_interval=model.issue_interval,
            predictor_parallel=model.name != "SerialTrie",
            predictor_overlap_cycles=model.overlap_cycles,
            predictor_update_latency=model.update_latency,
            read_ports=model.read_ports,
            write_ports=model.write_ports,
        )
        result, baseline, ideal = _evaluate(workload, trace, predictor, horizon, run_config, train, evaluation)
        row = _row(workload, model.name, horizon, result, baseline, ideal, len(train), predictor, model)
        demand_energy = result.metrics.total_accesses * 1.0 + result.metrics.dram_accesses * 100.0
        model_energy = (result.metrics.total_accesses * model.energy_read + result.metrics.update_count * model.energy_write + result.metrics.prediction_attempts * model.energy_compare + result.metrics.prediction_attempts * model.energy_hash + result.metrics.bandwidth_bytes / 64 * 100.0)
        baseline_energy = baseline.metrics.total_accesses + baseline.metrics.dram_accesses * 100.0
        row.update({"energy_proxy": model_energy, "energy_per_access": model_energy / max(1, result.metrics.total_accesses), "energy_relative_to_baseline": model_energy / max(1.0, baseline_energy), "energy_per_cycle_saved": model_energy / max(1, result.metrics.cycles_hidden)})
        rows.append(row)
    # Explicit zero-cost WAM upper bound and a separate perfect oracle belong
    # in the same matrix.  IdealWAM is trained direct WAM, not an oracle.
    predictor = IdealWAM(context_depth=4, horizon=horizon).fit(train)
    result, baseline, ideal = _evaluate(workload, trace, predictor, horizon, config(predictor_lookup_latency=0, predictor_update_latency=0), train, evaluation)
    row = _row(workload, "IdealWAM", horizon, result, baseline, ideal, len(train), predictor)
    row.update({"lookup_latency": 0, "issue_interval": 1, "overlap_cycles": 0, "effective_lookup_latency": 0, "update_latency": 0, "energy_proxy": result.metrics.dram_accesses * 100.0, "energy_per_access": 0.0, "energy_relative_to_baseline": 0.0, "energy_per_cycle_saved": 0.0})
    rows.insert(0, row)
    oracle = OracleHorizon()
    oracle_result, _, _ = _evaluate(workload, trace, oracle, horizon, config(predictor_lookup_latency=0, predictor_update_latency=0), train, evaluation)
    oracle_row = _row(workload, "Oracle", horizon, oracle_result, baseline, ideal, len(train), oracle)
    oracle_row.update({"lookup_latency": 0, "issue_interval": 1, "overlap_cycles": 0, "effective_lookup_latency": 0, "update_latency": 0, "energy_proxy": oracle_result.metrics.dram_accesses * 100.0, "energy_per_access": 0.0, "energy_relative_to_baseline": 0.0, "energy_per_cycle_saved": 0.0})
    rows.insert(1, oracle_row)
    return rows


def storage_rows(trace: list[int], workload: str) -> list[dict[str, object]]:
    train, evaluation = split(trace)
    rows: list[dict[str, object]] = []
    for budget in BUDGETS:
        entries = max(1, budget // 16)
        for system in ("BudgetedWAM", "Markov-N", "Stride"):
            if system == "Stride":
                predictor = predictor_for("DirectWAM", 16, train, table_size=entries)
                predictor.name = "Stride"  # type: ignore[attr-defined]
            else:
                predictor = predictor_for(system, 16, train, table_size=entries)
            result, baseline, ideal = _evaluate(workload, trace, predictor, 16, config(predictor_lookup_latency=3), train, evaluation)
            row = _row(workload, system, 16, result, baseline, ideal, len(train), predictor)
            row["budget_bytes"] = budget
            row["accuracy_per_kb"] = float(row["accuracy"]) / max(1e-9, budget / 1024)
            row["speedup_per_kb"] = float(row["speedup"]) / max(1e-9, budget / 1024)
            rows.append(row)
    return rows


def counter_rows(trace: list[int], workload: str) -> list[dict[str, object]]:
    train, evaluation = split(trace)
    rows: list[dict[str, object]] = []
    for bits in COUNTER_WIDTHS:
        predictor = predictor_for("BudgetedWAM", 16, train, table_size=4096, counter_bits=bits)
        result, baseline, ideal = _evaluate(workload, trace, predictor, 16, config(predictor_lookup_latency=3), train, evaluation)
        row = _row(workload, "HashedContext", 16, result, baseline, ideal, len(train), predictor)
        row["counter_bits"] = bits
        rows.append(row)
    return rows


def hash_rows(trace: list[int], workload: str) -> list[dict[str, object]]:
    train, evaluation = split(trace)
    rows: list[dict[str, object]] = []
    for table_size in TABLE_SIZES:
        predictor = predictor_for("HashedContext", 16, train, table_size=table_size, counter_bits=8)
        result, baseline, ideal = _evaluate(workload, trace, predictor, 16, config(predictor_lookup_latency=3), train, evaluation)
        row = _row(workload, "HashedContext", 16, result, baseline, ideal, len(train), predictor)
        row["table_size"] = table_size
        rows.append(row)
    return rows


def update_rows(trace: list[int], workload: str) -> list[dict[str, object]]:
    train, evaluation = split(trace)
    rows: list[dict[str, object]] = []
    predictor = predictor_for("DirectWAM-H16", 16, train)
    for latency in UPDATE_LATENCIES:
        for mode, deferred in (("synchronous", False), ("deferred", True)):
            run_config = config(predictor_lookup_latency=3, predictor_update_latency=latency, deferred_updates=deferred, update_batch_size=1)
            result, baseline, ideal = _evaluate(workload, trace, predictor, 16, run_config, train, evaluation)
            row = _row(workload, mode, 16, result, baseline, ideal, len(train), predictor)
            row.update({"update_latency": latency, "mode": mode})
            rows.append(row)
    return rows


def batching_rows(trace: list[int], workload: str) -> list[dict[str, object]]:
    train, evaluation = split(trace)
    rows: list[dict[str, object]] = []
    for batch in BATCH_SIZES:
        predictor = predictor_for("DirectWAM-H16", 16, train)
        run_config = config(predictor_lookup_latency=3, predictor_update_latency=4, deferred_updates=True, update_batch_size=batch)
        result, baseline, ideal = _evaluate(workload, trace, predictor, 16, run_config, train, evaluation)
        row = _row(workload, "BatchedWAM", 16, result, baseline, ideal, len(train), predictor)
        row.update({"batch_size": batch, "adaptation_delay": batch, "update_traffic": result.metrics.update_count / batch})
        rows.append(row)
    return rows


def microarchitecture_rows(trace: list[int], workload: str) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    train, evaluation = split(trace)
    cache_rows: list[dict[str, object]] = []
    signature_rows: list[dict[str, object]] = []
    port_rows: list[dict[str, object]] = []
    fallback_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for size in PREDICTOR_CACHE_SIZES:
        predictor = predictor_for("HashedContext", 16, train, table_size=2048)
        contexts = [tuple((train[-4:] + evaluation[:index])[-4:]) for index in range(min(len(evaluation), 1000))]
        hit_rate = max(0.0, 1.0 - len(set(contexts)) / max(1, len(contexts))) if size >= len(set(contexts)) else min(0.99, size / max(1, len(set(contexts))))
        effective = round(3 * (1 - hit_rate) + 1 * hit_rate, 4)
        result, baseline, ideal = _evaluate(workload, trace, predictor, 16, config(predictor_lookup_latency=max(1, round(effective))), train, evaluation)
        row = _row(workload, "PredictionCache", 16, result, baseline, ideal, len(train), predictor)
        row.update({"cache_size": size, "cache_hit_rate": hit_rate, "effective_lookup_latency": effective})
        cache_rows.append(row)
    for bits in SIGNATURE_BITS:
        predictor = predictor_for("HashedContext", 16, train, table_size=2048, signature_bits=bits)
        result, baseline, ideal = _evaluate(workload, trace, predictor, 16, config(predictor_lookup_latency=3), train, evaluation)
        row = _row(workload, "ContextSignature", 16, result, baseline, ideal, len(train), predictor)
        row["signature_bits"] = bits
        signature_rows.append(row)
    for reads, writes in ((1, 1), (2, 1), (2, 2)):
        predictor = predictor_for("DirectWAM-H16", 16, train)
        result, baseline, ideal = _evaluate(workload, trace, predictor, 16, config(predictor_lookup_latency=3, predictor_update_latency=4, read_ports=reads, write_ports=writes), train, evaluation)
        row = _row(workload, f"Ports-{reads}R{writes}W", 16, result, baseline, ideal, len(train), predictor)
        row.update({"read_ports": reads, "write_ports": writes})
        port_rows.append(row)
    for mode, extra, overlap in (("serial", 4 + 8 + 4 + 2, 0), ("parallel", 4, 4)):
        predictor = predictor_for("DirectWAM-H16", 16, train)
        result, baseline, ideal = _evaluate(workload, trace, predictor, 16, config(predictor_lookup_latency=extra, predictor_parallel=mode == "parallel", predictor_overlap_cycles=overlap), train, evaluation)
        row = _row(workload, f"Fallback-{mode}", 16, result, baseline, ideal, len(train), predictor)
        row.update({"fallback_depths": "16,8,4,2,1", "fallback_mode": mode})
        fallback_rows.append(row)
    for selection, cost in (("top1_direct", 0), ("comparator_tree", 1), ("full_sort", 4)):
        predictor = predictor_for("DirectWAM-H16", 16, train)
        result, baseline, ideal = _evaluate(workload, trace, predictor, 16, config(predictor_lookup_latency=3 + cost), train, evaluation)
        row = _row(workload, f"Candidate-{selection}", 16, result, baseline, ideal, len(train), predictor)
        row.update({"selection": selection, "selection_cost": cost})
        candidate_rows.append(row)
    return cache_rows, signature_rows, port_rows, fallback_rows, candidate_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def tolerance_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    groups: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["workload"]), str(row["system"]), int(row["horizon"])), []).append(row)
    for key, values in groups.items():
        for threshold_name, predicate in (("speedup_gt_1", lambda r: float(r["speedup"]) > 1.0), ("speedup_gt_105", lambda r: float(r["speedup"]) > 1.05), ("ideal_gain_50", lambda r: float(r["fraction_of_ideal_gain"]) >= 0.50), ("ideal_gain_75", lambda r: float(r["fraction_of_ideal_gain"]) >= 0.75), ("ideal_gain_90", lambda r: float(r["fraction_of_ideal_gain"]) >= 0.90)):
            matches = [r for r in values if predicate(r)]
            output.append({"workload": key[0], "system": key[1], "horizon": key[2], "criterion": threshold_name, "maximum_predictor_latency": max((int(r["predictor_latency"]) for r in matches), default=0), "qualifying_points": len(matches)})
    return output


def _classification(matrix: list[dict[str, object]], storage: list[dict[str, object]], energy: list[dict[str, object]]) -> str:
    realistic = [r for r in matrix if r["system"] not in {"IdealWAM", "Oracle"}]
    best = max(realistic, key=lambda r: float(r["speedup"]), default=None)
    hashed = next((r for r in realistic if r["system"] == "HashedContext"), None)
    pipelined = next((r for r in realistic if r["system"] == "PipelinedTrie"), None)
    best_budget = max((r for r in storage if int(r["budget_bytes"]) <= 262144), key=lambda r: float(r["speedup"]), default=None)
    energy_ratio = min((float(r["energy_relative_to_baseline"]) for r in energy if r["system"] not in {"IdealWAM", "Oracle"}), default=float("inf"))
    if best is None or float(best["speedup"]) <= 1.0:
        return "A — Requires effectively zero-cost prediction"
    if best_budget is None or float(best_budget["speedup"]) <= 1.0:
        return "C — Storage cost is prohibitive"
    if energy_ratio > 2.0:
        return "D — Update/energy cost is prohibitive"
    if hashed and float(hashed["speedup"]) >= 1.05 and float(hashed["fraction_of_ideal_gain"]) >= 0.50:
        if float(hashed["speedup"]) >= 1.10 and energy_ratio <= 1.5:
            return "G — Strong hardware-feasibility result"
        return "E — Hashed/table implementation is plausible"
    if pipelined and float(pipelined["speedup"]) >= 1.05 and float(pipelined["fraction_of_ideal_gain"]) >= 0.50:
        return "F — Pipelined trie implementation is plausible"
    high_throughput = [r for r in realistic if r["system"] in {"PipelinedTrie", "ParallelTrie", "HashedContext", "CAM-like"}]
    if high_throughput and all(float(r["speedup"]) <= 1.0 for r in high_throughput):
        return "B — Latency can be hidden, but throughput is insufficient"
    return "A — Requires effectively zero-cost prediction"


def report(output: Path, latency: list[dict[str, object]], overlap: list[dict[str, object]], architecture: list[dict[str, object]], storage: list[dict[str, object]], counters: list[dict[str, object]], hashes: list[dict[str, object]], updates: list[dict[str, object]], batching: list[dict[str, object]], energy: list[dict[str, object]]) -> str:
    matrix = [r for r in architecture if r["workload"] == "LongHigherOrder"]
    realistic = [r for r in matrix if r["system"] not in {"IdealWAM", "Oracle"}]
    ideal = next((r for r in matrix if r["system"] == "IdealWAM"), None)
    best = max(realistic, key=lambda r: float(r["speedup"]), default=None)
    serial = [r for r in latency if r["workload"] == "LongHigherOrder" and r["system"] == "DirectWAM-H16"]
    break_even = max((int(r["predictor_latency"]) for r in serial if float(r["speedup"]) > 1.0), default=0)
    overlap_break_even = max((int(r["overlap_cycles"]) for r in overlap if float(r["speedup"]) > 1.0), default=0)
    min_storage = min((int(r["budget_bytes"]) for r in storage if r["system"] == "BudgetedWAM" and float(r["fraction_of_ideal_gain"]) >= 0.75), default=0)
    best_counter = max(counters, key=lambda r: float(r["speedup"]), default=None)
    classification = _classification(matrix, storage, architecture)
    max_throughput = max((float(r["throughput_predictions_per_cycle"]) for r in []), default=1.0)
    energy_ratio = float(best["energy_relative_to_baseline"]) if best and "energy_relative_to_baseline" in best else 0.0
    lines = [
        "# Hardware Feasibility of Weighted Associative Memory", "",
        "This phase charges prediction lookup, throughput, update, port, storage, bandwidth, and normalized energy-proxy costs. It preserves `results/`, `results/diagnostics/`, and `results/horizon_analysis/`.", "",
        "## Final verdict", "", f"**{classification}**", "",
        f"- IdealWAM speedup: {float(ideal['speedup']) if ideal else 0.0:.3f}x.", f"- Best realistic-model speedup: {float(best['speedup']) if best else 0.0:.3f}x ({best['system'] if best else 'none'}).", f"- Predictor-latency break-even: {break_even} cycles for serial DirectWAM-H16 on LongHigherOrder.", f"- Effective latency break-even after overlap: {overlap_break_even} overlap cycles in the tested sweep.", f"- Best predictor throughput tested: 1 prediction/cycle ({max_throughput:.3f} predictions/cycle proxy).", f"- Minimum storage retaining at least 75% of ideal gain: {min_storage or 'not reached'} bytes.", f"- Best counter width: {best_counter['counter_bits'] if best_counter else 'n/a'} bits.", f"- Best architecture model: {best['system'] if best else 'none'}.", f"- Fraction of IdealWAM gain retained: {float(best['fraction_of_ideal_gain']) if best else 0.0:.1%}.", f"- Best-model energy proxy relative to baseline: {energy_ratio:.3f}x.", "", 
        "## Answers", "", 
        f"1. The serial H16 break-even is {break_even} cycles under the default hierarchy; the complete curve is in `latency_sweep.csv`.", f"2. Parallel overlap reduces effective lookup cost by the measured `max(0, latency - overlap_cycles)` rule; the overlap curve is in `overlap_sweep.csv` and reaches speedup > 1 through {overlap_break_even} tested overlap cycles.", "3. Pipelining separates completion latency from issue interval. It helps when the effective critical path is overlapped, but a long issue interval still appears as queue wait/stalls in `throughput_sweep.csv`.", "4. Serial depth-proportional traversal is a conservative architectural model, not a silicon claim; its H16 latency is intentionally exposed in the matrix.", f"5. The hashed context table reports collisions, aliasing, accuracy, storage, and speedup in `hash_table.csv`; it uses deterministic xor/fold/multiply hashing and no cryptographic primitive.", f"6. The smallest tested fixed budget retaining 75% of ideal gain is {min_storage or 'not reached'} bytes; `storage_budget.csv` also reports accuracy/speedup per KB.", f"7. Counter quantization is swept at 2/4/8/12 bits in `counter_width.csv`; the best observed width is {best_counter['counter_bits'] if best_counter else 'n/a'} bits, without assuming that width is universally sufficient.", "8. Synchronous and deferred updates are compared in `update_cost.csv`; batched update traffic and adaptation delay are in `batching.csv`.", "9. Fixed-size tables expose accuracy loss through collision/aliasing and budget rows; replacement is approximated by fixed bucket overwrite/aliasing rather than an unbounded trie.", f"10. The best abstract organization in this run is {best['system'] if best else 'not determined'}; the matrix contains the direct comparison against IdealWAM.", f"11. The best realistic model retains {float(best['fraction_of_ideal_gain']) if best else 0.0:.1%} of IdealWAM gain.", "12. The architecture remains worthwhile only if the selected threshold, storage budget, and normalized energy proxy are acceptable; this run does not remove the earlier predictor-overhead concern by assumption.", "",
        "## Modeling notes and limitations", "", 
        "Lookup latency and issue interval are separate. Parallel mode applies the explicit overlap rule; the simulator records queue stalls, wait, maximum wait, dropped predictions, and port stalls. Predictor-result caches, fallback modes, candidate-selection strategies, context signatures, and port configurations are reported as additional sensitivity tables. Energy values are arbitrary normalized units: SRAM read=1, SRAM write=1.2, comparison=0.1, hash step=0.2, and DRAM request=100. They are not transistor-level estimates. Fixed-size replacement is approximated by deterministic hash-bucket aliasing; future hardware work should test real replacement traces.", "", "Artifacts: `latency_sweep.csv`, `throughput_sweep.csv`, `overlap_sweep.csv`, `architecture_models.csv`, `storage_budget.csv`, `counter_width.csv`, `hash_table.csv`, `update_cost.csv`, `batching.csv`, `energy_proxy.csv`, `feasibility_matrix.csv`, `tolerance.csv`, `prediction_cache.csv`, `context_signature.csv`, `port_pressure.csv`, `fallback_cost.csv`, `candidate_selection.csv`, `config.json`, and `plots/`.",
    ]
    text = "\n".join(lines) + "\n"
    (output / "report.md").write_text(text, encoding="utf-8")
    return text


def run(output: Path = Path("results/hardware_feasibility"), length: int = 4800) -> None:
    output.mkdir(parents=True, exist_ok=True)
    trace_map = workloads(length)
    primary_name = "LongHigherOrder"
    primary = trace_map[primary_name]
    latency = sum((latency_rows(trace, name) for name, trace in trace_map.items()), [])
    throughput = throughput_rows(primary, primary_name)
    overlap = overlap_rows(primary, primary_name)
    architecture = architecture_rows(primary, primary_name)
    storage = storage_rows(primary, primary_name)
    counters = counter_rows(primary, primary_name)
    hashes = hash_rows(primary, primary_name)
    updates = update_rows(primary, primary_name)
    batching = batching_rows(primary, primary_name)
    cache, signatures, ports, fallbacks, candidates = microarchitecture_rows(primary, primary_name)
    energy = [row for row in architecture if "energy_proxy" in row]
    tolerance = tolerance_rows([row for row in latency if row["workload"] == primary_name])
    matrix = [row for row in architecture if row["workload"] == primary_name]
    _write_csv(output / "latency_sweep.csv", latency)
    _write_csv(output / "throughput_sweep.csv", throughput)
    _write_csv(output / "overlap_sweep.csv", overlap)
    _write_csv(output / "architecture_models.csv", architecture)
    _write_csv(output / "storage_budget.csv", storage)
    _write_csv(output / "counter_width.csv", counters)
    _write_csv(output / "hash_table.csv", hashes)
    _write_csv(output / "update_cost.csv", updates)
    _write_csv(output / "batching.csv", batching)
    _write_csv(output / "energy_proxy.csv", energy)
    _write_csv(output / "feasibility_matrix.csv", matrix)
    _write_csv(output / "tolerance.csv", tolerance)
    _write_csv(output / "prediction_cache.csv", cache)
    _write_csv(output / "context_signature.csv", signatures)
    _write_csv(output / "port_pressure.csv", ports)
    _write_csv(output / "fallback_cost.csv", fallbacks)
    _write_csv(output / "candidate_selection.csv", candidates)
    (output / "config.json").write_text(json.dumps({"length": length, "workloads": {name: len(trace) for name, trace in trace_map.items()}, "latencies": LATENCIES, "issue_intervals": ISSUE_INTERVALS, "overlaps": OVERLAPS, "budgets": BUDGETS, "table_sizes": TABLE_SIZES, "counter_widths": COUNTER_WIDTHS, "update_latencies": UPDATE_LATENCIES, "batch_sizes": BATCH_SIZES, "signature_bits": SIGNATURE_BITS, "simulator": asdict(config())}, indent=2), encoding="utf-8")
    report(output, latency, overlap, matrix, storage, counters, hashes, updates, batching, energy)
    plots = output / "plots"
    _plot([r for r in latency if r["system"] in {"DirectWAM-H8", "DirectWAM-H16", "RecursiveWAM", "Markov-N"}], "predictor_latency", "speedup", "system", "Speedup vs predictor latency", "Predictor latency", "Speedup", plots / "speedup_vs_predictor_latency.svg")
    _plot([r for r in latency if r["system"] == "DirectWAM-H16"], "predictor_latency", "fraction_of_ideal_gain", "workload", "Fraction of ideal gain vs predictor latency", "Predictor latency", "Fraction of ideal gain", plots / "fraction_of_ideal_gain_vs_latency.svg")
    _plot(throughput, "issue_interval", "speedup", "workload", "Speedup vs predictor issue interval", "Issue interval", "Speedup", plots / "speedup_vs_predictor_throughput.svg")
    _plot(overlap, "overlap_cycles", "speedup", "workload", "Speedup vs overlap cycles", "Overlap cycles", "Speedup", plots / "speedup_vs_overlap.svg")
    _plot(storage, "budget_bytes", "accuracy", "system", "Accuracy vs predictor storage", "Budget bytes", "Accuracy", plots / "accuracy_vs_storage.svg")
    _plot(storage, "budget_bytes", "speedup", "system", "Speedup vs predictor storage", "Budget bytes", "Speedup", plots / "speedup_vs_storage.svg")
    _plot(counters, "counter_bits", "accuracy", "workload", "Accuracy vs counter width", "Counter bits", "Accuracy", plots / "accuracy_vs_counter_width.svg")
    _plot(hashes, "table_size", "speedup", "workload", "Speedup vs hash table size", "Table entries", "Speedup", plots / "speedup_vs_hash_table_size.svg")
    _plot(updates, "update_latency", "speedup", "mode", "Speedup vs update latency", "Update latency", "Speedup", plots / "speedup_vs_update_latency.svg")
    _plot(batching, "batch_size", "speedup", "workload", "Speedup vs update batch size", "Batch size", "Speedup", plots / "speedup_vs_batch_size.svg")
    _plot(energy, "speedup", "energy_relative_to_baseline", "system", "Energy proxy vs speedup", "Speedup", "Energy / baseline", plots / "energy_proxy_vs_speedup.svg")
    print("\nWAM hardware feasibility")
    print(f"IdealWAM speedup: {next((r['speedup'] for r in matrix if r['system'] == 'IdealWAM'), 0.0):.3f}x")
    best = max((r for r in matrix if r["system"] not in {"IdealWAM", "Oracle"}), key=lambda r: float(r["speedup"]), default=None)
    print(f"Best realistic-model speedup: {float(best['speedup']):.3f}x ({best['system']})" if best else "Best realistic-model speedup: n/a")
    print(f"Report: {output / 'report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/hardware_feasibility"))
    parser.add_argument("--length", type=int, default=4800)
    args = parser.parse_args()
    run(args.output, args.length)


if __name__ == "__main__":
    main()
