"""Replay traces through a hierarchy with realistic-enough prefetch timing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .hierarchy import HierarchyConfig, MemoryHierarchy
from .metrics import SimulationMetrics
from .predictor import Prediction, Predictor


@dataclass(frozen=True)
class SimulatorConfig:
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    prefetch_issue_cost: int = 1
    # Backward-compatible alias used by the original MVP API.
    prefetch_cost: int | None = None
    prefetch_destination: str = "L1"
    address_bytes: int = 8
    top_k: int = 3
    max_outstanding_prefetches: int = 8
    predictor_lookup_cost: int | None = None
    predictor_update_cost: int | None = None

    @property
    def cache_line_size(self) -> int:
        return self.hierarchy.cache_line_size

    @property
    def effective_prefetch_issue_cost(self) -> int:
        return self.prefetch_issue_cost if self.prefetch_cost is None else self.prefetch_cost


@dataclass
class SimulationResult:
    metrics: SimulationMetrics
    predictor_name: str
    predictor_storage: dict[str, int]

    @property
    def cycles(self) -> int:
        return self.metrics.cycles

    @property
    def speedup(self) -> float:
        return 1.0


@dataclass(frozen=True)
class _OutstandingPrefetch:
    line: int
    ready_cycle: int
    destination: str


def _mark_evictions(
    evicted: Iterable[int],
    prefetched_resident: set[int],
    demand_resident: set[int],
    polluted_lines: set[int],
    metrics: SimulationMetrics,
) -> None:
    for line in evicted:
        if line in prefetched_resident:
            prefetched_resident.remove(line)
            metrics.unused_prefetches += 1
            polluted_lines.add(line)
        elif line in demand_resident:
            metrics.cache_evictions_caused_by_prefetching += 1
            demand_resident.remove(line)


def simulate(
    trace: Iterable[int],
    predictor: Predictor | None = None,
    config: SimulatorConfig = SimulatorConfig(),
    enable_prefetch: bool = True,
    learning: bool = False,
    initial_context: Iterable[int] | None = None,
) -> SimulationResult:
    """Replay raw byte addresses, while predictors see normalized line IDs.

    Prefetches are outstanding requests. A demand arriving before a request's
    ready cycle waits for the remaining time and is counted as late; a request
    ready before demand is inserted into the configured cache destination.
    """
    raw_addresses = list(trace)
    hierarchy = MemoryHierarchy(config.hierarchy)
    lines = [hierarchy.normalize(address) for address in raw_addresses]
    metrics = SimulationMetrics(total_accesses=0)
    pending: dict[int, _OutstandingPrefetch] = {}
    prefetched_resident: set[int] = set()
    demand_resident: set[int] = set()
    polluted_lines: set[int] = set()
    context: list[int] = list(initial_context or [])[-getattr(predictor, "context_depth", 1) :]
    cycle = 0
    predictor_name = predictor.name if predictor is not None else "None"
    predictor_storage = predictor.storage_stats() if predictor is not None else {"entries": 0, "nodes": 0, "edges": 0, "counters": 0, "weights": 0, "estimated_bytes": 0}

    # The control is computed with the same line abstraction and caches.
    baseline_hierarchy = MemoryHierarchy(config.hierarchy)
    for line in lines:
        if baseline_hierarchy.access(line).level == "DRAM":
            metrics.baseline_dram_accesses += 1

    def complete_ready() -> None:
        ready = [request for request in pending.values() if request.ready_cycle <= cycle]
        for request in ready:
            pending.pop(request.line, None)
            metrics.prefetches_completed += 1
            evicted = hierarchy.insert_prefetch(request.line, request.destination)
            prefetched_resident.add(request.line)
            _mark_evictions(evicted, prefetched_resident, demand_resident, polluted_lines, metrics)

    def prepare_predictions() -> list[Prediction]:
        nonlocal cycle
        if predictor is None:
            return []
        lookup_cost = config.predictor_lookup_cost if config.predictor_lookup_cost is not None else predictor.lookup_cost
        metrics.predictor_lookup_overhead += lookup_cost
        cycle += lookup_cost
        metrics.record_context_lookup(predictor.lookup_diagnostics(context))
        predictions = predictor.predict(context, config.top_k)
        if not enable_prefetch:
            return predictions
        for prediction in predictions:
            metrics.prefetch_requests += 1
            if prediction.address in pending or prediction.address in prefetched_resident or hierarchy.contains(prediction.address):
                metrics.duplicate_prefetches += 1
                continue
            if len(pending) >= config.max_outstanding_prefetches:
                metrics.dropped_prefetches += 1
                continue
            pending[prediction.address] = _OutstandingPrefetch(
                prediction.address,
                cycle + config.hierarchy.dram_latency,
                config.prefetch_destination,
            )
            metrics.prefetches_issued += 1
            metrics.bandwidth_bytes += config.address_bytes
            metrics.prefetch_overhead += config.effective_prefetch_issue_cost
            cycle += config.effective_prefetch_issue_cost
        return predictions

    predictions_for_current = prepare_predictions() if context else []

    for index, line in enumerate(lines):
        complete_ready()
        if predictions_for_current:
            metrics.prediction_attempts += 1
            if predictions_for_current[0].address == line:
                metrics.top1_correct += 1
            if any(prediction.address == line for prediction in predictions_for_current):
                metrics.topk_correct += 1
            else:
                metrics.incorrect_predictions += 1

        wait_for_prefetch = 0
        if line in pending:
            request = pending[line]
            wait_for_prefetch = max(0, request.ready_cycle - cycle)
            if wait_for_prefetch:
                metrics.late_prefetches += 1
                cycle += wait_for_prefetch
            complete_ready()

        result = hierarchy.access(line)
        metrics.total_accesses += 1
        metrics.raw_memory_cycles += result.latency
        cycle += result.latency
        if result.level == "L1":
            metrics.l1_hits += 1
        elif result.level == "L2":
            metrics.l2_hits += 1
        elif result.level == "L3":
            metrics.l3_hits += 1
        else:
            metrics.dram_accesses += 1
        _mark_evictions(result.evicted, prefetched_resident, demand_resident, polluted_lines, metrics)
        if line in polluted_lines and result.level == "DRAM":
            metrics.pollution_misses += 1
            polluted_lines.remove(line)
        demand_resident.add(line)

        if line in prefetched_resident:
            prefetched_resident.remove(line)
            metrics.useful_prefetches += 1
            demand_latency = result.latency + wait_for_prefetch
            metrics.latency_saved_by_useful_prefetches += max(0, config.hierarchy.dram_latency - demand_latency)

        if predictor is not None and learning:
            update_cost = config.predictor_update_cost if config.predictor_update_cost is not None else predictor.update_cost
            predictor.observe(line)
            metrics.predictor_update_overhead += update_cost
            cycle += update_cost
        context.append(line)
        if index + 1 < len(lines):
            predictions_for_current = prepare_predictions()

    # Any request/line that never serves demand is speculative waste.
    metrics.unused_prefetches += len(pending) + len(prefetched_resident)
    metrics.incorrect_prefetch_cost += (len(pending) + len(prefetched_resident)) * config.effective_prefetch_issue_cost
    metrics.cycles = cycle
    return SimulationResult(metrics, predictor_name, predictor_storage)
