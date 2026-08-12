"""Run traces through the hierarchy with optional predictive prefetching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .hierarchy import HierarchyConfig, MemoryHierarchy
from .metrics import SimulationMetrics
from .predictor import Prediction, Predictor


@dataclass(frozen=True)
class SimulatorConfig:
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    prefetch_cost: int = 8
    prefetch_destination: str = "L1"
    address_bytes: int = 8
    top_k: int = 3


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


def simulate(trace: Iterable[int], predictor: Predictor | None = None, config: SimulatorConfig = SimulatorConfig(), enable_prefetch: bool = True) -> SimulationResult:
    """Simulate a trace; predictions are issued immediately before each access."""
    addresses = list(trace)
    hierarchy = MemoryHierarchy(config.hierarchy)
    metrics = SimulationMetrics()
    baseline_hierarchy = MemoryHierarchy(config.hierarchy)
    for address in addresses:
        if baseline_hierarchy.access(address).level == "DRAM":
            metrics.baseline_dram_accesses += 1
    context: list[int] = []
    pending_prefetches: set[int] = set()
    predictor_name = predictor.name if predictor is not None else "None"

    for current in addresses:
        predictions = predictor.predict(context, config.top_k) if predictor is not None else []
        if predictions:
            metrics.prediction_attempts += 1
            if predictions[0].address == current:
                metrics.top1_correct += 1
            if any(prediction.address == current for prediction in predictions):
                metrics.topk_correct += 1
            else:
                metrics.incorrect_predictions += 1

        if enable_prefetch and predictions:
            for prediction in predictions:
                if prediction.address in pending_prefetches or hierarchy.contains(prediction.address):
                    metrics.duplicate_prefetches += 1
                    continue
                inserted, evicted = hierarchy.prefetch(prediction.address, config.prefetch_destination)
                if inserted:
                    metrics.prefetches_issued += 1
                    metrics.bandwidth_bytes += config.address_bytes
                    metrics.cycles += config.prefetch_cost
                    pending_prefetches.add(prediction.address)
                    if evicted is not None and evicted in pending_prefetches:
                        pending_prefetches.remove(evicted)
                        metrics.unused_prefetches += 1
                        metrics.incorrect_prefetch_cost += config.prefetch_cost

        if current in pending_prefetches and not hierarchy.contains(current):
            # It was evicted before use; this prediction consumed bandwidth but
            # did not help the demand access.
            pending_prefetches.remove(current)
            metrics.unused_prefetches += 1
            metrics.incorrect_prefetch_cost += config.prefetch_cost

        result = hierarchy.access(current)
        metrics.total_accesses += 1
        metrics.cycles += result.latency
        if result.level == "L1":
            metrics.l1_hits += 1
        elif result.level == "L2":
            metrics.l2_hits += 1
        else:
            metrics.dram_accesses += 1

        if result.evicted_by_fill is not None and result.evicted_by_fill in pending_prefetches:
            pending_prefetches.remove(result.evicted_by_fill)
            metrics.unused_prefetches += 1
            metrics.incorrect_prefetch_cost += config.prefetch_cost

        if current in pending_prefetches:
            pending_prefetches.remove(current)
            metrics.useful_prefetches += 1
            metrics.latency_saved_by_useful_prefetches += max(0, config.hierarchy.dram_latency - result.latency)
        context.append(current)

    metrics.unused_prefetches += len(pending_prefetches)
    metrics.incorrect_prefetch_cost += len(pending_prefetches) * config.prefetch_cost
    metrics.cache_evictions_caused_by_prefetching = hierarchy.prefetch_evictions
    return SimulationResult(metrics, predictor_name, predictor.storage_stats() if predictor else {"nodes": 0, "edges": 0, "estimated_bytes": 0})
