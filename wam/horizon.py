"""Prediction-horizon models and latency-aware prefetch experiments.

This module intentionally sits beside the original simulator. It reuses the
same ``MemoryHierarchy`` and cache-line configuration while allowing a
prediction to target an address several accesses in the future.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .hierarchy import HierarchyConfig, MemoryHierarchy
from .predictor import Prediction, Predictor, WeightedTriePredictor
from .trie import WeightedTrie


@dataclass(frozen=True)
class HorizonPrediction:
    address: int
    horizon: int
    confidence: float


class DirectHorizonWAM(Predictor):
    """Learn a separate weighted context table for one target horizon."""

    name = "DirectWAM"

    def __init__(self, context_depth: int = 4, horizon: int = 1, threshold: float = 0.0):
        self.context_depth = context_depth
        self.horizon = horizon
        self.threshold = threshold
        self.lookup_cost = max(1, context_depth)
        self.update_cost = max(1, context_depth)
        self.reset()

    def reset(self) -> "DirectHorizonWAM":
        self.trie = WeightedTrie(context_depth=self.context_depth)
        return self

    def fit(self, sequence: Iterable[int]) -> "DirectHorizonWAM":
        values = list(sequence)
        self.reset()
        for position in range(max(0, len(values) - self.horizon)):
            context = values[max(0, position - self.context_depth + 1) : position + 1]
            self.trie.update(context, values[position + self.horizon])
        return self

    def predict_horizon(self, context: Iterable[int], k: int = 1) -> list[HorizonPrediction]:
        return [HorizonPrediction(address, self.horizon, weight) for address, weight in self.trie.predict(context, k, self.threshold)]

    def predict(self, context: Iterable[int], k: int = 1) -> list[Prediction]:
        return [Prediction(item.address, item.confidence) for item in self.predict_horizon(context, k)]

    def storage_stats(self) -> dict[str, int]:
        return self.trie.storage_stats()

    def lookup_diagnostics(self, context: Iterable[int]) -> dict[str, float | int | bool | None]:
        return self.trie.lookup_diagnostics(context)


class DirectMarkovHorizon(Predictor):
    """Flat exact-context table for the same horizon as direct WAM."""

    name = "DirectMarkov"

    def __init__(self, context_depth: int = 4, horizon: int = 1, threshold: float = 0.0):
        self.context_depth = context_depth
        self.horizon = horizon
        self.threshold = threshold
        self.lookup_cost = max(1, context_depth)
        self.update_cost = max(1, context_depth)
        self.reset()

    def reset(self) -> "DirectMarkovHorizon":
        self.counts: dict[tuple[int, ...], dict[int, int]] = defaultdict(lambda: defaultdict(int))
        return self

    def fit(self, sequence: Iterable[int]) -> "DirectMarkovHorizon":
        values = list(sequence)
        self.reset()
        for position in range(max(0, len(values) - self.horizon)):
            start = max(0, position - self.context_depth + 1)
            context = tuple(values[start : position + 1])
            self.counts[context][values[position + self.horizon]] += 1
        return self

    def predict_horizon(self, context: Iterable[int], k: int = 1) -> list[HorizonPrediction]:
        context_tuple = tuple(context)
        key = context_tuple[-self.context_depth :]
        transitions = self.counts.get(key, {})
        total = sum(transitions.values())
        ranked = [(address, count / total) for address, count in transitions.items() if total and count / total >= self.threshold]
        return [HorizonPrediction(address, self.horizon, weight) for address, weight in sorted(ranked, key=lambda item: (-item[1], item[0]))[:k]]

    def predict(self, context: Iterable[int], k: int = 1) -> list[Prediction]:
        return [Prediction(item.address, item.confidence) for item in self.predict_horizon(context, k)]

    def storage_stats(self) -> dict[str, int]:
        edges = sum(len(transitions) for transitions in self.counts.values())
        return {"entries": edges, "nodes": len(self.counts), "edges": edges, "counters": edges, "weights": edges, "estimated_bytes": edges * 24}

    def lookup_diagnostics(self, context: Iterable[int]) -> dict[str, float | int | bool | None]:
        context_tuple = tuple(context)
        key = context_tuple[-self.context_depth :]
        transitions = self.counts.get(key, {})
        return {"requested_depth": self.context_depth, "requested_available_depth": min(self.context_depth, len(context_tuple)), "matched_depth": self.context_depth if transitions else 0, "fallback": False, "unseen": not bool(transitions), "observations": sum(transitions.values()), "entropy": None}


class RecursiveWAM(Predictor):
    """Repeated one-step traversal with cumulative path confidence."""

    name = "RecursiveWAM"

    def __init__(self, context_depth: int = 4, max_horizon: int = 1, threshold: float = 0.0, cumulative_threshold: float = 0.0, speculative_width: int | None = None):
        self.context_depth = context_depth
        self.max_horizon = max_horizon
        self.threshold = threshold
        self.cumulative_threshold = cumulative_threshold
        self.speculative_width = speculative_width if speculative_width is not None else max_horizon
        self.lookup_cost = max(1, context_depth)
        self.update_cost = max(1, context_depth)
        self.base = WeightedTriePredictor(context_depth=context_depth, threshold=threshold)

    def reset(self) -> "RecursiveWAM":
        self.base.reset()
        return self

    def fit(self, sequence: Iterable[int]) -> "RecursiveWAM":
        self.base.fit(sequence)
        return self

    def predict_path(self, context: Iterable[int], max_horizon: int | None = None) -> list[HorizonPrediction]:
        history = list(context)[-self.context_depth :]
        cumulative = 1.0
        path: list[HorizonPrediction] = []
        for distance in range(1, (max_horizon or self.max_horizon) + 1):
            predictions = self.base.predict(history, k=1)
            if not predictions:
                break
            prediction = predictions[0]
            cumulative *= prediction.weight
            if cumulative < self.cumulative_threshold:
                break
            path.append(HorizonPrediction(prediction.address, distance, cumulative))
            history.append(prediction.address)
        return path[: self.speculative_width]

    def predict_horizon(self, context: Iterable[int], horizon: int) -> list[HorizonPrediction]:
        return [item for item in self.predict_path(context, horizon) if item.horizon == horizon]

    def predict(self, context: Iterable[int], k: int = 1) -> list[Prediction]:
        path = self.predict_path(context, self.max_horizon)
        return [Prediction(item.address, item.confidence) for item in path[:k]]

    def storage_stats(self) -> dict[str, int]:
        return self.base.storage_stats()

    def lookup_diagnostics(self, context: Iterable[int]) -> dict[str, float | int | bool | None]:
        return self.base.lookup_diagnostics(context)


class OracleHorizon:
    name = "Oracle"
    context_depth = 1
    lookup_cost = 0
    update_cost = 0

    def predict_horizon(self, index: int, trace: list[int], horizon: int) -> list[HorizonPrediction]:
        target = index + horizon
        if target >= len(trace):
            return []
        return [HorizonPrediction(trace[target], horizon, 1.0)]

    def storage_stats(self) -> dict[str, int]:
        return {"entries": 0, "nodes": 0, "edges": 0, "counters": 0, "weights": 0, "estimated_bytes": 0}


class NoHorizonPredictor:
    name = "None"
    context_depth = 1
    lookup_cost = 0
    update_cost = 0

    def storage_stats(self) -> dict[str, int]:
        return {"entries": 0, "nodes": 0, "edges": 0, "counters": 0, "weights": 0, "estimated_bytes": 0}


@dataclass(frozen=True)
class HorizonConfig:
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    prefetch_issue_cost: int = 1
    prefetch_destination: str = "L1"
    address_bytes: int = 8
    top_k: int = 3
    max_outstanding_prefetches: int = 8
    compute_cycles_between_accesses: int = 0
    predictor_lookup_latency: int | None = None
    predictor_issue_interval: int = 1
    predictor_overlap_cycles: int = 0
    predictor_parallel: bool = False
    predictor_update_latency: int | None = None
    predictor_update_interval: int = 1
    deferred_updates: bool = False
    update_batch_size: int = 1
    predictor_queue_depth: int = 16
    read_ports: int = 1
    write_ports: int = 1


@dataclass
class HorizonMetrics:
    total_accesses: int = 0
    l1_hits: int = 0
    l2_hits: int = 0
    l3_hits: int = 0
    dram_accesses: int = 0
    raw_memory_cycles: int = 0
    predictor_overhead: int = 0
    prefetch_overhead: int = 0
    cycles: int = 0
    prediction_attempts: int = 0
    correct_predictions: int = 0
    topk_correct: int = 0
    wrong_predictions: int = 0
    prefetch_requests: int = 0
    prefetches_issued: int = 0
    prefetches_completed: int = 0
    dropped_prefetches: int = 0
    useful_prefetches: int = 0
    late_prefetches: int = 0
    unused_prefetches: int = 0
    duplicate_prefetches: int = 0
    pollution_misses: int = 0
    bandwidth_bytes: int = 0
    cycles_hidden: int = 0
    fully_hidden_misses: int = 0
    partially_hidden_misses: int = 0
    unhidden_misses: int = 0
    queue_occupancy_sum: int = 0
    queue_occupancy_samples: int = 0
    lead_times: list[int] = field(default_factory=list)
    slacks: list[int] = field(default_factory=list)
    wrong_prediction_cost: int = 0
    lateness_cost: int = 0
    bandwidth_cost: int = 0
    pollution_cost: int = 0
    predictor_queue_stalls: int = 0
    predictor_queue_wait: int = 0
    max_predictor_queue_wait: int = 0
    update_queue_stalls: int = 0
    port_stalls: int = 0
    dropped_predictions: int = 0
    update_count: int = 0
    energy_proxy: float = 0.0

    @property
    def top1_accuracy(self) -> float:
        return self.correct_predictions / self.prediction_attempts if self.prediction_attempts else 0.0

    @property
    def topk_accuracy(self) -> float:
        return self.topk_correct / self.prediction_attempts if self.prediction_attempts else 0.0

    @property
    def prefetch_precision(self) -> float:
        return self.useful_prefetches / self.prefetches_issued if self.prefetches_issued else 0.0

    @property
    def late_prefetch_rate(self) -> float:
        return self.late_prefetches / self.useful_prefetches if self.useful_prefetches else 0.0

    @property
    def mean_lead_time(self) -> float:
        return sum(self.lead_times) / len(self.lead_times) if self.lead_times else 0.0

    @property
    def median_lead_time(self) -> float:
        return sorted(self.lead_times)[len(self.lead_times) // 2] if self.lead_times else 0.0

    @property
    def mean_slack(self) -> float:
        return sum(self.slacks) / len(self.slacks) if self.slacks else 0.0

    @property
    def median_slack(self) -> float:
        return sorted(self.slacks)[len(self.slacks) // 2] if self.slacks else 0.0

    @property
    def bandwidth_utilization(self) -> float:
        return self.prefetches_issued / max(1, self.prefetches_issued + self.dropped_prefetches)

    @property
    def average_predictor_queue_wait(self) -> float:
        return self.predictor_queue_wait / self.predictor_queue_stalls if self.predictor_queue_stalls else 0.0


@dataclass
class HorizonResult:
    metrics: HorizonMetrics
    predictor_name: str
    predictor_storage: dict[str, int]

    @property
    def cycles(self) -> int:
        return self.metrics.cycles


@dataclass(frozen=True)
class _Request:
    line: int
    issue_cycle: int
    ready_cycle: int
    destination: str
    target_index: int


def simulate_horizon(trace: Iterable[int], predictor, horizon: int, config: HorizonConfig = HorizonConfig(), enable_prefetch: bool = True, initial_context: Iterable[int] = ()) -> HorizonResult:
    """Replay a byte-address trace with future-target prefetch requests."""
    raw = list(trace)
    hierarchy = MemoryHierarchy(config.hierarchy)
    lines = [hierarchy.normalize(address) for address in raw]
    metrics = HorizonMetrics()
    cycle = 0
    context = list(initial_context)[-getattr(predictor, "context_depth", 1) :]
    pending: dict[int, _Request] = {}
    prefetched: dict[int, _Request] = {}
    demand_lines: set[int] = set()
    polluted: set[int] = set()
    next_lookup_issue = 0
    next_update_issue = 0
    buffered_updates = 0

    def mark_evictions(evicted: Iterable[int]) -> None:
        for line in evicted:
            if line in prefetched:
                prefetched.pop(line, None)
                metrics.unused_prefetches += 1
                metrics.pollution_cost += config.hierarchy.dram_latency
                polluted.add(line)
            elif line in demand_lines:
                metrics.pollution_cost += config.hierarchy.dram_latency
                demand_lines.discard(line)

    def complete_ready() -> None:
        ready = [request for request in pending.values() if request.ready_cycle <= cycle]
        for request in ready:
            pending.pop(request.line, None)
            metrics.prefetches_completed += 1
            evicted = hierarchy.insert_prefetch(request.line, request.destination)
            prefetched[request.line] = request
            mark_evictions(evicted)

    def predictions_for(index: int) -> list[HorizonPrediction]:
        if isinstance(predictor, OracleHorizon):
            return predictor.predict_horizon(index, lines, horizon)
        if hasattr(predictor, "predict_horizon"):
            if isinstance(predictor, RecursiveWAM):
                return predictor.predict_horizon(context, horizon)
            return predictor.predict_horizon(context, config.top_k)
        return []

    for index, line in enumerate(lines):
        # Requests issued after the previous demand may complete before this
        # demand. A demand can wait for the remaining request time, which is
        # the partial-latency-hiding case.
        complete_ready()
        demand_cycle = cycle
        wait = 0
        request = pending.get(line)
        if request is None:
            request = prefetched.get(line)
        if request is not None:
            lead = demand_cycle - request.issue_cycle
            slack = demand_cycle - request.ready_cycle
            metrics.lead_times.append(lead)
            metrics.slacks.append(slack)
            if slack < 0:
                metrics.late_prefetches += 1
                wait = -slack
                metrics.lateness_cost += wait
                cycle += wait
                complete_ready()
            else:
                complete_ready()

        was_cached = hierarchy.contains(line)
        served_by_prefetch = line in prefetched
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
        mark_evictions(result.evicted)
        if line in prefetched:
            prefetched.pop(line, None)
            metrics.useful_prefetches += 1
            if served_by_prefetch:
                hidden = max(0, config.hierarchy.dram_latency - wait - result.latency)
                metrics.cycles_hidden += hidden
                if wait == 0:
                    metrics.fully_hidden_misses += 1
                elif hidden > 0:
                    metrics.partially_hidden_misses += 1
                else:
                    metrics.unhidden_misses += 1
        elif line in polluted:
            polluted.remove(line)
            metrics.pollution_misses += 1
            metrics.unhidden_misses += 1
        elif result.level == "DRAM":
            metrics.unhidden_misses += 1
        demand_lines.add(line)
        context.append(line)
        update_latency = config.predictor_update_latency if config.predictor_update_latency is not None else getattr(predictor, "update_cost", 0)
        if not config.deferred_updates:
            update_start = cycle
            if update_start < next_update_issue:
                wait = next_update_issue - update_start
                metrics.update_queue_stalls += 1
                metrics.port_stalls += wait if config.write_ports < 1 else 0
                cycle += wait
                update_start = cycle
            cycle += update_latency
            next_update_issue = update_start + max(1, config.predictor_update_interval)
            metrics.predictor_overhead += update_latency
            metrics.update_count += 1
        else:
            buffered_updates += 1
            metrics.update_count += 1
            if buffered_updates >= max(1, config.update_batch_size):
                cycle += update_latency
                metrics.predictor_overhead += update_latency
                buffered_updates = 0

        # The current address is now part of the available history. Direct,
        # recursive, and oracle predictions target index+h from this point.
        predictions = predictions_for(index)
        lookup_latency = config.predictor_lookup_latency if config.predictor_lookup_latency is not None else getattr(predictor, "lookup_cost", 0)
        pipeline_occupancy = max(1, (lookup_latency + max(1, config.predictor_issue_interval) - 1) // max(1, config.predictor_issue_interval))
        if config.predictor_parallel and pipeline_occupancy > max(1, config.predictor_queue_depth):
            metrics.dropped_predictions += 1
            predictions = []
        lookup_start = cycle
        if lookup_start < next_lookup_issue:
            wait = next_lookup_issue - lookup_start
            metrics.predictor_queue_stalls += 1
            metrics.predictor_queue_wait += wait
            metrics.max_predictor_queue_wait = max(metrics.max_predictor_queue_wait, wait)
            cycle += wait
            lookup_start = cycle
        effective_lookup = max(0, lookup_latency - config.predictor_overlap_cycles) if config.predictor_parallel else lookup_latency
        cycle += effective_lookup
        next_lookup_issue = lookup_start + max(1, config.predictor_issue_interval)
        metrics.predictor_overhead += effective_lookup
        metrics.energy_proxy += 1.0 + 0.1 * getattr(predictor, "context_depth", 1)
        valid_predictions = []
        for prediction in predictions:
            target_index = index + prediction.horizon
            if target_index >= len(lines):
                continue
            valid_predictions.append(prediction)
            metrics.prediction_attempts += 1
            actual = lines[target_index]
            if prediction.address == actual:
                metrics.correct_predictions += 1
            else:
                metrics.wrong_predictions += 1
                metrics.wrong_prediction_cost += config.hierarchy.dram_latency
        if valid_predictions:
            metrics.topk_correct += int(any(prediction.address == lines[index + prediction.horizon] for prediction in valid_predictions[: config.top_k]))
        if enable_prefetch:
            for prediction in valid_predictions:
                metrics.prefetch_requests += 1
                target_index = index + prediction.horizon
                if prediction.address in pending or prediction.address in prefetched or hierarchy.contains(prediction.address):
                    metrics.duplicate_prefetches += 1
                    continue
                if len(pending) >= config.max_outstanding_prefetches:
                    metrics.dropped_prefetches += 1
                    metrics.bandwidth_cost += config.hierarchy.dram_latency
                    continue
                request = _Request(prediction.address, cycle, cycle + config.hierarchy.dram_latency, config.prefetch_destination, target_index)
                pending[prediction.address] = request
                metrics.prefetches_issued += 1
                metrics.bandwidth_bytes += config.address_bytes
                metrics.prefetch_overhead += config.prefetch_issue_cost
                cycle += config.prefetch_issue_cost
        read_port_stall = max(0, len(valid_predictions) - max(1, config.read_ports))
        metrics.port_stalls += read_port_stall
        cycle += read_port_stall
        cycle += config.compute_cycles_between_accesses

    if config.deferred_updates and buffered_updates:
        update_latency = config.predictor_update_latency if config.predictor_update_latency is not None else getattr(predictor, "update_cost", 0)
        cycle += update_latency
        metrics.predictor_overhead += update_latency

    metrics.unused_prefetches += len(pending) + len(prefetched)
    metrics.cycles = cycle
    return HorizonResult(metrics, getattr(predictor, "name", type(predictor).__name__), predictor.storage_stats())
