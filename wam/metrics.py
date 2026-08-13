"""Primary metrics emitted by the research simulator."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SimulationMetrics:
    total_accesses: int = 0
    l1_hits: int = 0
    l2_hits: int = 0
    l3_hits: int = 0
    dram_accesses: int = 0
    raw_memory_cycles: int = 0
    predictor_lookup_overhead: int = 0
    predictor_update_overhead: int = 0
    prefetch_overhead: int = 0
    cycles: int = 0
    prediction_attempts: int = 0
    top1_correct: int = 0
    topk_correct: int = 0
    prefetch_requests: int = 0
    prefetches_issued: int = 0
    prefetches_completed: int = 0
    dropped_prefetches: int = 0
    useful_prefetches: int = 0
    late_prefetches: int = 0
    unused_prefetches: int = 0
    duplicate_prefetches: int = 0
    incorrect_predictions: int = 0
    bandwidth_bytes: int = 0
    cache_evictions_caused_by_prefetching: int = 0
    pollution_misses: int = 0
    latency_saved_by_useful_prefetches: int = 0
    incorrect_prefetch_cost: int = 0
    baseline_dram_accesses: int = 0
    context_lookups: int = 0
    exact_context_hits: int = 0
    fallback_count: int = 0
    unseen_context_count: int = 0
    matched_depth_histogram: dict[int, int] = field(default_factory=dict)
    context_observation_sum: int = 0
    context_entropy_sum: float = 0.0
    context_entropy_observations: int = 0

    @property
    def l1_hit_rate(self) -> float:
        return self.l1_hits / self.total_accesses if self.total_accesses else 0.0

    @property
    def l2_hit_rate(self) -> float:
        return self.l2_hits / self.total_accesses if self.total_accesses else 0.0

    @property
    def l3_hit_rate(self) -> float:
        return self.l3_hits / self.total_accesses if self.total_accesses else 0.0

    @property
    def dram_access_rate(self) -> float:
        return self.dram_accesses / self.total_accesses if self.total_accesses else 0.0

    @property
    def average_memory_latency(self) -> float:
        return self.raw_memory_cycles / self.total_accesses if self.total_accesses else 0.0

    @property
    def average_access_latency(self) -> float:
        return self.cycles / self.total_accesses if self.total_accesses else 0.0

    @property
    def prediction_accuracy(self) -> float:
        return self.top1_accuracy

    @property
    def top1_accuracy(self) -> float:
        return self.top1_correct / self.prediction_attempts if self.prediction_attempts else 0.0

    @property
    def topk_accuracy(self) -> float:
        return self.topk_correct / self.prediction_attempts if self.prediction_attempts else 0.0

    @property
    def prefetch_precision(self) -> float:
        return self.useful_prefetches / self.prefetches_issued if self.prefetches_issued else 0.0

    @property
    def prefetch_coverage(self) -> float:
        denominator = self.baseline_dram_accesses or self.dram_accesses
        return self.useful_prefetches / denominator if denominator else 0.0

    @property
    def net_latency_benefit(self) -> int:
        return self.latency_saved_by_useful_prefetches - self.incorrect_prefetch_cost

    @property
    def context_reuse_ratio(self) -> float:
        return self.exact_context_hits / self.context_lookups if self.context_lookups else 0.0

    @property
    def mean_context_observations(self) -> float:
        return self.context_observation_sum / self.context_lookups if self.context_lookups else 0.0

    @property
    def mean_context_entropy(self) -> float:
        return self.context_entropy_sum / self.context_entropy_observations if self.context_entropy_observations else 0.0

    def record_context_lookup(self, diagnostics: dict[str, float | int | bool | None]) -> None:
        self.context_lookups += 1
        requested = int(diagnostics.get("requested_available_depth", 0) or 0)
        matched = int(diagnostics.get("matched_depth", 0) or 0)
        self.matched_depth_histogram[matched] = self.matched_depth_histogram.get(matched, 0) + 1
        if matched and matched == requested:
            self.exact_context_hits += 1
        if bool(diagnostics.get("fallback")):
            self.fallback_count += 1
        if bool(diagnostics.get("unseen")):
            self.unseen_context_count += 1
        self.context_observation_sum += int(diagnostics.get("observations", 0) or 0)
        entropy = diagnostics.get("entropy")
        if entropy is not None:
            self.context_entropy_sum += float(entropy)
            self.context_entropy_observations += 1

    def speedup_vs(self, baseline_cycles: int) -> float:
        return baseline_cycles / self.cycles if self.cycles else 0.0
