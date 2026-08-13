"""Primary metrics emitted by the research simulator."""

from __future__ import annotations

from dataclasses import dataclass


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

    def speedup_vs(self, baseline_cycles: int) -> float:
        return baseline_cycles / self.cycles if self.cycles else 0.0
