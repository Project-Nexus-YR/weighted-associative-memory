"""Bounded predictor models used by the hardware-feasibility experiment.

The classes in this module deliberately model architectural trade-offs rather
than claiming a gate-level implementation.  ``HashedContextPredictor`` keeps
the same horizon-conditioned prediction interface as the direct WAM model but
stores observations in a fixed number of hash buckets with saturating integer
counters.  This makes storage, collision, signature, and counter-width sweeps
reproducible without changing the learned workload or simulator.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .horizon import DirectHorizonWAM, HorizonPrediction
from .predictor import Prediction, Predictor


def _mix(values: Iterable[int], bits: int = 64) -> int:
    """Small deterministic xor/fold/multiply hash suitable for a hardware proxy."""
    mask = (1 << min(bits, 64)) - 1
    value = 0x9E3779B97F4A7C15 & mask
    for item in values:
        value ^= (int(item) + 0x9E3779B9 + ((value << 6) & mask) + (value >> 2)) & mask
        value = (value * 0xBF58476D1CE4E5B9) & mask
    value ^= value >> 29
    return value & mask


class IdealWAM(DirectHorizonWAM):
    """Zero-cost direct-WAM upper bound; intentionally not implementable hardware."""

    name = "IdealWAM"

    def __init__(self, context_depth: int = 16, horizon: int = 16):
        super().__init__(context_depth=context_depth, horizon=horizon)
        self.lookup_cost = 0
        self.update_cost = 0


class HashedContextPredictor(Predictor):
    """Fixed-size context table with quantized counters and deterministic aliasing."""

    name = "HashedContext"

    def __init__(
        self,
        context_depth: int = 4,
        horizon: int = 1,
        table_size: int = 1024,
        counter_bits: int = 8,
        signature_bits: int = 64,
        threshold: float = 0.0,
        entry_bytes: int = 16,
    ) -> None:
        if context_depth < 1 or horizon < 1 or table_size < 1:
            raise ValueError("context_depth, horizon, and table_size must be positive")
        if counter_bits not in {2, 4, 8, 12}:
            raise ValueError("counter_bits must be one of 2, 4, 8, or 12")
        if signature_bits not in {8, 12, 16, 32, 64}:
            raise ValueError("signature_bits must be one of 8, 12, 16, 32, or 64")
        self.context_depth = context_depth
        self.horizon = horizon
        self.table_size = table_size
        self.counter_bits = counter_bits
        self.signature_bits = signature_bits
        self.threshold = threshold
        self.entry_bytes = entry_bytes
        self.lookup_cost = 3
        self.update_cost = 1
        self.reset()

    def reset(self) -> "HashedContextPredictor":
        self.counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.contexts_by_bucket: dict[int, set[tuple[int, ...]]] = defaultdict(set)
        self.observations = 0
        self.collision_events = 0
        self.aliasing_events = 0
        return self

    def _bucket(self, context: Iterable[int]) -> tuple[int, tuple[int, ...]]:
        key = tuple(context)[-self.context_depth :]
        signature = _mix(key, self.signature_bits)
        return signature % self.table_size, key

    def fit(self, sequence: Iterable[int]) -> "HashedContextPredictor":
        values = list(sequence)
        self.reset()
        for position in range(max(0, len(values) - self.horizon)):
            context = values[max(0, position - self.context_depth + 1) : position + 1]
            bucket, key = self._bucket(context)
            known = self.contexts_by_bucket[bucket]
            if known and key not in known:
                self.collision_events += 1
                self.aliasing_events += 1
            known.add(key)
            self.counts[bucket][values[position + self.horizon]] += 1
            self.observations += 1
        return self

    def _quantized(self, transitions: dict[int, int]) -> list[tuple[int, float]]:
        if not transitions:
            return []
        maximum = max(transitions.values())
        levels = (1 << self.counter_bits) - 1
        quantized = {address: max(1, round(count / maximum * levels)) for address, count in transitions.items()}
        total = sum(quantized.values())
        return [(address, value / total) for address, value in quantized.items() if value / total >= self.threshold]

    def predict_horizon(self, context: Iterable[int], k: int = 1) -> list[HorizonPrediction]:
        bucket, _ = self._bucket(context)
        ranked = sorted(self._quantized(self.counts.get(bucket, {})), key=lambda item: (-item[1], item[0]))[:k]
        return [HorizonPrediction(address, self.horizon, confidence) for address, confidence in ranked]

    def predict(self, context: Iterable[int], k: int = 1) -> list[Prediction]:
        return [Prediction(item.address, item.confidence) for item in self.predict_horizon(context, k)]

    def storage_stats(self) -> dict[str, int]:
        entries = sum(len(values) for values in self.counts.values())
        context_register_bytes = max(1, self.context_depth * self.signature_bits // 8)
        return {
            "entries": entries,
            "nodes": self.table_size,
            "edges": entries,
            "counters": entries,
            "weights": entries,
            "estimated_bytes": self.table_size * self.entry_bytes + context_register_bytes,
        }

    def lookup_diagnostics(self, context: Iterable[int]) -> dict[str, float | int | bool | None]:
        bucket, key = self._bucket(context)
        transitions = self.counts.get(bucket, {})
        return {
            "requested_depth": self.context_depth,
            "requested_available_depth": min(self.context_depth, len(tuple(context))),
            "matched_depth": self.context_depth if transitions else 0,
            "fallback": False,
            "unseen": not bool(transitions),
            "observations": sum(transitions.values()),
            "entropy": None,
            "bucket": bucket,
            "aliased_context": key not in self.contexts_by_bucket.get(bucket, set()),
        }

    @property
    def collision_rate(self) -> float:
        return self.collision_events / max(1, self.observations)

    @property
    def aliasing_rate(self) -> float:
        return self.aliasing_events / max(1, len(self.contexts_by_bucket))


@dataclass(frozen=True)
class HardwareModel:
    """Abstract timing/area model used in the comparison matrix."""

    name: str
    lookup_latency: int
    issue_interval: int
    overlap_cycles: int
    update_latency: int
    read_ports: int
    write_ports: int
    candidate_cost: int
    energy_read: float
    energy_write: float
    energy_compare: float
    energy_hash: float
    notes: str


def hardware_models(context_depth: int = 16) -> tuple[HardwareModel, ...]:
    return (
        HardwareModel("SerialTrie", context_depth + 1, context_depth + 1, 0, context_depth, 1, 1, 1, 1.0, 1.2, 0.1, 0.0, "one traversal step plus winner selection; no overlap"),
        HardwareModel("PipelinedTrie", context_depth + 1, 1, context_depth, context_depth, 1, 1, 1, 1.0, 1.2, 0.1, 0.0, "depth pipeline; latency remains, one lookup can issue each cycle"),
        HardwareModel("ParallelTrie", 3, 1, 2, 2, 2, 1, 1, 1.0, 1.2, 0.1, 0.0, "parallel SRAM level reads and a small comparator tree"),
        HardwareModel("HashedContext", 3, 1, 2, 1, 2, 1, 1, 1.0, 1.2, 0.1, 0.2, "hash, SRAM read, and winner selection"),
        HardwareModel("CAM-like", 2, 1, 1, 1, 2, 1, 2, 1.5, 1.8, 0.2, 0.2, "small associative table; high area/energy proxy"),
    )
