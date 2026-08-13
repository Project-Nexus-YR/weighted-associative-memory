"""Budgeted temporal baselines for the real-trace comparison.

These are deliberately small, common-interface approximations.  VLDP uses the
longest matching variable-length delta history, SPP recursively extends a
one-step delta path with cumulative confidence, and GMC selects among multiple
history orders.  They are not claims of bit-for-bit reproductions of the
original papers; the evaluator records this limitation in its report.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

from .horizon import HorizonPrediction
from .predictor import Prediction, Predictor, StridePredictor


def _sat(value: int, bits: int) -> int:
    return min((1 << bits) - 1, value + 1)


class DeltaContextPredictor(Predictor):
    """Fixed-depth delta-context table with a bounded approximate footprint."""

    def __init__(self, name: str = "DeltaContext", context_depth: int = 4, horizon: int = 1, budget_bytes: int = 8192, counter_bits: int = 8, longest_match: bool = False):
        self.name = name
        self.context_depth = context_depth
        self.horizon = horizon
        self.budget_bytes = budget_bytes
        self.counter_bits = counter_bits
        self.longest_match = longest_match
        self.lookup_cost = 2 if name != "VLDP" else max(1, context_depth // 2)
        self.update_cost = 1
        self.entry_limit = max(1, budget_bytes // 16)
        self.reset()

    def reset(self) -> "DeltaContextPredictor":
        self.counts: dict[tuple[int, ...], dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.order: deque[tuple[int, ...]] = deque()
        self._history: list[int] = []
        self.observations = 0
        return self

    def _key(self, values: list[int], length: int) -> tuple[int, ...]:
        deltas = [values[index] - values[index - 1] for index in range(1, len(values))]
        return tuple(deltas[-length:])

    def fit(self, sequence: Iterable[int]) -> "DeltaContextPredictor":
        values = list(sequence)
        self.reset()
        for position in range(max(0, len(values) - self.horizon)):
            if position < 1:
                continue
            target_delta = values[position + self.horizon] - values[position]
            lengths = range(1, min(self.context_depth, position) + 1) if not self.longest_match else (self.context_depth,)
            for length in lengths:
                key = self._key(values[: position + 1], length)
                if key not in self.counts and len(self.counts) >= self.entry_limit:
                    old = self.order.popleft()
                    self.counts.pop(old, None)
                if key not in self.counts:
                    self.order.append(key)
                self.counts[key][target_delta] = _sat(self.counts[key][target_delta], self.counter_bits)
                self.observations += 1
        return self

    def _predict_delta(self, context: Iterable[int], k: int = 1) -> list[Prediction]:
        values = list(context)
        if len(values) < 2:
            return []
        lengths = [min(self.context_depth, len(values) - 1)] if self.longest_match else range(min(self.context_depth, len(values) - 1), 0, -1)
        transitions: dict[int, int] = {}
        for length in lengths:
            transitions = self.counts.get(self._key(values, length), {})
            if transitions:
                break
        total = sum(transitions.values())
        if not total:
            return []
        ranked = sorted(((delta, count / total) for delta, count in transitions.items()), key=lambda item: (-item[1], item[0]))[:k]
        return [Prediction(delta, confidence) for delta, confidence in ranked]

    def predict_horizon(self, context: Iterable[int], k: int = 1) -> list[HorizonPrediction]:
        values = list(context)
        predictions = self._predict_delta(values, k)
        return [HorizonPrediction(values[-1] + prediction.address, self.horizon, prediction.weight) for prediction in predictions] if values else []

    def predict(self, context: Iterable[int], k: int = 1) -> list[Prediction]:
        values = list(context)
        return [Prediction(values[-1] + prediction.address, prediction.weight) for prediction in self._predict_delta(values, k)] if values else []

    def storage_stats(self) -> dict[str, int]:
        entries = sum(len(transitions) for transitions in self.counts.values())
        return {"entries": entries, "nodes": len(self.counts), "edges": entries, "counters": entries, "weights": entries, "estimated_bytes": min(self.budget_bytes, max(16, len(self.counts) * 16))}


class AddressContextPredictor(DeltaContextPredictor):
    """Bounded flat raw-address context table for equal-budget Markov-N."""

    def __init__(self, context_depth: int = 4, horizon: int = 16, budget_bytes: int = 8192):
        super().__init__("Markov-N", context_depth, horizon, budget_bytes)

    def _key(self, values: list[int], length: int) -> tuple[int, ...]:
        return tuple(values[-length:])


class SPPStylePredictor(DeltaContextPredictor):
    """Signature/path predictor proxy with recursive speculative extension."""

    def __init__(self, context_depth: int = 4, horizon: int = 16, budget_bytes: int = 8192):
        super().__init__("SPP", context_depth, 1, budget_bytes, counter_bits=8, longest_match=True)
        self.max_horizon = horizon
        self.lookup_cost = 3

    def predict_path(self, context: Iterable[int], max_horizon: int | None = None) -> list[HorizonPrediction]:
        history = list(context)
        cumulative = 1.0
        result: list[HorizonPrediction] = []
        for distance in range(1, (max_horizon or self.max_horizon) + 1):
            predictions = self._predict_delta(history, 1)
            if not predictions:
                break
            prediction = predictions[0]
            cumulative *= prediction.weight
            if cumulative < 0.05:
                break
            address = history[-1] + prediction.address
            result.append(HorizonPrediction(address, distance, cumulative))
            history.append(address)
        return result

    def predict_horizon(self, context: Iterable[int], k: int = 1) -> list[HorizonPrediction]:
        return [prediction for prediction in self.predict_path(context, self.max_horizon) if prediction.horizon == self.max_horizon][:k]


class GMCStylePredictor(DeltaContextPredictor):
    """Multi-order context table that chooses the longest supported order."""

    def __init__(self, horizon: int = 16, budget_bytes: int = 8192):
        super().__init__("GMC", 16, horizon, budget_bytes, counter_bits=8, longest_match=False)
        self.lookup_cost = 3


class HybridPredictor(Predictor):
    """Simple conventional-first arbitration between stride and WAM."""

    name = "Hybrid"

    def __init__(self, contextual: Predictor, stride_confidence: int = 2, confidence_threshold: float = 0.5):
        self.contextual = contextual
        self.stride = StridePredictor(stride_confidence)
        self.context_depth = getattr(contextual, "context_depth", 1)
        self.lookup_cost = max(getattr(contextual, "lookup_cost", 1), self.stride.lookup_cost)
        self.update_cost = max(getattr(contextual, "update_cost", 1), self.stride.update_cost)
        self.confidence_threshold = confidence_threshold

    def fit(self, sequence: Iterable[int]) -> "HybridPredictor":
        values = list(sequence)
        self.contextual.fit(values)
        self.stride.reset()
        for value in values:
            self.stride.observe(value)
        return self

    def predict_horizon(self, context: Iterable[int], k: int = 1) -> list[HorizonPrediction]:
        if hasattr(self.contextual, "predict_horizon"):
            predictions = self.contextual.predict_horizon(context, k)
            if predictions and predictions[0].confidence >= self.confidence_threshold:
                return predictions
        stride = self.stride.predict(context, k)
        horizon = getattr(self.contextual, "horizon", 1)
        return [HorizonPrediction(item.address, horizon, item.weight) for item in stride]

    def predict(self, context: Iterable[int], k: int = 1) -> list[Prediction]:
        return [Prediction(item.address, item.confidence) for item in self.predict_horizon(context, k)]

    def storage_stats(self) -> dict[str, int]:
        contextual = self.contextual.storage_stats()
        return {**contextual, "estimated_bytes": contextual.get("estimated_bytes", 0) + self.stride.storage_stats().get("estimated_bytes", 0)}


class NextLineHorizon(Predictor):
    name = "NextLine"
    context_depth = 1
    lookup_cost = 1
    update_cost = 0

    def __init__(self, horizon: int = 1):
        self.horizon = horizon

    def predict_horizon(self, context: Iterable[int], k: int = 1) -> list[HorizonPrediction]:
        values = list(context)
        return [HorizonPrediction(values[-1] + self.horizon, self.horizon, 1.0)] if values and k else []

    def predict(self, context: Iterable[int], k: int = 1) -> list[Prediction]:
        return [Prediction(item.address, item.confidence) for item in self.predict_horizon(context, k)]

    def storage_stats(self) -> dict[str, int]:
        return {"entries": 1, "nodes": 1, "edges": 1, "counters": 0, "weights": 0, "estimated_bytes": 8}


class StrideHorizon(Predictor):
    name = "Stride"
    context_depth = 2
    lookup_cost = 1
    update_cost = 1

    def __init__(self, horizon: int = 1, confidence_threshold: int = 2):
        self.horizon = horizon
        self.base = StridePredictor(confidence_threshold)

    def fit(self, sequence: Iterable[int]) -> "StrideHorizon":
        self.base.reset()
        for value in sequence:
            self.base.observe(value)
        return self

    def predict_horizon(self, context: Iterable[int], k: int = 1) -> list[HorizonPrediction]:
        values = list(context)
        predictions = self.base.predict(values, 1)
        return [HorizonPrediction(values[-1] + (prediction.address - values[-1]) * self.horizon, self.horizon, prediction.weight) for prediction in predictions] if values else []

    def predict(self, context: Iterable[int], k: int = 1) -> list[Prediction]:
        return [Prediction(item.address, item.confidence) for item in self.predict_horizon(context, k)]

    def storage_stats(self) -> dict[str, int]:
        return self.base.storage_stats()
