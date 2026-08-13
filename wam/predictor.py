"""Common streaming predictor interface and comparable baselines."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .trie import WeightedTrie


@dataclass(frozen=True)
class Prediction:
    address: int
    weight: float


class Predictor:
    name = "Predictor"
    context_depth = 1
    lookup_cost = 1
    update_cost = 1

    def reset(self) -> "Predictor":
        return self

    def fit(self, sequence: Iterable[int]) -> "Predictor":
        self.reset()
        for address in sequence:
            self.observe(address)
        return self

    def observe(self, address: int) -> None:
        del address

    def predict(self, context: Iterable[int], k: int = 1) -> list[Prediction]:
        del context, k
        return []

    def storage_stats(self) -> dict[str, int]:
        return {"entries": 0, "nodes": 0, "edges": 0, "counters": 0, "weights": 0, "estimated_bytes": 0}


class WeightedTriePredictor(Predictor):
    name = "WeightedTrie"

    def __init__(self, context_depth: int = 2, strategy: str = "frequency", alpha: float = 0.25, threshold: float = 0.0):
        self.context_depth = context_depth
        self.strategy = strategy
        self.alpha = alpha
        self.threshold = threshold
        self.lookup_cost = max(1, context_depth)
        self.update_cost = max(1, context_depth)
        self.trie = WeightedTrie(context_depth=context_depth, strategy=strategy, alpha=alpha)
        self._history: list[int] = []

    def reset(self) -> "WeightedTriePredictor":
        self.trie = WeightedTrie(context_depth=self.context_depth, strategy=self.strategy, alpha=self.alpha)
        self._history = []
        return self

    def observe(self, address: int) -> None:
        if self._history:
            self.trie.update(self._history, address)
        self._history.append(address)

    def fit(self, sequence: Iterable[int]) -> "WeightedTriePredictor":
        return super().fit(sequence)  # type: ignore[return-value]

    def predict(self, context: Iterable[int], k: int = 1) -> list[Prediction]:
        return [Prediction(address, weight) for address, weight in self.trie.predict(context, k, self.threshold)]

    def storage_stats(self) -> dict[str, int]:
        return self.trie.storage_stats()


class LastTransitionPredictor(Predictor):
    name = "Markov-1"
    context_depth = 1
    lookup_cost = 1
    update_cost = 1

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
        self.reset()

    def reset(self) -> "LastTransitionPredictor":
        self.counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._last: int | None = None
        return self

    def observe(self, address: int) -> None:
        if self._last is not None:
            self.counts[self._last][address] += 1
        self._last = address

    def fit(self, sequence: Iterable[int]) -> "LastTransitionPredictor":
        return super().fit(sequence)  # type: ignore[return-value]

    def predict(self, context: Iterable[int], k: int = 1) -> list[Prediction]:
        context_tuple = tuple(context)
        if not context_tuple or k < 1:
            return []
        transitions = self.counts.get(context_tuple[-1], {})
        total = sum(transitions.values())
        ranked = [(address, count / total) for address, count in transitions.items() if total and count / total >= self.threshold]
        return [Prediction(address, weight) for address, weight in sorted(ranked, key=lambda item: (-item[1], item[0]))[:k]]

    def storage_stats(self) -> dict[str, int]:
        edges = sum(len(transitions) for transitions in self.counts.values())
        return {"entries": edges, "nodes": len(self.counts), "edges": edges, "counters": edges, "weights": edges, "estimated_bytes": edges * 24}


class NextLinePredictor(Predictor):
    name = "NextLine"
    context_depth = 1
    lookup_cost = 1
    update_cost = 0

    def __init__(self, offset: int = 1):
        self.offset = offset

    def predict(self, context: Iterable[int], k: int = 1) -> list[Prediction]:
        context_tuple = tuple(context)
        if not context_tuple or k < 1:
            return []
        return [Prediction(context_tuple[-1] + self.offset, 1.0)]


class StridePredictor(Predictor):
    """A small-confidence conventional stride prefetcher."""

    name = "Stride"
    context_depth = 2
    lookup_cost = 1
    update_cost = 1

    def __init__(self, confidence_threshold: int = 2):
        self.confidence_threshold = confidence_threshold
        self.reset()

    def reset(self) -> "StridePredictor":
        self._last: int | None = None
        self._stride: int | None = None
        self.confidence = 0
        return self

    def observe(self, address: int) -> None:
        if self._last is not None:
            stride = address - self._last
            if stride == self._stride:
                self.confidence += 1
            else:
                self._stride = stride
                self.confidence = 1
        self._last = address

    def predict(self, context: Iterable[int], k: int = 1) -> list[Prediction]:
        context_tuple = tuple(context)
        if not context_tuple or self._stride is None or self.confidence < self.confidence_threshold:
            return []
        return [Prediction(context_tuple[-1] + self._stride, 1.0)]

    def storage_stats(self) -> dict[str, int]:
        return {"entries": 1, "nodes": 1, "edges": 1, "counters": 1, "weights": 0, "estimated_bytes": 16}
