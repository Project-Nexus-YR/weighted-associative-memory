"""Predictor interfaces and simple hardware-friendly predictor baselines."""

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

    def fit(self, sequence: Iterable[int]) -> "Predictor":
        return self

    def predict(self, context: Iterable[int], k: int = 1) -> list[Prediction]:
        return []

    def storage_stats(self) -> dict[str, int]:
        return {"nodes": 0, "edges": 0, "estimated_bytes": 0}


class WeightedTriePredictor(Predictor):
    name = "WeightedTrie"

    def __init__(self, context_depth: int = 2, strategy: str = "frequency", alpha: float = 0.25, threshold: float = 0.0):
        self.trie = WeightedTrie(context_depth=context_depth, strategy=strategy, alpha=alpha)
        self.threshold = threshold

    @property
    def context_depth(self) -> int:
        return self.trie.context_depth

    def fit(self, sequence: Iterable[int]) -> "WeightedTriePredictor":
        self.trie.fit(sequence)
        return self

    def predict(self, context: Iterable[int], k: int = 1) -> list[Prediction]:
        return [Prediction(address, weight) for address, weight in self.trie.predict(context, k, self.threshold)]

    def storage_stats(self) -> dict[str, int]:
        return self.trie.storage_stats()


class LastTransitionPredictor(Predictor):
    name = "Markov-1"

    def __init__(self, threshold: float = 0.0):
        self.counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.threshold = threshold

    def fit(self, sequence: Iterable[int]) -> "LastTransitionPredictor":
        iterator = iter(sequence)
        try:
            previous = next(iterator)
        except StopIteration:
            return self
        for current in iterator:
            self.counts[previous][current] += 1
            previous = current
        return self

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
        return {"nodes": len(self.counts), "edges": edges, "estimated_bytes": edges * 24}


class NextLinePredictor(Predictor):
    name = "NextLine"

    def __init__(self, offset: int = 1):
        self.offset = offset

    def predict(self, context: Iterable[int], k: int = 1) -> list[Prediction]:
        context_tuple = tuple(context)
        if not context_tuple or k < 1:
            return []
        return [Prediction(context_tuple[-1] + self.offset, 1.0)]
