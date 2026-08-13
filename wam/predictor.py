"""Common streaming predictor interface and comparable baselines."""

from __future__ import annotations

from collections import defaultdict
import math
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

    def lookup_diagnostics(self, context: Iterable[int]) -> dict[str, float | int | bool | None]:
        return {
            "requested_depth": self.context_depth,
            "requested_available_depth": min(self.context_depth, len(tuple(context))),
            "matched_depth": 0,
            "fallback": False,
            "unseen": True,
            "observations": 0,
            "entropy": None,
        }


class WeightedTriePredictor(Predictor):
    name = "WeightedTrie"

    def __init__(self, context_depth: int = 2, strategy: str = "frequency", alpha: float = 0.25, threshold: float = 0.0, minimum_observations: int = 1, support_k: float = 0.0, entropy_threshold: float | None = None, exact_only: bool = False):
        self.context_depth = context_depth
        self.strategy = strategy
        self.alpha = alpha
        self.threshold = threshold
        self.minimum_observations = minimum_observations
        self.support_k = support_k
        self.entropy_threshold = entropy_threshold
        self.exact_only = exact_only
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
        super().fit(sequence)
        self.trie.prune(self.minimum_observations)
        return self

    def predict(self, context: Iterable[int], k: int = 1) -> list[Prediction]:
        node, _, _ = self.trie.match(context, self.minimum_observations, self.exact_only)
        if node is None or (self.entropy_threshold is not None and self.trie.node_entropy(node) > self.entropy_threshold):
            return []
        observations = sum(node.counts.values())
        ranked = []
        for address, probability in node.weights.items():
            confidence = probability * (observations / (observations + self.support_k)) if self.support_k > 0 else probability
            if confidence >= self.threshold:
                ranked.append((address, confidence))
        return [Prediction(address, weight) for address, weight in sorted(ranked, key=lambda item: (-item[1], item[0]))[:k]]

    def lookup_diagnostics(self, context: Iterable[int]) -> dict[str, float | int | bool | None]:
        return self.trie.lookup_diagnostics(context, self.minimum_observations, self.exact_only)

    def context_statistics(self) -> dict[int, dict[str, float | int]]:
        return self.trie.context_statistics()

    def conditional_entropy(self, depth: int | None = None) -> float:
        return self.trie.conditional_entropy(depth or self.context_depth)

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


class HigherOrderMarkovPredictor(Predictor):
    """A flat fixed-depth context table used to isolate trie structure."""

    def __init__(self, context_depth: int = 2, threshold: float = 0.0):
        self.context_depth = context_depth
        self.threshold = threshold
        self.lookup_cost = max(1, context_depth)
        self.update_cost = max(1, context_depth)
        self.reset()

    def reset(self) -> "HigherOrderMarkovPredictor":
        self.counts: dict[tuple[int, ...], dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._history: list[int] = []
        return self

    def observe(self, address: int) -> None:
        for length in range(1, min(self.context_depth, len(self._history)) + 1):
            context = tuple(self._history[-length:])
            self.counts[context][address] += 1
        self._history.append(address)

    def fit(self, sequence: Iterable[int]) -> "HigherOrderMarkovPredictor":
        super().fit(sequence)
        return self

    def predict(self, context: Iterable[int], k: int = 1) -> list[Prediction]:
        context_tuple = tuple(context)
        length = min(self.context_depth, len(context_tuple))
        transitions = self.counts.get(context_tuple[-length:], {}) if length else {}
        total = sum(transitions.values())
        ranked = [(address, count / total) for address, count in transitions.items() if total and count / total >= self.threshold]
        return [Prediction(address, weight) for address, weight in sorted(ranked, key=lambda item: (-item[1], item[0]))[:k]]

    def lookup_diagnostics(self, context: Iterable[int]) -> dict[str, float | int | bool | None]:
        context_tuple = tuple(context)
        requested = min(self.context_depth, len(context_tuple))
        node = self.counts.get(context_tuple[-requested:], {}) if requested else {}
        return {"requested_depth": self.context_depth, "requested_available_depth": requested, "matched_depth": requested if node else 0, "fallback": False, "unseen": not bool(node), "observations": sum(node.values()), "entropy": -sum((count / sum(node.values())) * math.log2(count / sum(node.values())) for count in node.values()) if node else None}

    def storage_stats(self) -> dict[str, int]:
        entries = sum(len(transitions) for transitions in self.counts.values())
        return {"entries": entries, "nodes": len(self.counts), "edges": entries, "counters": entries, "weights": entries, "estimated_bytes": entries * 24}


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
