"""A bounded, weighted prefix graph for memory-access sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import statistics
from typing import Iterable, Iterator


@dataclass
class WeightedTrieNode:
    """One prefix in the predictor.

    ``counts`` and ``weights`` are keyed by the next address represented by
    each outgoing edge. Keeping both makes the update logic inspectable and
    mirrors the counter-plus-weight storage a hardware implementation could use.
    """

    children: dict[int, "WeightedTrieNode"] = field(default_factory=dict)
    counts: dict[int, int] = field(default_factory=dict)
    weights: dict[int, float] = field(default_factory=dict)


class WeightedTrie:
    """Learn ``P(next | recent context)`` with bounded context depth."""

    VALID_STRATEGIES = {"frequency", "ema"}

    def __init__(self, context_depth: int = 2, strategy: str = "frequency", alpha: float = 0.25):
        if context_depth < 1:
            raise ValueError("context_depth must be at least 1")
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"strategy must be one of {sorted(self.VALID_STRATEGIES)}")
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.context_depth = context_depth
        self.strategy = strategy
        self.alpha = alpha
        self.root = WeightedTrieNode()

    def update(self, context: Iterable[int], next_address: int) -> None:
        """Record one transition using the most recent bounded context."""
        context_tuple = tuple(context)[-self.context_depth :]
        # Store every suffix, not only the longest one. This makes a depth-N
        # predictor able to use the longest-known-suffix fallback path.
        for length in range(1, len(context_tuple) + 1):
            node = self.root
            for address in context_tuple[-length:]:
                node = node.children.setdefault(address, WeightedTrieNode())
            node.counts[next_address] = node.counts.get(next_address, 0) + 1
            if self.strategy == "frequency":
                total = sum(node.counts.values())
                node.weights = {address: count / total for address, count in node.counts.items()}
            else:
                # Update all known edges with a one-hot observation, then normalize
                # so weights remain probability-like and easy to threshold.
                for address in list(node.weights):
                    observation = 1.0 if address == next_address else 0.0
                    node.weights[address] = (1 - self.alpha) * node.weights[address] + self.alpha * observation
                if next_address not in node.weights:
                    node.weights[next_address] = self.alpha
                total = sum(node.weights.values())
                if total:
                    node.weights = {address: weight / total for address, weight in node.weights.items()}

    def fit(self, sequence: Iterable[int]) -> "WeightedTrie":
        """Train on every transition in a sequence."""
        history: list[int] = []
        for address in sequence:
            if history:
                self.update(history, address)
            history.append(address)
        return self

    def _find_node(self, context: Iterable[int]) -> WeightedTrieNode | None:
        """Find the longest available suffix, with bounded fallback."""
        context_tuple = tuple(context)
        max_length = min(self.context_depth, len(context_tuple))
        for length in range(max_length, 0, -1):
            node = self.root
            try:
                for address in context_tuple[-length:]:
                    node = node.children[address]
            except KeyError:
                continue
            if node.weights:
                return node
        return self.root if self.root.weights else None

    def match(self, context: Iterable[int], minimum_observations: int = 1, exact_only: bool = False) -> tuple[WeightedTrieNode | None, int, int]:
        """Return ``(node, matched_depth, requested_depth)`` for diagnostics."""
        context_tuple = tuple(context)
        requested_depth = min(self.context_depth, len(context_tuple))
        if requested_depth == 0:
            return None, 0, requested_depth
        lengths = (requested_depth,) if exact_only else range(requested_depth, 0, -1)
        for length in lengths:
            node = self.root
            try:
                for address in context_tuple[-length:]:
                    node = node.children[address]
            except KeyError:
                continue
            if node.weights and sum(node.counts.values()) >= minimum_observations:
                return node, length, requested_depth
        return None, 0, requested_depth

    @staticmethod
    def node_entropy(node: WeightedTrieNode | None) -> float:
        if node is None:
            return 0.0
        return -sum(weight * math.log2(weight) for weight in node.weights.values() if weight > 0)

    def lookup_diagnostics(self, context: Iterable[int], minimum_observations: int = 1, exact_only: bool = False) -> dict[str, float | int | bool | None]:
        node, matched_depth, requested_depth = self.match(context, minimum_observations, exact_only)
        observations = sum(node.counts.values()) if node else 0
        return {
            "requested_depth": self.context_depth,
            "requested_available_depth": requested_depth,
            "matched_depth": matched_depth,
            "fallback": bool(node and matched_depth < requested_depth),
            "unseen": node is None,
            "observations": observations,
            "entropy": self.node_entropy(node) if node else None,
        }

    def context_statistics(self, max_depth: int | None = None) -> dict[int, dict[str, float | int]]:
        """Summarize context reuse/support by prefix depth."""
        limit = max_depth or self.context_depth
        by_depth: dict[int, list[int]] = {depth: [] for depth in range(1, limit + 1)}
        stack: list[tuple[WeightedTrieNode, int]] = [(self.root, 0)]
        while stack:
            node, depth = stack.pop()
            if 1 <= depth <= limit and node.counts:
                by_depth[depth].append(sum(node.counts.values()))
            if depth < limit:
                stack.extend((child, depth + 1) for child in node.children.values())
        result: dict[int, dict[str, float | int]] = {}
        for depth, observations in by_depth.items():
            result[depth] = {
                "unique_contexts": len(observations),
                "total_observations": sum(observations),
                "mean_observations": statistics.mean(observations) if observations else 0.0,
                "median_observations": statistics.median(observations) if observations else 0.0,
                "contexts_seen_once": sum(value == 1 for value in observations),
                "contexts_seen_at_least_2": sum(value >= 2 for value in observations),
                "contexts_seen_at_least_5": sum(value >= 5 for value in observations),
                "contexts_seen_at_least_10": sum(value >= 10 for value in observations),
            }
        return result

    def conditional_entropy(self, depth: int) -> float:
        stats = self.context_statistics(depth).get(depth, {})
        total = float(stats.get("total_observations", 0))
        if not total:
            return 0.0
        entropies: list[tuple[int, float]] = []
        stack: list[tuple[WeightedTrieNode, int]] = [(self.root, 0)]
        while stack:
            node, current_depth = stack.pop()
            if current_depth == depth and node.counts:
                observations = sum(node.counts.values())
                entropies.append((observations, self.node_entropy(node)))
            elif current_depth < depth:
                stack.extend((child, current_depth + 1) for child in node.children.values())
        return sum(observations * entropy for observations, entropy in entropies) / total

    def prune(self, minimum_observations: int) -> int:
        """Remove context branches with fewer than the requested observations."""
        if minimum_observations <= 1:
            return 0
        removed = 0

        def visit(node: WeightedTrieNode) -> None:
            nonlocal removed
            for address, child in list(node.children.items()):
                visit(child)
                if sum(child.counts.values()) < minimum_observations:
                    del node.children[address]
                    removed += 1

        visit(self.root)
        return removed

    def predict(self, context: Iterable[int], k: int = 1, threshold: float = 0.0) -> list[tuple[int, float]]:
        """Return up to ``k`` highest-weight next addresses.

        Sorting is limited to the small outgoing edge set of one matched node;
        the entire trie is never scanned.
        """
        if k < 1:
            return []
        node = self._find_node(context)
        if node is None:
            return []
        ranked = ((address, weight) for address, weight in node.weights.items() if weight >= threshold)
        return sorted(ranked, key=lambda item: (-item[1], item[0]))[:k]

    def iter_nodes(self) -> Iterator[WeightedTrieNode]:
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(node.children.values())

    def storage_stats(self, address_bytes: int = 8, weight_bytes: int = 4, counter_bytes: int = 4, pointer_bytes: int = 8) -> dict[str, int]:
        """Return a deliberately rough predictor storage estimate."""
        nodes = list(self.iter_nodes())
        edges = sum(len(node.children) for node in nodes)
        # Each edge stores an address, counter, weight, and child index/pointer.
        estimated_bytes = edges * (address_bytes + weight_bytes + counter_bytes + pointer_bytes)
        return {
            "entries": edges,
            "nodes": len(nodes),
            "edges": edges,
            "counters": edges,
            "weights": edges,
            "estimated_bytes": estimated_bytes,
        }

    def ascii(self, max_depth: int = 4) -> str:
        """Render a small trie for inspection and README-like demos."""
        lines = ["ROOT"]

        def walk(node: WeightedTrieNode, prefix: str, depth: int) -> None:
            if depth >= max_depth:
                return
            children = sorted(node.children.items())
            for index, (address, child) in enumerate(children):
                branch = "└── " if index == len(children) - 1 else "├── "
                weight = node.weights.get(address, 0.0)
                lines.append(f"{prefix}{branch}{address} [{weight:.2f}]")
                walk(child, prefix + ("    " if index == len(children) - 1 else "│   "), depth + 1)

        walk(self.root, "", 0)
        return "\n".join(lines)
