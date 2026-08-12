"""A bounded, weighted prefix graph for memory-access sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
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
        node = self.root
        for address in context_tuple:
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
        return {"nodes": len(nodes), "edges": edges, "estimated_bytes": estimated_bytes}

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
