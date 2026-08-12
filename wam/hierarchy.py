"""Cycle-accounting model of an L1/L2/DRAM hierarchy."""

from __future__ import annotations

from dataclasses import dataclass

from .cache import LRUCache


@dataclass(frozen=True)
class HierarchyConfig:
    l1_capacity: int = 64
    l2_capacity: int = 256
    l1_latency: int = 4
    l2_latency: int = 12
    dram_latency: int = 100


@dataclass(frozen=True)
class AccessResult:
    level: str
    latency: int
    evicted_by_fill: int | None = None


class MemoryHierarchy:
    def __init__(self, config: HierarchyConfig = HierarchyConfig()):
        self.config = config
        self.l1 = LRUCache(config.l1_capacity, "L1")
        self.l2 = LRUCache(config.l2_capacity, "L2")
        self.prefetch_evictions = 0

    def contains(self, address: int) -> bool:
        return address in self.l1 or address in self.l2

    def access(self, address: int) -> AccessResult:
        if self.l1.get(address):
            return AccessResult("L1", self.config.l1_latency)
        if self.l2.get(address):
            evicted = self.l1.put(address)
            return AccessResult("L2", self.config.l2_latency, evicted)
        evicted_l2 = self.l2.put(address)
        evicted_l1 = self.l1.put(address)
        evicted = evicted_l1 if evicted_l1 is not None else evicted_l2
        return AccessResult("DRAM", self.config.dram_latency, evicted)

    def prefetch(self, address: int, destination: str = "L1") -> tuple[bool, int | None]:
        """Insert an address; return (inserted, evicted address)."""
        if destination not in {"L1", "L2"}:
            raise ValueError("prefetch destination must be L1 or L2")
        if self.contains(address):
            return False, None
        target = self.l1 if destination == "L1" else self.l2
        evicted = target.put(address)
        if evicted is not None:
            self.prefetch_evictions += 1
        return True, evicted

    def stats(self) -> dict[str, int]:
        return {
            "l1_hits": self.l1.hits,
            "l1_misses": self.l1.misses,
            "l2_hits": self.l2.hits,
            "l2_misses": self.l2.misses,
            "l1_evictions": self.l1.evictions,
            "l2_evictions": self.l2.evictions,
            "prefetch_evictions": self.prefetch_evictions,
        }
