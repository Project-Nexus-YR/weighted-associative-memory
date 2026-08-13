"""Configurable L1/L2/L3/DRAM cycle-accounting hierarchy."""

from __future__ import annotations

from dataclasses import dataclass

from .cache import LRUCache


@dataclass(frozen=True)
class HierarchyConfig:
    l1_capacity: int = 64
    l2_capacity: int = 256
    l3_capacity: int = 1024
    l1_latency: int = 4
    l2_latency: int = 12
    l3_latency: int = 40
    dram_latency: int = 150
    cache_line_size: int = 64

    def __post_init__(self) -> None:
        if self.cache_line_size < 1:
            raise ValueError("cache_line_size must be positive")


@dataclass(frozen=True)
class AccessResult:
    level: str
    latency: int
    evicted: tuple[int, ...] = ()


class MemoryHierarchy:
    """Inclusive-ish hierarchy operating only on normalized line addresses."""

    def __init__(self, config: HierarchyConfig = HierarchyConfig()):
        self.config = config
        self.l1 = LRUCache(config.l1_capacity, "L1")
        self.l2 = LRUCache(config.l2_capacity, "L2")
        self.l3 = LRUCache(config.l3_capacity, "L3")
        self.prefetch_evictions = 0

    def normalize(self, address: int) -> int:
        return address // self.config.cache_line_size

    def level_of(self, line: int) -> str | None:
        if line in self.l1:
            return "L1"
        if line in self.l2:
            return "L2"
        if line in self.l3:
            return "L3"
        return None

    def contains(self, line: int) -> bool:
        return self.level_of(line) is not None

    @staticmethod
    def _unique(*addresses: int | None) -> tuple[int, ...]:
        return tuple(dict.fromkeys(address for address in addresses if address is not None))

    def access(self, line: int) -> AccessResult:
        if self.l1.get(line):
            return AccessResult("L1", self.config.l1_latency)
        if self.l2.get(line):
            evicted = self._unique(self.l1.put(line))
            return AccessResult("L2", self.config.l2_latency, evicted)
        if self.l3.get(line):
            evicted_l1 = self.l1.put(line)
            evicted_l2 = self.l2.put(line)
            return AccessResult("L3", self.config.l3_latency, self._unique(evicted_l1, evicted_l2))
        evicted_l1 = self.l1.put(line)
        evicted_l2 = self.l2.put(line)
        evicted_l3 = self.l3.put(line)
        return AccessResult("DRAM", self.config.dram_latency, self._unique(evicted_l1, evicted_l2, evicted_l3))

    def insert_prefetch(self, line: int, destination: str = "L1") -> tuple[int, ...]:
        """Complete a prefetch into a cache and return lines evicted by it."""
        if destination not in {"L1", "L2", "L3"}:
            raise ValueError("prefetch destination must be L1, L2, or L3")
        if self.contains(line):
            return ()
        target = {"L1": self.l1, "L2": self.l2, "L3": self.l3}[destination]
        evicted = target.put(line)
        if evicted is not None:
            self.prefetch_evictions += 1
        return self._unique(evicted)

    def stats(self) -> dict[str, int]:
        return {
            "l1_hits": self.l1.hits,
            "l1_misses": self.l1.misses,
            "l2_hits": self.l2.hits,
            "l2_misses": self.l2.misses,
            "l3_hits": self.l3.hits,
            "l3_misses": self.l3.misses,
            "l1_evictions": self.l1.evictions,
            "l2_evictions": self.l2.evictions,
            "l3_evictions": self.l3.evictions,
            "prefetch_evictions": self.prefetch_evictions,
        }
