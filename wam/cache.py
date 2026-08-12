"""Small LRU caches used by the hierarchy simulator."""

from __future__ import annotations

from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity: int, name: str):
        if capacity < 1:
            raise ValueError("cache capacity must be at least 1")
        self.capacity = capacity
        self.name = name
        self._items: OrderedDict[int, None] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def __contains__(self, address: int) -> bool:
        return address in self._items

    def __len__(self) -> int:
        return len(self._items)

    def get(self, address: int) -> bool:
        if address not in self._items:
            self.misses += 1
            return False
        self._items.move_to_end(address)
        self.hits += 1
        return True

    def put(self, address: int) -> int | None:
        if address in self._items:
            self._items.move_to_end(address)
            return None
        self._items[address] = None
        if len(self._items) > self.capacity:
            evicted, _ = self._items.popitem(last=False)
            self.evictions += 1
            return evicted
        return None

    def remove(self, address: int) -> bool:
        if address in self._items:
            del self._items[address]
            return True
        return False

    def contents(self) -> tuple[int, ...]:
        return tuple(self._items.keys())
