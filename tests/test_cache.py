from wam.cache import LRUCache
from wam.hierarchy import HierarchyConfig, MemoryHierarchy


def test_lru_eviction_and_hit():
    cache = LRUCache(2, "test")
    cache.put(1)
    cache.put(2)
    assert cache.get(1)
    assert cache.put(3) == 2
    assert 2 not in cache
    assert cache.evictions == 1


def test_hierarchy_promotes_l2_hit_to_l1():
    hierarchy = MemoryHierarchy(HierarchyConfig(l1_capacity=1, l2_capacity=2))
    assert hierarchy.access(1).level == "DRAM"
    assert hierarchy.access(1).level == "L1"
    hierarchy.access(2)
    assert hierarchy.access(1).level == "L2"
    assert 1 in hierarchy.l1
