from wam.hierarchy import HierarchyConfig, MemoryHierarchy
from wam.predictor import NextLinePredictor, StridePredictor, WeightedTriePredictor
from wam.simulator import SimulatorConfig, simulate
from wam.traces import iter_addresses, normalize_addresses


def test_l3_hit_is_distinguished_from_dram():
    hierarchy = MemoryHierarchy(HierarchyConfig(l1_capacity=1, l2_capacity=1, l3_capacity=2, cache_line_size=1))
    assert hierarchy.access(1).level == "DRAM"
    assert hierarchy.access(2).level == "DRAM"
    assert hierarchy.access(1).level == "L3"


def test_trace_loader_streams_hex_integer_and_comments(tmp_path):
    path = tmp_path / "trace.txt"
    path.write_text("# header\n0x40\n12 # inline\n\n", encoding="utf-8")
    assert list(iter_addresses(path)) == [64, 12]
    assert list(normalize_addresses([64, 127, 128], 64)) == [1, 1, 2]


def test_stride_requires_stable_confidence():
    predictor = StridePredictor(confidence_threshold=2)
    predictor.observe(0)
    assert predictor.predict([0]) == []
    predictor.observe(4)
    assert predictor.predict([4]) == []
    predictor.observe(8)
    assert predictor.predict([8])[0].address == 12


def test_outstanding_prefetch_can_arrive_late():
    config = SimulatorConfig(
        hierarchy=HierarchyConfig(cache_line_size=1, l1_latency=1, l2_latency=2, l3_latency=4, dram_latency=20),
        predictor_lookup_cost=0,
        prefetch_issue_cost=0,
    )
    result = simulate([0, 1, 2, 3], NextLinePredictor(), config)
    assert result.metrics.prefetches_issued > 0
    assert result.metrics.late_prefetches > 0
    assert result.metrics.prefetches_completed > 0


def test_train_test_evaluation_preserves_frozen_predictor_state():
    trace = [0, 1, 10, 2, 1, 11] * 10
    config = SimulatorConfig(hierarchy=HierarchyConfig(cache_line_size=1))
    predictor = WeightedTriePredictor(context_depth=2).fit(trace[:30])
    before = predictor.storage_stats()
    simulate(trace[30:], predictor, config, initial_context=trace[28:30], learning=False)
    after = predictor.storage_stats()
    assert before == after
