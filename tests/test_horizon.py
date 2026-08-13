from wam.horizon import DirectHorizonWAM, HorizonConfig, OracleHorizon, simulate_horizon
from wam.hierarchy import HierarchyConfig
from wam.workloads import contextual


def test_direct_horizon_uses_only_training_prefix():
    trace = contextual(30)
    train = trace[:60]
    held_out = trace[60:]
    predictor = DirectHorizonWAM(context_depth=2, horizon=2).fit(train)
    prediction = predictor.predict_horizon(train[-2:], k=1)[0]
    assert prediction.horizon == 2
    assert prediction.address == held_out[1]


def test_oracle_horizon_is_perfect_on_future_targets():
    trace = [address * 64 for address in range(20)]
    config = HorizonConfig(hierarchy=HierarchyConfig(cache_line_size=64), compute_cycles_between_accesses=8)
    result = simulate_horizon(trace, OracleHorizon(), horizon=4, config=config, enable_prefetch=False)
    assert result.metrics.top1_accuracy == 1.0
    assert result.metrics.prediction_attempts == 16


def test_partial_latency_hiding_is_measured():
    trace = [address * 64 for address in range(30)]
    config = HorizonConfig(
        hierarchy=HierarchyConfig(cache_line_size=64, l1_capacity=2, dram_latency=20),
        compute_cycles_between_accesses=5,
        prefetch_issue_cost=0,
    )
    result = simulate_horizon(trace, OracleHorizon(), horizon=1, config=config)
    assert result.metrics.late_prefetches > 0
    assert result.metrics.partially_hidden_misses > 0
    assert result.metrics.cycles_hidden > 0
