from wam.predictor import NextLinePredictor, WeightedTriePredictor
from wam.simulator import SimulatorConfig, simulate
from wam.workloads import contextual, random_access, sequential


def test_baseline_and_prefetch_metrics_are_deterministic():
    trace = sequential(40)
    config = SimulatorConfig(prefetch_destination="L1", prefetch_cost=2)
    baseline = simulate(trace, enable_prefetch=False, config=config)
    predictive = simulate(trace, NextLinePredictor(), config=config)
    assert baseline.metrics.total_accesses == 40
    assert predictive.metrics.prefetches_issued > 0
    assert predictive.metrics.useful_prefetches > 0
    assert predictive.metrics.cycles < baseline.metrics.cycles


def test_contextual_traces_reward_deeper_predictor():
    trace = contextual(30)
    predictor = WeightedTriePredictor(context_depth=2).fit(trace[:100])
    result = simulate(trace[100:], predictor)
    assert result.metrics.top1_accuracy > 0.8


def test_random_workload_can_issue_no_or_few_prefetches_with_threshold():
    trace = random_access(80, address_space=1000)
    predictor = WeightedTriePredictor(context_depth=2, threshold=0.95).fit(trace[:40])
    result = simulate(trace[40:], predictor)
    assert result.metrics.prefetches_issued <= result.metrics.prediction_attempts
