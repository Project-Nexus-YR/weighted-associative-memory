from wam.hardware import HashedContextPredictor
from wam.horizon import DirectHorizonWAM, HorizonConfig, simulate_horizon
from wam.workloads import higher_order_ambiguous, to_byte_addresses


def _trace() -> tuple[list[int], list[int]]:
    values = higher_order_ambiguous(context_count=8, repeats=5)
    cut = int(len(values) * 0.7)
    return values[:cut], values[cut:]


def test_lookup_latency_is_separate_from_zero_cost_prediction() -> None:
    train, evaluation = _trace()
    predictor = DirectHorizonWAM(context_depth=4, horizon=4).fit(train)
    raw = to_byte_addresses(evaluation)
    zero = simulate_horizon(raw, predictor, 4, HorizonConfig(predictor_lookup_latency=0, predictor_update_latency=0), initial_context=train[-4:])
    expensive = simulate_horizon(raw, predictor, 4, HorizonConfig(predictor_lookup_latency=16, predictor_update_latency=0), initial_context=train[-4:])
    assert expensive.metrics.predictor_overhead > zero.metrics.predictor_overhead
    assert expensive.cycles > zero.cycles


def test_deferred_batched_updates_reduce_update_overhead() -> None:
    train, evaluation = _trace()
    predictor = DirectHorizonWAM(context_depth=4, horizon=4).fit(train)
    raw = to_byte_addresses(evaluation)
    synchronous = simulate_horizon(raw, predictor, 4, HorizonConfig(predictor_lookup_latency=1, predictor_update_latency=4), initial_context=train[-4:])
    deferred = simulate_horizon(raw, predictor, 4, HorizonConfig(predictor_lookup_latency=1, predictor_update_latency=4, deferred_updates=True, update_batch_size=8), initial_context=train[-4:])
    assert deferred.metrics.predictor_overhead < synchronous.metrics.predictor_overhead
    assert deferred.metrics.update_count == synchronous.metrics.update_count


def test_hashed_context_reports_collisions_and_quantized_storage() -> None:
    train, _ = _trace()
    predictor = HashedContextPredictor(context_depth=4, horizon=4, table_size=4, counter_bits=4, signature_bits=8).fit(train)
    assert predictor.collision_rate > 0
    assert predictor.storage_stats()["estimated_bytes"] <= 4 * predictor.entry_bytes + 4
    assert predictor.predict(train[-4:], k=1)
