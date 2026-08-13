# Hardware Feasibility of Weighted Associative Memory

This phase charges prediction lookup, throughput, update, port, storage, bandwidth, and normalized energy-proxy costs. It preserves `results/`, `results/diagnostics/`, and `results/horizon_analysis/`.

## Final verdict

**A — Requires effectively zero-cost prediction**

- IdealWAM speedup: 1.716x.
- Best realistic-model speedup: 1.107x (HashedContext).
- Predictor-latency break-even: 4 cycles for serial DirectWAM-H16 on LongHigherOrder.
- Effective latency break-even after overlap: 16 overlap cycles in the tested sweep.
- Best predictor throughput tested: 1 prediction/cycle (1.000 predictions/cycle proxy).
- Minimum storage retaining at least 75% of ideal gain: not reached bytes.
- Best counter width: 4 bits.
- Best architecture model: HashedContext.
- Fraction of IdealWAM gain retained: 14.9%.
- Best-model energy proxy relative to baseline: 0.632x.

## Answers

1. The serial H16 break-even is 4 cycles under the default hierarchy; the complete curve is in `latency_sweep.csv`.
2. Parallel overlap reduces effective lookup cost by the measured `max(0, latency - overlap_cycles)` rule; the overlap curve is in `overlap_sweep.csv` and reaches speedup > 1 through 16 tested overlap cycles.
3. Pipelining separates completion latency from issue interval. It helps when the effective critical path is overlapped, but a long issue interval still appears as queue wait/stalls in `throughput_sweep.csv`.
4. Serial depth-proportional traversal is a conservative architectural model, not a silicon claim; its H16 latency is intentionally exposed in the matrix.
5. The hashed context table reports collisions, aliasing, accuracy, storage, and speedup in `hash_table.csv`; it uses deterministic xor/fold/multiply hashing and no cryptographic primitive.
6. The smallest tested fixed budget retaining 75% of ideal gain is not reached bytes; `storage_budget.csv` also reports accuracy/speedup per KB.
7. Counter quantization is swept at 2/4/8/12 bits in `counter_width.csv`; the best observed width is 4 bits, without assuming that width is universally sufficient.
8. Synchronous and deferred updates are compared in `update_cost.csv`; batched update traffic and adaptation delay are in `batching.csv`.
9. Fixed-size tables expose accuracy loss through collision/aliasing and budget rows; replacement is approximated by fixed bucket overwrite/aliasing rather than an unbounded trie.
10. The best abstract organization in this run is HashedContext; the matrix contains the direct comparison against IdealWAM.
11. The best realistic model retains 14.9% of IdealWAM gain.
12. The architecture remains worthwhile only if the selected threshold, storage budget, and normalized energy proxy are acceptable; this run does not remove the earlier predictor-overhead concern by assumption.

## Modeling notes and limitations

Lookup latency and issue interval are separate. Parallel mode applies the explicit overlap rule; the simulator records queue stalls, wait, maximum wait, dropped predictions, and port stalls. Predictor-result caches, fallback modes, candidate-selection strategies, context signatures, and port configurations are reported as additional sensitivity tables. Energy values are arbitrary normalized units: SRAM read=1, SRAM write=1.2, comparison=0.1, hash step=0.2, and DRAM request=100. They are not transistor-level estimates. Fixed-size replacement is approximated by deterministic hash-bucket aliasing; future hardware work should test real replacement traces.

Artifacts: `latency_sweep.csv`, `throughput_sweep.csv`, `overlap_sweep.csv`, `architecture_models.csv`, `storage_budget.csv`, `counter_width.csv`, `hash_table.csv`, `update_cost.csv`, `batching.csv`, `energy_proxy.csv`, `feasibility_matrix.csv`, `tolerance.csv`, `prediction_cache.csv`, `context_signature.csv`, `port_pressure.csv`, `fallback_cost.csv`, `candidate_selection.csv`, `config.json`, and `plots/`.
