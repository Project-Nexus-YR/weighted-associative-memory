# Weighted Associative Memory Research Report

This report is generated from deterministic chronological train/test benchmark runs. Synthetic traces are represented as cache-line IDs and converted to raw byte addresses using a 64-byte default line size. The reported latencies are simulation parameters, not claims about a particular CPU.

## Verdict

- WAM wins: none under the default configuration.
- WAM loses or does not beat the best simpler system: Contextual, LongerDependency, PhaseChanging, Probabilistic, Random, Repeating, Sequential, Stride.
- Best observed WAM configuration: WAM depth=1 with mean speedup 1.025x.
- Geometric-mean speedup of the best WAM depth per workload: 0.974x.
- Maximum mean WAM speedup observed: 1.025x.
- Best confidence threshold in the break-even sweep: 0.00 (DRAM=300), or not reached.
- Storage at the best WAM row: 96 bytes.
- Incremental storage per accuracy percentage point versus depth 1: -2.7 bytes/point.
- Online learning warm-up to approximately 80% of final top-1 accuracy: 36 accesses.
- Approximate accuracy break-even: 150 cycles: not reached; 300 cycles: 0.8990825688073395.

The result should be read workload-by-workload. Sequential and constant-stride streams favor conventional prefetchers; WAM is only expected to justify its state and lookup overhead where higher-order context survives the held-out split. Random access is a negative control.

## Where context helped

Depths beyond 1 were the speedup winner on: none.
Depths that improved top-1 accuracy by more than one percentage point over depth 1: none under this split.
The best contextual sweep point was 1 depth, threshold 0.5, top-K 3 at 1.021x.
The benchmark does not assume monotonic improvement: larger contexts increase storage and lookup cost, and can under-train when a trace is short or phase-changing.

## Adaptive weighting and ablations

The ablation table compares frequency weighting, EMA weighting, thresholds, depth, and no-prefetch operation. On the contextual ablation, EMA speedup was 0.830x versus 0.830x for frequency depth 2.
EMA should be judged primarily on the phase-changing trace, where stale frequency counts can remain misleading after a transition.

## Limitations and next experiment

The simulator is not cycle-accurate hardware: it models serialized demand progress, bounded outstanding prefetches, cache pollution attribution, and a documented predictor overhead. The single most important next experiment is replaying the same benchmark against long traces captured from representative programs, with a calibrated memory-level parallelism and bandwidth model.

Artifacts: `summary.csv`, `detailed_results.csv`, `sweep.csv`, `ablation.csv`, `learning_curve.csv`, `break_even.csv`, `config.json`, and `plots/`.
