# WAM Prediction-Horizon Analysis

This report tests whether accurate higher-order prediction arrives early enough to overlap memory latency. It preserves the previous benchmark and diagnostics artifacts.

## Final classification

**E — Predictor overhead bottleneck**

- Best oracle horizon: H8 (Sequential).
- Best WAM result: DirectWAM H16.
- Oracle maximum speedup: 7.141x.
- WAM maximum speedup: 1.142x.
- WAM fraction of oracle gain at its best row: 11.1%.
- H1 WAM late-prefetch rate: 100.0%.
- Best-horizon WAM late-prefetch rate: 0.0%.

## Answers

1. Perfect predictor speedup: maximum observed oracle speedup was 7.141x; see `oracle_horizon.csv` for workload-specific results.
2. Best oracle horizon: H8.
3. H1 is too late relative to the best oracle horizon.
4. Estimated accesses needed to hide latency: L2 0.5, L3 1.8, DRAM 6.8; use `compute_gap.csv` for measured sensitivity.
5. WAM accuracy at H1/Hbest: 100.0%/100.0%.
6. Direct versus recursive: compare system rows in `summary.csv`; recursive traversal pays repeated lookup cost and can lose confidence multiplicatively.
7. Fraction of oracle: reported above and per row in `summary.csv`.
8. Remaining loss buckets are in `failure_breakdown.csv`; lateness, wrong predictions, bandwidth drops, pollution, and overhead are estimated separately.
9. Compute gaps: the best horizon/maximum speedup curve is in `compute_gap.csv`; mean tested gap was 18.0 cycles.
10. DRAM sweep: `dram_sweep.csv` tests 80/150/300/500-cycle models without claiming universal hardware values.
11. The optimum is empirical, not assumed; oracle and WAM optima are reported above.
12. This evidence strengthens the case only if WAM captures a substantial oracle fraction on irregular workloads; under the default run, the prior negative result is not overturned.

## Integrity and limitations

Direct horizon training only creates examples where both the context position and target position are inside the training prefix. Evaluation addresses are never used to train the predictor. The horizon simulator reuses the existing L1/L2/L3/DRAM hierarchy, outstanding-request limit, cache insertion, pollution attribution, and line normalization. Partial latency hiding is measured from demand wait time rather than treating every late request as useless.

Artifacts: `summary.csv`, `horizon_accuracy.csv`, `oracle_horizon.csv`, `timeliness.csv`, `compute_gap.csv`, `dram_sweep.csv`, `bandwidth_sweep.csv`, `failure_breakdown.csv`, `config.json`, and `plots/`.
