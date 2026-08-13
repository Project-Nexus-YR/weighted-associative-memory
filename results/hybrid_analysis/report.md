# Hybrid complementarity analysis

This study uses 37 source-instrumented trace files, frozen 70/30 chronological splits, and non-overlapping evaluation windows. To keep the cycle-level replay bounded, it samples up to 3 windows per size (early, middle, and late when available). The baseline is explicitly **GMC-style**, not the original GMC design; see `gmc_audit.md`.

## Executive result

- GMC-style geomean: **1.420x**
- WAM-H16 geomean: **1.105x**
- OracleHybrid geomean: **1.499x**
- OracleHybrid incremental gain over GMC-style: **5.58%**
- Best cheap selector: **ConfidenceSelector / 1000** at **1.480x**
- Best equal-budget row: **8192 bytes**, GMC 6000 / WAM 2000 / selector 192 bytes

## Answers to the research questions

1. **GMC fidelity:** the current implementation is a **simplified approximation**, not a paper-level reproduction. It lacks the original local/global context organization, PC context, published table organization, and update/arbitration policy.
2. **Consistent WAM wins:** WAM wins 46 of 111 primary windows; per-workload detail is in `complementarity.csv`.
3. **Workloads:** 4 benchmark families have at least one WAM-winning primary window (18 workload/seed traces).
4. **Window fraction:** 41.4% of primary windows are WAM wins.
5. **Advantage:** mean WAM advantage in winning primary windows is 0.179x speedup.
6. **Oracle selector:** the oracle gain is 5.58%; all window sizes are in `oracle_hybrid.csv`.
7. **Complementarity:** substantial under the critical 2% stopping rule.
8. **Discriminating properties:** top basic-statistics signals are **wam_confidence, stride_stability, entropy_reduction**; see `features_summary.csv` for means, effect sizes, and threshold separability.
9. **Cheap selector:** the best implementable selector is ConfidenceSelector with 1.480x; oracle-only rows are not treated as implementable.
10. **Realistic hybrid:** beats GMC-style in this bounded study.
11–12. **Storage:** selector state and lookup/update costs are explicit in `selector_results.csv`; equal-budget rows include all three components in `budget_split.csv`.
13. **Sidecar:** WAM activation rates are reported per selector; the confidence selector is GMC-primary and activates WAM only when GMC confidence is low and WAM confidence is high.
14. **Direct horizon:** `direct_horizon.csv` compares WAM-H16, WAM-H1, RecursiveWAM, and GMC-style. Direct-H16 is not credited automatically for wins that H1 or recursive WAM also obtains.
15. **Contribution:** the current result supports **niche sidecar value**, not a general WAM replacement claim.

## Disagreement analysis

`disagreement.csv` records agreement, disagreement, one-sided predictions, and which predictor is correct on disagreements. It is based on the same frozen contexts and never allows the selector to see future outcomes.

## Selector accounting

Implementable selectors use only start-of-window confidence, prior-window usefulness, and history-derived entropy. `StaticPerWorkloadOracle` and `WindowOracle` are ceilings. The `per_access` line is labeled as a finest-100-access proxy because the current cycle simulator does not expose a composable per-access state snapshot; it must not be read as cycle-exact per-access arbitration.

## Classification

**C — Niche sidecar value**

Continue research: **YES**

Next step: Validate the sidecar on binary-level traces and a faithful GMC implementation.

## Limitations

The traces are source-instrumented, not binary-instrumented. The GMC-style implementation is intentionally simplified. The study freezes each predictor after the 70% training prefix, so it measures phase complementarity under a fixed predictor state rather than adaptive retraining. Measured window sizes are [100, 500, 1000]; sizes [5000, 10000, 50000] were not measured because the configured per-trace cap did not leave a complete window. Selector overhead is modeled as one cycle per arbitration block and bounded state bytes; union-prefetch mode is not used in the primary results, so both predictors never prefetch simultaneously.
