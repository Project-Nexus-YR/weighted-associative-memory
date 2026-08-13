# Real-trace evaluation

This report uses source-instrumented data-load traces captured from actual benchmark executions. These traces are not equivalent to binary instrumentation; the capture method is recorded as `source_instrumented`.

## Final verdict

- Traces captured: 37; evaluated representatives: 9.
- Total references analyzed: 45000.
- Best WAM configuration: DirectWAM-H16-miss-only (1.098x geomean).
- Best prior-art-style baseline: GMC (1.714x geomean).
- WAM irregular geomean: 1.079x; overall DirectWAM-H16 geomean: 1.070x.
- Best WAM workload speedup: 1.287x.
- Worst DirectWAM-H16 regression: -0.001.
- Best tested bounded storage budget: 8192 bytes.
- Best prediction horizon: H32.
- Direct-vs-recursive advantage: +0.007x geomean.
- Fraction of workloads where DirectWAM-H16 wins all listed prior baselines: 11.1%.
- Hybrid geomean: 1.028x.
- Depth-1 to depth-16 H1 oracle change: 100.0% -> 100.0% (+0.0%).
- Best H16 empirical oracle accuracy at depth 16: 12.8%.
- Cross-run retention: not measured in this representative run; five-seed traces are captured in `capture_inventory.csv`.

## Answers

1–3. Real traces show repeated structure, but the measured depth-1/depth-16 H1 oracle change is only +0.0%; long-horizon opportunity is limited in this bounded sample.
4–9. DirectWAM-H16 reaches 1.070x, below GMC at 1.714x; VLDP/SPP/GMC and Markov-N are included in `summary.csv`.
10–13. Direct-vs-recursive and budget rows are reported, but the positive WAM result is narrow and source-instrumented rather than binary-traced.
14–20. Multi-seed captures exist, while cross-run generalization and binary-level confirmation remain open; the evidence does not justify RTL or a novelty claim over prior-art-style predictors.

## Classification

**C — Narrow workload-specific win**

## Limitations

VLDP, SPP, and GMC are simplified architectural approximations documented in `wam/real_predictors.py`. The main evaluation uses one seed per benchmark and a 5,000-access chronological prefix for tractability; the raw captures are longer and multi-seed metadata is retained. No instruction PCs are fabricated, and source-instrumented traces should be followed by binary-tracer confirmation when available.

## Paper readiness

- [x] Real traces demonstrate some repeated structure
- [ ] Long-horizon signal is measurable
- [ ] WAM beats the strongest baseline
- [x] Direct horizon beats recursive speculation
- [ ] Equal-budget multi-seed conclusion is complete
- [ ] Binary-level trace confirmation is complete
- [ ] Novelty is distinguishable from existing predictor families

Paper-readiness score: 3/10.

Single most important next step: aggregate the captured five-seed traces under the same bounded configuration and obtain binary-level data traces for confirmation.
