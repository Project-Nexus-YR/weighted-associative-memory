# ChampSim real-trace evaluation

## Final verdict

- Real native ChampSim traces evaluated: **10**; fixed matrix rows: **50**.
- Primary run: **5,000,000 warmup + 10,000,000 simulation instructions**, one core, fixed across traces.
- Strong native baseline selected by aggregate geomean IPC ratio: **NativeSPP** (1.120x).
- DirectWAM-H16 overall geomean: **1.000x**; versus StrongBaseline: **0.893x**; irregular geomean: **1.000x**.
- Hybrid-SPP+WAM overall geomean: **1.120x**; increment over StrongBaseline: **+0.000%**; irregular increment: **+0.000%**.
- Best WAM workload speedup: **1.000x**; worst: **1.000x**.
- WAM wins at the preregistered +2% threshold on **0/10** workloads; Hybrid wins on **0/10**; Hybrid worst regression versus StrongBaseline: **+0.000%**.
- Final classification: **A — No benefit beyond modern baseline**.
- Exact RTL decision: **MOVE_TO_RTL = NO**.

## Answers to the requested questions

1. **Trace legitimacy:** the manifest records ten native compressed ChampSim traces from a public SPEC CPU 2006 trace record, with local size and MD5 verification.
2. **Workload breadth:** ten workloads are evaluated, including more than five frozen irregular/scientific/pointer/tree classes.
3. **Reproducibility:** ChampSim and vcpkg commits, base configuration, command lines, JSON, stdout, stderr, and elapsed time are recorded.
4. **Primary comparison:** NoPrefetch, native SPP, native IP-stride, DirectWAM-H16, and the cheap online SPP+WAM selector are included.
5. **Strong baseline:** NativeSPP is the strongest native candidate by the recorded aggregate IPC ratio; its canonical result is reported separately from the fixed WAM ledger.
6. **WAM result:** DirectWAM-H16 reaches 1.000x overall, 0.893x versus StrongBaseline, and 1.000x on irregular workloads; it issued no trace-level useful stream in this matrix.
7. **Budget:** H16 declares 8,448 bytes of fixed state and fits in all recorded 16/32/64 KiB budget points; H8/H32 sweeps were intentionally deferred until after the primary.
8. **Hybrid:** the implementable selector is measured in `hybrid.csv`; its result is 1.120x overall, +0.000% over StrongBaseline, and +0.000% on irregular workloads.
9. **Oracle:** a true 1M-window oracle is not claimed because this JSON path has no address/event stream; the trace-level ceiling is recorded separately in `oracle_hybrid.csv` and is only +0.000% over StrongBaseline.
10. **Disagreement:** performance disagreement versus native SPP is measured per workload; address-level overlap is not measurable from aggregate JSON.
11. **Bandwidth:** DRAM read/write counts and 64-byte traffic proxies are reported; exact prefetch-only DRAM bytes are not exposed by this output path.
12. **Pollution/timeliness:** exact cache-pollution and late-prefetch counts are not claimed; the limitation is explicit in the per-workload table and failure analysis.
13. **Memory intensity:** memory-bound grouping is frozen as NoPrefetch LLC demand MPKI >= 10 and is reported in `aggregate.csv`.
14. **Failure analysis:** `failure_analysis.csv` identifies every WAM workload below the strongest native and distinguishes performance evidence from unmeasured address-level causes.
15. **Regression guard:** the -2% guard is applied per workload, and the count is reported above and in `aggregate.csv`.
16. **RTL readiness:** the gate is not cleared; the exact decision is **MOVE_TO_RTL = NO**.
17. **Bottom line:** A — No benefit beyond modern baseline. The evidence supports the recorded benchmark conclusion only; it does not justify changing WAM or claiming novelty beyond this experiment.

## Evidence boundary

The trace files are not committed because they are multi-gigabyte external artifacts. Recreate them from the URLs and checksums in `trace_manifest.csv`, then run the recorded scripts. Prior `results/champsim_validation` evidence is intentionally preserved.

## Final console summary

```text
ChampSim commit: 51588e1d6f97875fe8de1a3621d28668bff83fcf
traces evaluated: 10
irregular traces evaluated: 6
strongest native baseline: NativeSPP
WAM-H16 geomean vs NoPrefetch: 1.000x
WAM-H16 geomean vs StrongBaseline: 0.893x
Hybrid geomean vs NoPrefetch: 1.120x
Hybrid increment over StrongBaseline: +0.000%
Hybrid irregular-workload increment: +0.000%
OracleHybrid increment: +0.000% (trace-level ceiling; 1M-window status: not_run)
fraction of traces WAM wins: 0.0%
fraction of traces Hybrid wins: 0.0%
best real workload gain: +0.000% Hybrid over StrongBaseline
worst regression: +0.000% Hybrid over StrongBaseline
WAM state bytes: 8448
Hybrid total state bytes: 60224
DRAM traffic delta: +28.487%
DirectH16 vs Recursive result: not_run; no native RecursiveWAM implementation
best limited horizon: H16 primary; H8/H32 not_run
best limited storage budget: 16 KiB minimum fit; primary headline budget 32 KiB
paper readiness / 10: 3/10
final classification: A — No benefit beyond modern baseline
MOVE_TO_RTL = NO
single most important next step: stop the hardware path; if pursuing WAM research, add address-level/window instrumentation and test a clearly differentiated mechanism.
```
