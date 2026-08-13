# Set-associative WAM diagnostic

## Experimental scope

This is the single authorized follow-up to the final DirectMappedWAM diagnostic: a 64-set × 4-way table with 256 total entries. Each way preserves the DirectMappedWAM entry layout and H16 update semantics. The confidence threshold remains 8, the prefetch path is unchanged, and the replacement pointer adds 64 logical bytes. The evaluation uses the same ten native traces, 5,000,000 warmup instructions, 10,000,000 simulated instructions, one core, and pinned ChampSim commit `51588e1d6f97875fe8de1a3621d28668bff83fcf`. DirectMappedWAM, NativeSPP, and NoPrefetch are frozen controls from the prior fixed-window evaluation.

## Results

- Direct context hit rate: **0.387%**.
- Set-associative context hit rate: **0.189%**.
- Direct hash-alias loss rate: **99.466%**.
- Set unresolved all-ways conflict rate: **99.662%**.
- Oracle H16 accuracy / coverage: **32.616% / 37.526%**.
- Direct ShadowWAM accuracy / coverage: **0.000% / 0.000%**.
- Set ShadowWAM accuracy / coverage: **0.000% / 0.000%**.
- Set accuracy / coverage recovery versus oracle: **0.000% / 0.000%**.
- IPC geomean speedup: DirectMappedWAM **1.000×**, SetAssociativeWAM **1.000×**, NativeSPP **1.120×**.
- Frozen delta-oracle advantage: **7.813%** absolute; no delta simulation was run.

## Required questions

1. **Did four-way associativity recover the lost context state?** No; the aggregate tag-hit rate did not materially exceed the direct-mapped control.
2. **How much did it reduce alias loss?** Direct alias loss was 99.466%; SA unresolved conflict loss was 99.662%. The SA-specific direct-map-equivalent counter is in `hash_comparison.csv`.
3. **How many contexts compete per set?** The per-trace median/p90/p95/p99/max pressure counters are in `set_pressure.csv`; sets reaching four ways are recorded per trace.
4. **Was the fixed 256-entry budget preserved?** Yes. The table remains 256 entries; only 64 one-byte round-robin pointers were added.
5. **Did SA improve context hit rate, coverage, or reuse?** See `per_trace.csv`, `context_hit_rate.svg`, `set_occupancy.svg`, and `set_pressure.svg`; the aggregate hit rates above are the primary gate.
6. **Did SA recover the offline H16 oracle?** The oracle is unchanged because the trace and semantics are unchanged; `oracle_recovery.csv` measures actual recovery for both state layouts.
7. **Did ShadowWAM accuracy improve?** Direct ShadowWAM was 0.000%; SA was 0.000%. The per-trace comparison is in `shadow_accuracy.csv`.
8. **Did ShadowWAM coverage improve?** Direct was 0.000%; SA was 0.000%.
9. **Did confidence or support distributions improve?** The frozen direct and new SA distributions are in `confidence.csv` and `support.csv`; no threshold or support policy was retuned.
10. **Did production predictions improve?** Direct generated 0; SA generated 0. The funnel is in `failure_funnel.csv`.
11. **Did useful prefetches improve?** Direct requested 0 and recorded 0 useful; SA requested 0 and recorded 0 useful.
12. **Did IPC improve?** The primary comparison is in `ipc.csv` and `ipc_comparison.svg`; NativeSPP remains the unchanged reference.
13. **What did occupancy show?** SA occupancy bins 0..4, occupied sets, and final occupied entries are in `occupancy.csv`; direct per-set bins are not exposed.
14. **What did set pressure show?** SA pressure is measured as distinct context keys competing for each set; distribution statistics are in `set_pressure.csv`.
15. **Was the storage budget respected?** Yes: DirectMappedWAM 8448 bytes, SetAssociativeWAM 8512 bytes, delta 64 bytes.
16. **Does the evidence justify a delta variant?** No. The frozen delta advantage is 7.813%, below the meaningful continuation threshold used here, and SA did not satisfy the state-recovery gate.
17. **What is the final research decision?** **RESEARCH_DECISION = STOP**; next variant: **NONE**.

## Final classification

**A — Aliasing hypothesis falsified**

Dominant remaining bottleneck: **four-way associativity does not recover context state; set pressure and replacement churn remain destructive**.

## Required final console summary

traces evaluated: 10
DirectMappedWAM storage bytes: 8448
SetAssociativeWAM storage bytes: 8512
direct-map alias rate: 99.466%
set-associative unresolved conflict rate: 99.662%
direct context hit rate: 0.387%
set-associative context hit rate: 0.189%
oracle H16 accuracy: 32.616%
DirectShadow H16 accuracy: 0.000%
SetAssociativeShadow H16 accuracy: 0.000%
oracle H16 coverage: 37.526%
DirectShadow coverage: 0.000%
SetAssociativeShadow coverage: 0.000%
accuracy recovery %: 0.000%
coverage recovery %: 0.000%
predictions generated:
direct: 0
set-associative: 0
prefetches requested:
direct: 0
set-associative: 0
useful prefetches:
direct: 0
set-associative: 0
DirectMappedWAM geomean IPC speedup: 1.000x
SetAssociativeWAM geomean IPC speedup: 1.000x
NativeSPP geomean IPC speedup: 1.120x
dominant remaining bottleneck: four-way associativity does not recover context state; set pressure and replacement churn remain destructive
final classification: A — Aliasing hypothesis falsified
RESEARCH_DECISION = STOP
next variant: NONE
