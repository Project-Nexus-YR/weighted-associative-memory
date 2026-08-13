# Final WAM failure diagnostic

## Verdict

- Traces diagnosed: **10** native ChampSim traces.
- Event definition: **every non-PREFETCH access callback received by the L2 WAM module**; miss-only streams are offline filters, not production execution.
- Mean depth-16 context revisit rate: **17.965%**.
- Mean H16 oracle top-1 accuracy: **32.616%**; mean oracle coverage: **37.526%**.
- Mean ShadowWAM H16 accuracy: **0.000%**; mean actual coverage: **0.000%**.
- Oracle-to-Shadow gap: **+32.616%**.
- Absolute-line H16 oracle: **32.616%**; delta H16 oracle: **40.429%**.
- Dominant failure stage: **substantial H16 signal exists, but direct-mapped WAM state is dominated by hash-alias misses**.
- Evidence of implementation bug: **NO**; deterministic H1/H8/H16 alignment checks are recorded in `alignment_validation.csv`.
- Final classification: **C — Learning/state failure**.
- Continuation decision: **RESEARCH_DECISION = CONTINUE_ONE_VARIANT**.

## Diagnostic questions

## NativeSPP diagnostic comparison

Across the same fixed-window runs, the preserved NativeSPP baseline issued approximately **7139354** requests and recorded **164963** useful prefetches. WAM issued **0** requests and recorded **0** useful prefetches. NativeSPP numbers are diagnostic context from the prior frozen evaluation, not a new optimization target.

1. **Does WAM observe enough events?** The event rate is recorded per trace in `activation.csv`; the definition is L2 non-prefetch callbacks, not every instruction or every L1 access.
2. **Are H16 training pairs formed correctly?** `training_pairs_created`, `pending_max`, and `pending_expired` are instrumented; deterministic H1/H8/H16 alignment checks pass.
3. **Do higher-order contexts recur?** `context_reuse.csv` reports exact sequence reuse by depth for warmup, measurement, and combined streams.
4. **What is depth-16 reuse?** The aggregate measurement revisit rate is 17.965%.
5. **What is empirical H16 oracle accuracy?** The aggregate top-1 accuracy is 32.616%, with aggregate oracle coverage 37.526%; the full depth/horizon matrix is in `oracle_predictability.csv`.
6. **What is ShadowWAM H16 accuracy?** 0.000%, with actual coverage 0.000% and no additional diagnostic prefetches issued.
7. **Is there an oracle-to-actual gap?** +32.616%.
8. **Are predictions generated?** 0 in the aggregate WAM path.
9. **Are confidence/support gates suppressing them?** Confidence and support histograms are in `confidence.csv`; `threshold_counterfactual.csv` reports offline confidence cuts at 0.25, 0.50, 0.75, and the current discrete threshold. The current WAM has a confidence gate but no independent support gate; no production threshold was changed.
10. **Are predictions already cached?** Exact L1/L2/LLC state at lookup is not exposed by this ChampSim module API and is marked unavailable.
11. **Are requests rejected or deduplicated?** Accepted versus generated requests is exact; duplicate versus generic API rejection reason is not exposed.
12. **Are accepted prefetches late?** Not measurable through the current prefetcher callback API; recorded as unavailable.
13. **Is table collision/replacement destructive?** Hash aliases, insertions, evictions, and reuse-before-eviction are recorded in `hash_stats.csv` and `replacement_stats.csv`.
14. **Is absolute addressing less predictable than deltas?** `representation_diagnostics.csv` compares absolute and delta oracle accuracy without changing production WAM.
15. **Is H16 an appropriate temporal distance?** `event_distance.csv` reports exact qualifying-access distance and cycle distance; instruction distance is unavailable from this L2 callback API. `plots/h16_instruction_distance.svg` is an explicit unavailable-metric placeholder, while `plots/h16_cycle_distance.svg` reports the exposed cycle distribution.
16. **What caused the negative IPC result?** The evidence funnel is in `failure_funnel.csv`; the final attribution is **substantial H16 signal exists, but direct-mapped WAM state is dominated by hash-alias misses**.
17. **Is there an implementation bug?** No alignment mismatch was found by the deterministic tests; no production semantics were changed.
18. **Should this exact WAM architecture be abandoned?** **No, only one narrow evidence-backed variant is justified**.
19. **Is there a scientifically justified follow-up?** **One narrow follow-up is justified: preserve H16 semantics and test a single alias-resistant table-state/indexing variant; do not broaden the mechanism or retune the gate.**
20. **Should hardware work stop?** **Not yet, but only one narrow diagnostic follow-up is justified.**

## Required final console summary

```text
traces diagnosed: 10
WAM event type: L2 non-PREFETCH access callback
events per 1K instructions: 14.843
depth1 context reuse: 63.857%
depth4 context reuse: 36.417%
depth16 context reuse: 17.965%
H1 oracle accuracy: 0.331%
H8 oracle accuracy: 0.185%
H16 oracle accuracy: 32.616%
H32 oracle accuracy: 0.007%
actual ShadowWAM H16 accuracy: 0.000%
H16 oracle gap: +32.616%
predictions generated: 0
predictions above threshold: 0
prefetches requested: 0
prefetches accepted: 0
useful prefetches: 0
hash collision rate: 99.466%
entry eviction rate: 0.000%
absolute-address H16 oracle accuracy: 32.616%
delta H16 oracle accuracy: 40.429%
dominant failure stage: substantial H16 signal exists, but direct-mapped WAM state is dominated by hash-alias misses
evidence of implementation bug: NO
final classification: C — Learning/state failure
RESEARCH_DECISION = CONTINUE_ONE_VARIANT
```
