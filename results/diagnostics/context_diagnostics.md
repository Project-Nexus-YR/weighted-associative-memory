# WAM Context Diagnostics

This is a falsification report. It keeps the prior benchmark untouched and separates representation quality from prefetch execution.

## Final classification

**C — Prefetch execution problem**

Prediction is accurate in the isolated oracle mode; the full benchmark must determine whether timing, bandwidth, pollution, and lookup cost erase the benefit.

## Answers to the diagnostic questions

1. Higher-order information present: depth-1 oracle accuracy was 80.7%; depth-4 oracle accuracy was 100.0%; conditional entropy changed by 0.916 bits.
2. WAM learning it: actual depth-4 accuracy was 100.0%, with an oracle gap of 0.0%.
3. Deep-context reuse: depth-4 training/evaluation reuse was 100.0%; at one million accesses it was 100.0%.
4. Fallback/unseen behavior is in `context_depth.csv`; the simulator records matched-depth histograms, fallback counts, and unseen contexts for every lookup.
5. Repetition density: depth-2 target accuracy first exceeded 90% at not reached repetitions per context.
6. WAM vs equivalent Markov-N: see `markov_comparison.csv`; both use the same context depth and the same train/test split.
7. Entropy gating: see `entropy_policy.csv`; it is useful only when entropy correlates with harmful speculative requests.
8. Bottleneck: the depth-4 one-million-access oracle/actual pair was 100.0%/100.0%; compare full-mode speedups in the prior report to isolate execution cost.
9. Empirical maximum: the `oracle_accuracy` column is the training-distribution context oracle for each depth; at one million accesses depth 1/4 were 75.2%/100.0%.
10. Continue? Only as a targeted diagnostic; the default evidence does not justify broad hardware investment yet.

## Interpretation

The largest actual diagnostic gap is 0.0%. Repetition sweep maximum tested density was 128 repetitions/context. The pruning, support-confidence, and entropy-gating sweeps are deliberately small and expose storage/accuracy/speedup tradeoffs rather than selecting a flattering configuration.

Real traces are supported by `wam.traces.iter_addresses`. Capture representative programs externally with Valgrind/Lackey, Pin, DynamoRIO, or perf, convert to one byte address per line, and run `python -m wam.diagnostics --trace path/to/trace.txt`. Priority workloads are linked-list traversal, tree/graph traversal, hash tables, sorting, dynamic programming, pointer chasing, and SQLite queries.

## Artifacts

`context_depth.csv`, `markov_comparison.csv`, `repetition_sweep.csv`, `trace_length_sweep.csv`, `pruning.csv`, `support_confidence.csv`, `entropy_policy.csv`, and `plots/`.
