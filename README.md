# Weighted Associative Memory

Weighted Associative Memory (WAM) is an experimental software prototype for testing a predictive memory architecture. It asks whether a dynamically weighted, trie-like graph over memory-access sequences can predict future accesses accurately enough to hide cache/DRAM latency.

This is a research simulator, not a hardware design and not a claim of a novel invention.

## Motivation

Conventional cache logic primarily asks whether an address is already present. WAM adds a bounded sequence context:

```text
recent access prefix  ->  likely next access(es)

ROOT
 ├── A
 │    └── B [0.81]
 │         ├── C [0.93]
 │         └── X [0.07]
 └── D
      └── B [1.00]
```

The predictor stores fixed-size counters and normalized weights on outgoing edges. A lookup follows at most `context_depth` addresses and ranks only the matched node's small child set. There is no neural network, embedding, external service, or whole-trie scan.

## Architecture

```text
trace -> workload/pipeline -> predictor training
                           -> bounded context lookup -> top-K predictions
                                                       |
                                                       v
                          L1 LRU cache <- prefetch <- L2 <- DRAM
                               |
                               v
                         cycle accounting + metrics
```

The package is split into small modules:

* `wam.trie`: weighted prefix graph, frequency and exponential-moving-average updates, thresholded prediction, storage estimate.
* `wam.predictor`: weighted trie plus Markov-1 and next-line baselines.
* `wam.cache` and `wam.hierarchy`: LRU caches and configurable L1/L2/DRAM cycle accounting.
* `wam.simulator`: demand accesses, prefetch cost, duplicate/useful/unused prefetch accounting, cache pollution, bandwidth, prediction accuracy, and speedup.
* `wam.workloads`: sequential, repeating, branching, context-sensitive, and random deterministic traces.
* `wam.experiment`: comparison table and context-depth/threshold sweeps.
* `wam.visualization`: four optional matplotlib plots.

## Install and run

Python 3.10+ is required.

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m wam.experiment
python -m wam.experiment --length 2000 --plot-dir artifacts/plots
python -m wam.benchmark
```

The original MVP experiment has a deliberately simple hierarchy:

```text
L1:   64 entries, 4 cycles
L2:  256 entries, 12 cycles
DRAM: unlimited, 100 cycles
```

The original MVP simulator charges 1 cycle to issue a prefetch and 8 bandwidth bytes per non-duplicate address by default. The research simulator additionally models DRAM completion latency, outstanding-request limits, late arrivals, and configurable L1/L2/L3 destinations.

## Baselines and metrics

The experiment runner compares:

1. `None`: LRU hierarchy with no prefetching.
2. `NextLine`: `X -> X + 1`.
3. `Markov-1`: the most frequent next address for the current address.
4. `WeightedTrie`: a depth-2 weighted context trie with a 0.05 confidence threshold.

It reports total accesses, L1/L2 hit rates, average latency, top-1/top-K prediction accuracy, speedup, prefetch precision, and coverage. The simulator additionally records DRAM accesses, useful/unused/duplicate prefetches, incorrect predictions, bandwidth, prefetch-caused evictions, latency saved by useful prefetches, and the net benefit:

```text
net latency benefit = latency saved by useful prefetches
                     - cost of unused prefetches
```

The predictive system can therefore lose: low-confidence predictions consume cycles and cache capacity, and may evict useful lines.

## Comparative research benchmark

`python -m wam.benchmark` runs the serious comparative experiment. It replays identical traces against a no-prefetch L1/L2/L3/DRAM control, next-line prefetching, a confidence-based stride prefetcher, first-order Markov prediction, and weighted trie depths 1, 2, 3, 4, and 8.

The benchmark uses a chronological 70/30 split by default. Predictors are trained only on the first portion and frozen during evaluation. The same simulator also supports online learning (`learning=True`) and emits a learning curve. No trace shuffling is performed.

The research hierarchy defaults to 64-byte cache lines and configurable 4/12/40/150-cycle L1/L2/L3/DRAM parameters. Prefetches are outstanding requests rather than instantaneous cache inserts: they have a completion time, can arrive late, consume bounded outstanding-request slots, and can cause cache pollution. Predictor lookup/update cycles are included in effective cycles.

The command writes:

```text
results/
├── summary.csv                 # mean/std across trials
├── detailed_results.csv        # one row per workload/system/trial
├── sweep.csv                   # depth, threshold, EMA, top-K, destination sweeps
├── ablation.csv                # depth/threshold/EMA/no-prefetch ablations
├── learning_curve.csv          # online accuracy/latency over prefixes
├── break_even.csv              # accuracy vs speedup at two DRAM latencies
├── config.json
├── report.md                   # data-derived verdict and limitations
└── plots/                      # 11 matplotlib figures
```

Use a different trace length or a plain-text trace file:

```bash
python -m wam.benchmark --length 2000 --trials 10
python -m wam.benchmark --trace path/to/trace.txt
python -m wam.diagnostics --output results/diagnostics
```

Trace files contain one integer or hexadecimal byte address per line. Blank lines and `#` comments are accepted. `wam.traces.iter_addresses` is streaming, so large files need not be loaded unless an experiment explicitly requires a chronological split. Traces can be generated with Valgrind/Lackey, Intel Pin, DynamoRIO, or `perf` and converted to this one-address-per-line format; those tools are not test dependencies.

The generated report is deliberately allowed to conclude that WAM loses. In particular, sequential and constant-stride streams should favor conventional prefetchers, random streams should punish speculative state, and deeper contexts should pay storage and lookup costs unless the workload contains repeatable higher-order structure. The report identifies WAM wins/losses, the best depth and threshold, warm-up, storage, maximum speedup, geometric-mean speedup, break-even accuracy, and the next recommended experiment.

## Falsification diagnostics

`python -m wam.diagnostics` preserves the primary benchmark and writes a separate higher-order diagnostic set. It verifies depth-2 and depth-4 discrimination with regression tests, instruments exact/fallback/unseen context matches, measures context support and reuse, sweeps repetition density and trace lengths through one million accesses, compares WAM against flat Markov-N tables, and evaluates pruning, support-based confidence, entropy gating, and empirical context-oracle accuracy. Its `context_diagnostics.md` is a diagnostic conclusion, not an optimized headline result.

## Prediction-horizon analysis

`python -m wam.horizon_analysis` tests whether accurate higher-order predictions arrive early enough to overlap memory latency. It compares direct horizon WAM, recursive traversal, direct Markov-N, and perfect `Oracle-H1` through `Oracle-H32` configurations under the same hierarchy. It records lead time, slack, late/partial/fully hidden misses, compute gaps, DRAM latency, bandwidth limits, failure buckets, and long higher-order accuracy at 100K and 1M accesses. Results are isolated in `results/horizon_analysis/` so earlier evidence remains reproducible.

## Hardware-feasibility analysis

`python -m wam.hardware_feasibility` evaluates whether the horizon benefit survives an explicit predictor cost model. It separates lookup latency from issue interval, supports serial and overlapped/pipelined lookup, queues and port pressure, deferred/batched updates, fixed hash tables, integer counters, context signatures, prediction-result caches, fallback/candidate-selection costs, and a normalized energy proxy. The default run prioritizes `DirectWAM-H16` and repeats key sweeps at H8/H32 in the latency table.

The experiment writes a new `results/hardware_feasibility/` directory containing latency, throughput, overlap, architecture, storage-budget, counter-width, hash-collision, update, batching, energy, tolerance, and microarchitecture CSVs, a feasibility matrix, plots, `config.json`, and a data-derived `report.md`. `IdealWAM` is a zero-cost direct-WAM upper bound; `Oracle` is reported separately and is not treated as implementable hardware. Hash-table replacement is approximated by deterministic bucket aliasing, and energy values are normalized comparative units rather than silicon estimates.

## Context-sensitive experiment

The important motivating trace is:

```text
A, B, X, C, B, Y, A, B, X, C, B, Y, ...
```

`B -> ?` is ambiguous to a first-order predictor. A depth-2 trie can learn `(A,B) -> X` and `(C,B) -> Y`. The CLI's context-depth sweep writes `accuracy_vs_context_depth.png` and includes estimated nodes/edges/bytes so accuracy can be viewed alongside predictor growth.

## Example output

Exact values depend on trace length, cache configuration, and random seeds. A representative run has this shape:

```text
Workload     Predictor        L1 hit  L2 hit   Avg cyc    Top-1  Speedup   Prec.  Cover.
-----------------------------------------------------------------------------------------
Sequential   None                ...      ...      ...       ...    1.00x      ...      ...
Sequential   NextLine            ...      ...      ...       ...    >1.00x      ...      ...
Contextual   Markov-1             ...      ...      ...       ...    modest      ...      ...
Contextual   WeightedTrie         ...      ...      ...       ...    higher      ...      ...
Random       WeightedTrie         ...      ...      ...       ...    may fall below 1.00x ...
```

The ellipses are intentional: the runner is the source of truth for the current configuration rather than a hard-coded benchmark claim.

## Limitations and future hardware directions

This MVP models entries, not cache lines or bytes; assumes one outstanding operation at a time; does not model overlap, queues, coherence, virtual addresses, replacement policies beyond LRU, or real DRAM bandwidth. Prefetch cost is a configurable cycle charge rather than a detailed bus model. The predictor is trained on a prefix and evaluated on a later suffix in the experiment runner to avoid direct test-trace leakage.

Future work could use fixed-width saturating counters, quantized weights, bounded fan-out tables, compact child indices, parallel comparators, confidence decay, multi-step speculative traversal, and traces from real programs. Each should be evaluated against storage, lookup energy, bandwidth, pollution, and latency rather than accuracy alone.
