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
```

The default hierarchy is deliberately simple:

```text
L1:   64 entries, 4 cycles
L2:  256 entries, 12 cycles
DRAM: unlimited, 100 cycles
```

Prefetching costs 8 cycles and 8 bandwidth bytes per non-duplicate address by default. Prefetches are inserted into L1 by default; use the Python API to select L2.

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
