# Real-trace evaluation

## Status

**Blocked on external traces: no captured data-address traces were available in the environment.** The native benchmark suite was added and compiled, but no Valgrind/Lackey, Intel Pin, DynamoRIO, or perf load trace was available. No synthetic trace was substituted, so this phase makes no claim about real software.

## What is ready

The evaluator supports chronological 50/50, 70/30, and 80/20 splits; equal 2–64 KB budgets; direct H8/H16/H32 WAM; hashed context; recursive WAM; Markov-N; VLDP-style delta history; SPP-style recursive signatures; GMC-style multi-order deltas; stride/next-line; hybrid arbitration; miss-only training; horizon oracles; context reuse/entropy; phase windows; and cross-input files when supplied.

## Tooling limitation

`clang`/`gcc` were available. No captured real traces were found and no supported external tracer was installed. Running a benchmark binary alone is not a memory trace and was intentionally not counted as one. Use `scripts/capture_trace.sh`, `scripts/convert_trace.py`, and `python3 -m wam.real_trace_evaluation --trace-dir traces` after installing/configuring a tracer.

## Classification

**Not classified A–F yet.** The requested classification requires real-trace measurements; assigning A would incorrectly treat missing evidence as a negative result.

## Paper-readiness

0/10 evidence items can be marked true from this run because no real trace was evaluated.

## Requested final verdict fields

- Real workloads evaluated: 0
- Best WAM configuration: N/A
- Best prior-art-style baseline: N/A
- WAM geomean speedup on irregular workloads: N/A
- WAM geomean speedup overall: N/A
- Best real-workload speedup: N/A
- Worst regression: N/A
- Best storage budget: N/A
- Best prediction horizon: N/A
- Direct-vs-recursive advantage: N/A
- Fraction of workloads where WAM wins: N/A
- Hybrid geomean speedup: N/A
- Dominant success condition: N/A
- Dominant failure condition: Missing external traces, not predictor failure
- Paper-readiness score: 0/10
- Final classification: Not classified A–F
- Single most important next step: capture data-only traces with an external tracer

## Single most important next step

Capture at least one data-only trace each for pointer chasing, graph/tree/hash access, and sequential controls with an external load-instrumentation tool, then rerun this command.
