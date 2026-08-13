# GMC-style implementation audit

## Classification

**Simplified approximation**

The repository implementation is intentionally not presented as a faithful
reimplementation of the published GMC prefetcher. The implementation is
`wam.real_predictors.GMCStylePredictor`, which subclasses
`DeltaContextPredictor`.

## What the code actually does

| Concern | Repository implementation |
|---|---|
| Context representation | Cache-line addresses are converted to adjacent address deltas. |
| Delta/stride representation | Signed integer deltas; no page/region, PC, or instruction context. |
| Local/global history | One global stream is used. There is no per-PC local history and no separate global-history buffer. |
| Context orders | Orders 1 through 16 are stored; lookup tries the longest available suffix and falls back to shorter suffixes. |
| Confidence | Normalized frequency of the selected target delta in the matched table entry. |
| Prediction target | One cache-line address at a fixed horizon, computed as current line plus predicted delta. |
| Table organization | A bounded FIFO-evicted Python dictionary keyed by delta tuples; each key stores target-delta counters. |
| Fallback behavior | Longest matching suffix with non-empty transitions, then shorter suffixes; no explicit confidence gating or multi-table arbitration. |
| Hardware accounting | A rough byte budget and lookup/update cycle proxy; no GMC-specific table or metadata model. |

## High-level comparison

Published GMC work describes global-aware, multi-order context analysis with
local/global context signals and prediction structures intended to increase
coverage while preserving accuracy. This repository captures only the broad
intuition of multi-order delta-context prediction. It omits program-counter
context, the local/global organization, the published training/update policy,
table replacement details, and GMC-specific candidate/confidence arbitration.

Therefore all result files and plots use the name **GMC-style**, not GMC, and
the result cannot support a paper-level claim against the original design.

Reference: [Global-aware and multi-order context-based prefetching for
high-performance processors](https://journals.sagepub.com/doi/10.1177/1094342010394386).

## Consequence for interpretation

The complementarity experiment asks whether WAM adds value to this concrete
GMC-style approximation. A positive result would motivate a follow-up against
an independently reproduced GMC implementation; a negative result is already
useful evidence against investing in WAM tuning before baseline fidelity is
improved.
