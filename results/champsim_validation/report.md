# ChampSim validation status

## Scope

The pinned ChampSim checkout, unmodified no-prefetch baseline, DirectWAM-H16 module, and native `spp_dev` module all built and completed a deterministic 10,000-instruction smoke run after a 1,000-instruction warmup. The three smoke JSON files and raw stdout/stderr are under `smoke/`.

## Evidence boundary

The smoke trace is synthetic and exists only to verify the native instruction-record format, PC transport, configuration wiring, executable launch, and result capture. It is explicitly marked `paper_quality_trace=False` in every summary artifact. It must not be used to support a workload-speedup or predictor-quality claim. No legally obtained public ChampSim-compatible real trace was available in this phase, so the primary trace-based evaluation remains blocked.

## Smoke outcome

All three predictors produced valid ChampSim JSON. On this deliberately simple smoke input, the reported ROI metrics are identical: 10,000 instructions, 10,988 cycles, IPC 0.9101, and 597 L2 RFO misses. WAM-H16 reported zero predictions/issued prefetches and zero delayed updates because the smoke input generated only two qualifying L2 callbacks; this is a format/build result, not a negative scientific result.

## Decision

Classification: E — platform and integration validated, primary trace-based evaluation not yet complete. Do not start RTL implementation or claim paper readiness. The next gating action is to add legally obtained, ChampSim-compatible instruction traces containing PCs and real load/store behavior, then run the preregistered H8/H16/H32, 8/16/32/64 KiB, baseline, SPP, VLDP/GMC-faithful, hybrid, oracle, timeliness, traffic, and pollution matrix.

## Reproducibility

Run `python3 scripts/build_champsim.py`, then `python3 scripts/generate_champsim_smoke_trace.py --output results/champsim_validation/smoke/smoke_input.champsimtrace --instructions 20000`, `python3 scripts/run_champsim_eval.py ...`, and `python3 scripts/summarize_champsim_validation.py`.
