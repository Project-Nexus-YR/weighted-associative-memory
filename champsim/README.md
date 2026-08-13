# ChampSim validation integration

ChampSim is the primary validation platform for this phase because it is a
trace-based cache/memory simulator built for prefetcher comparisons. The
upstream source is kept outside the Python simulator under
`external/champsim/`; this directory contains only the integration layer,
configuration metadata, and reproducibility scripts.

## Pinned environment

- ChampSim: `51588e1d6f97875fe8de1a3621d28668bff83fcf`
- vcpkg submodule: `6d7bf7ef2193e2d1c5798a5ff8811d533104c861`
- compiler: Apple Clang 17.0.0 (`clang-1700.0.13.5`)
- target: `arm64-apple-darwin24.5.0`
- build: C++17, `-O3`, warnings enabled

The exact runtime configuration is written to
`results/champsim_validation/environment.json` and
`baseline_config.json`.

## DirectWAM-H16 implementation

`prefetcher/wam_h16/` is a fixed-footprint ChampSim L2 prefetcher. It uses:

- 256 entries of 32-byte metadata (`8 KiB` table),
- a four-line context signature,
- 4-bit saturating confidence,
- signed cache-line deltas,
- a 16-entry delayed-training queue.

At qualifying access `t`, the predictor records the current context. Only
when access `t+16` actually arrives does it train
`context_at_t -> line_at_t+16`. Thus H16 is 16 qualifying L2 cache accesses,
not 16 instructions or 16 cycles, and no future address is exposed early.
Predictions are direct target-line predictions; the implementation does not
walk a recursively generated path.

## Build and run

The wrapper below expects an existing ChampSim checkout at
`external/champsim`, or accepts `--champsim-root` explicitly:

```bash
python3 scripts/build_champsim.py --champsim-root external/champsim
python3 scripts/generate_champsim_smoke_trace.py --output /tmp/wam-smoke.champsimtrace --instructions 20000
python3 scripts/run_champsim_eval.py --champsim-root external/champsim --trace /tmp/wam-smoke.champsimtrace --output results/champsim_validation/smoke
```

The smoke trace is a deterministic instruction-level format check only. It is
not a paper-quality workload and is never included as evidence for the WAM
claim. The primary result remains blocked until legally obtained public
ChampSim-compatible traces containing PCs and actual instruction load/store
behavior are supplied.

SPP is available in the pinned ChampSim checkout as `spp_dev`; its source and
configuration are recorded as an available verified ChampSim baseline. The
current integration does not relabel the prior Python approximation as SPP.
VLDP and GMC-inspired implementations remain explicitly pending a faithful
ChampSim implementation or verified upstream module.
