# Real-trace workflow

Run `./check_trace_tools.sh` first. Build the deterministic native programs
with `./build_benchmarks.sh`. A trace is not created by simply running a
program; it must come from a load-instrumentation tool.

On hosts without a usable binary tracer, the specified fallback captures real
allocated addresses at benchmark source load/store sites:

```sh
./scripts/capture_source_traces.sh
python3 -m wam.real_trace_evaluation \
  --trace-dir traces/source_instrumented/loads \
  --output results/real_trace_evaluation --max-accesses 5000 --seed 0
```

The capture script records 5 seeds for randomized benchmarks, keeps raw
load/store logs locally, emits load-only byte-address traces plus normalized
cache-line files, and writes per-trace metadata. Raw traces are ignored by
Git because they are large; the evaluation report and capture inventory are
the committed research artifacts. The main run above is explicitly labeled
`source_instrumented`, uses one representative seed, and caps analysis to a
chronological prefix; it is not binary-level tracing.

Examples after installing/configuring an external tool:

```sh
valgrind --tool=lackey --trace-mem=yes ./build/benchmarks/pointer_chase 65536 > raw.lackey
python3 scripts/convert_trace.py raw.lackey traces/pointer_chase.trace
python3 -m wam.real_trace_evaluation --trace-dir traces --output results/real_trace_evaluation
```

For Pin or DynamoRIO, use a load-only pintool/client and emit one explicit
hexadecimal data address per line; `convert_trace.py` can normalize common text
exports. Keep instruction records out of the main trace. The evaluator treats
each input file as data accesses only and documents that assumption in
`trace_metadata.csv` and `report.md`.
