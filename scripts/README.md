# Real-trace workflow

Run `./check_trace_tools.sh` first. Build the deterministic native programs
with `./build_benchmarks.sh`. A trace is not created by simply running a
program; it must come from a load-instrumentation tool.

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
