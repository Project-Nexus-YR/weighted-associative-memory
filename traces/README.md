# External traces

This directory intentionally contains no fabricated trace. Place externally
captured, data-only `.trace`, `.addr`, or `.txt` files here and run:

```sh
python3 -m wam.real_trace_evaluation --trace-dir traces --output results/real_trace_evaluation
```

See `scripts/README.md` for capture and conversion instructions.
