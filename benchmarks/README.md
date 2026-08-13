# Reproducible native benchmark suite

These small C programs provide deterministic workloads for external memory
tracing. They print a checksum so compilers cannot discard the traversals.
They are benchmark generators, not traces: a trace is only considered real
after an external load-instrumentation tool captures it.

Build and run a smoke test:

```sh
./scripts/build_benchmarks.sh
./build/benchmarks/linked_list 4096
```

Workload labels are pointer-chasing (`linked_list`, `pointer_chase`), trees,
graphs, hash tables, sorting, dynamic-programming-like matrix access, and
sequential/stride negative controls. Dataset size is the first positional
argument. Seeds are fixed in source.
