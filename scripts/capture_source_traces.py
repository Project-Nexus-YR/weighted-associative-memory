"""Capture actual benchmark loads/stores using source-level wrappers.

This is an honest fallback for hosts without Valgrind/Pin/DynamoRIO. It
records addresses passed through WAM_LOAD/WAM_STORE at the benchmark's actual
data accesses and labels outputs ``source_instrumented``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from convert_trace import convert


SIZES = {
    "linked_list": 65536,
    "pointer_chase": 65536,
    "binary_tree": 65536,
    "graph_bfs": 65536,
    "graph_dfs": 16384,
    "hash_table": 65536,
    "quicksort": 65536,
    "matrix_scan": 512,
    "stride": 524288,
}
RANDOMIZED = {"linked_list", "pointer_chase", "binary_tree", "graph_bfs", "graph_dfs", "hash_table", "quicksort"}


def count_records(path: Path) -> tuple[int, int, int, int]:
    total = loads = stores = 0
    lines: set[int] = set()
    for raw in path.open("r", encoding="utf-8"):
        tokens = raw.split()
        if len(tokens) < 2:
            continue
        kind, token = tokens[0], tokens[1].split(",", 1)[0]
        try:
            address = int(token, 0)
        except ValueError:
            continue
        total += 1
        loads += kind == "L"
        stores += kind == "S"
        lines.add(address // 64)
    return total, loads, stores, len(lines)


def capture_one(root: Path, output: Path, benchmark: str, size: int, seed: int, compiler: str) -> dict[str, object]:
    source = root / "benchmarks" / f"{benchmark}.c"
    build = root / "build" / "source_instrumented"
    build.mkdir(parents=True, exist_ok=True)
    binary = build / benchmark
    flags = [compiler, "-O2", "-std=c11", "-DWAM_SOURCE_TRACE", str(source), "-o", str(binary)]
    subprocess.run(flags, check=True, cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    raw = output / f"{benchmark}_size{size}_seed{seed}.raw"
    loads = output / f"{benchmark}_size{size}_seed{seed}_loads.trace"
    normalized_dir = output / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    normalized = normalized_dir / loads.name
    env = dict(os.environ)
    env["WAM_TRACE_OUT"] = str(raw)
    env["WAM_SEED"] = str(seed)
    subprocess.run([str(binary), str(size)], check=True, cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=300)
    # Keep an explicit load-only primary trace; stores remain available in the
    # raw file and metadata for a separate all-data experiment.
    with raw.open("r", encoding="utf-8") as source_handle, loads.open("w", encoding="utf-8") as load_handle:
        for line in source_handle:
            if line.startswith("L "):
                load_handle.write(line.split()[1] + "\n")
    kept, skipped = convert(loads, normalized)
    with loads.open("r", encoding="utf-8") as source_handle, normalized.open("w", encoding="utf-8") as normalized_handle:
        for line in source_handle:
            try:
                normalized_handle.write(f"0x{int(line.strip(), 0) // 64:x}\n")
            except ValueError:
                pass
    total, load_count, store_count, unique_lines = count_records(raw)
    metadata = {
        "benchmark": benchmark,
        "input_size": size,
        "seed": seed,
        "capture_method": "source_instrumented",
        "compiler": compiler,
        "compiler_flags": "-O2 -std=c11 -DWAM_SOURCE_TRACE",
        "trace_length": kept,
        "load_count": load_count,
        "store_count": store_count,
        "total_references_raw": total,
        "unique_cache_lines": unique_lines,
        "raw_trace_path": raw.name,
        "normalized_trace_path": str(Path("normalized") / normalized.name),
        "normalized_by": "cache_line_size=64; normalized file contains cache-line IDs; evaluator consumes raw byte addresses",
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip(),
        "host_os": platform.platform(),
        "architecture": platform.machine(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "converter_kept": kept,
        "converter_skipped": skipped,
    }
    loads.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("traces/source_instrumented/loads"))
    parser.add_argument("--benchmarks", nargs="*", default=list(SIZES))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--size", type=int)
    parser.add_argument("--compiler", default=os.environ.get("CC", "cc"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    rows = []
    for benchmark in args.benchmarks:
        if benchmark not in SIZES:
            raise SystemExit(f"unknown benchmark: {benchmark}")
        size = args.size or SIZES[benchmark]
        for seed in (seeds if benchmark in RANDOMIZED else seeds[:1]):
            rows.append(capture_one(root, args.output, benchmark, size, seed, args.compiler))
            print(f"captured {benchmark} size={size} seed={seed} loads={rows[-1]['load_count']}")
    (args.output / "capture_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
