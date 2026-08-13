#!/usr/bin/env python3
"""Run the fixed native ChampSim real-trace matrix.

The trace files are intentionally outside git. This runner records every
command, return code, elapsed time, and raw stdout/stderr beside the JSON
statistics so a partial or resumed run remains auditable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/champsim_real_eval"
PINNED_COMMIT = "51588e1d6f97875fe8de1a3621d28668bff83fcf"

PREDICTORS = {
    "NoPrefetch": "champsim",
    "NativeSPP": "champsim_spp_dev",
    "NativeIPStride": "champsim_ip_stride",
    "WAM-H16": "champsim_wam_h16",
    "Hybrid-SPP+WAM": "champsim_hybrid_spp_wam",
}


def manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missing = [row["trace_name"] for row in rows if row["download_status"] != "verified_local"]
    if missing:
        raise SystemExit(f"manifest contains unavailable or unverified traces: {', '.join(missing)}")
    return rows


def run_one(job: dict[str, object]) -> dict[str, object]:
    output_dir = Path(str(job["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "stats.json"
    stdout_path = output_dir / "stdout.txt"
    stderr_path = output_dir / "stderr.txt"
    metadata_path = output_dir / "run.json"
    command = [str(job["executable"]), "--hide-heartbeat", "--warmup-instructions", str(job["warmup"]), "--simulation-instructions", str(job["simulation"]), "--json", str(json_path), str(job["trace"])]
    started = time.time()
    status = "completed"
    returncode = 0
    stdout = ""
    stderr = ""
    try:
        completed = subprocess.run(command, cwd=str(job["champsim_root"]), text=True, capture_output=True, check=False)
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        if returncode != 0 or not json_path.exists():
            status = "failed"
    except OSError as exc:
        status = "failed"
        returncode = -1
        stderr = f"{type(exc).__name__}: {exc}\n"
    elapsed = time.time() - started
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    result = {
        "trace_name": job["trace_name"],
        "predictor": job["predictor"],
        "trace": str(job["trace"]),
        "executable": str(job["executable"]),
        "command": command,
        "warmup_instructions": job["warmup"],
        "simulation_instructions": job["simulation"],
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "status": status,
        "json_path": str(json_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    metadata_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--champsim-root", type=Path, default=ROOT / "external/champsim")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT / "trace_manifest.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmup", type=int, default=5_000_000)
    parser.add_argument("--simulation", type=int, default=10_000_000)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    champsim_root = args.champsim_root.resolve()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=champsim_root, text=True).strip()
    if actual != PINNED_COMMIT:
        raise SystemExit(f"expected ChampSim {PINNED_COMMIT}, found {actual}")
    if args.warmup < 1 or args.simulation < 1 or args.workers < 1:
        raise SystemExit("warmup, simulation, and workers must be positive")

    output = args.output.resolve()
    rows = manifest_rows(args.manifest.resolve())
    jobs: list[dict[str, object]] = []
    for row in rows:
        trace = Path(row["local_path"]).resolve()
        trace_stem = trace.name.removesuffix(".champsimtrace.xz")
        for predictor, binary in PREDICTORS.items():
            executable = champsim_root / "bin" / binary
            if not executable.exists():
                raise SystemExit(f"missing executable for {predictor}: {executable}")
            output_dir = output / "runs" / trace_stem / predictor.replace("+", "_")
            existing = output_dir / "run.json"
            if args.resume and existing.exists():
                try:
                    if json.loads(existing.read_text(encoding="utf-8")).get("status") == "completed":
                        continue
                except (OSError, json.JSONDecodeError):
                    pass
            jobs.append({"trace_name": row["trace_name"], "trace": trace, "predictor": predictor, "executable": executable, "output_dir": output_dir, "champsim_root": champsim_root, "warmup": args.warmup, "simulation": args.simulation})

    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{result['status']} {result['trace_name']} {result['predictor']} {float(result['elapsed_seconds']):.1f}s", flush=True)

    existing_results: list[dict[str, object]] = []
    runs_root = output / "runs"
    if runs_root.exists():
        for metadata in sorted(runs_root.glob("*/*/run.json")):
            try:
                existing_results.append(json.loads(metadata.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
    existing_results.sort(key=lambda item: (str(item.get("trace_name")), str(item.get("predictor"))))
    fields = ["trace_name", "predictor", "status", "returncode", "warmup_instructions", "simulation_instructions", "elapsed_seconds", "json_path", "stdout_path", "stderr_path", "executable", "command"]
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in existing_results:
            writer.writerow({field: json.dumps(item[field]) if field == "command" else item.get(field, "") for field in fields})
    summary = {"status": "completed" if existing_results and all(item.get("status") == "completed" for item in existing_results) else "partial_or_failed", "champsim_commit": actual, "warmup_instructions": args.warmup, "simulation_instructions": args.simulation, "workers": args.workers, "predictors": list(PREDICTORS), "traces": len(rows), "jobs_requested_this_invocation": len(jobs), "completed_records": len(existing_results)}
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
