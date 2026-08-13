#!/usr/bin/env python3
"""Run the single authorized 4-way SetAssociativeWAM variant."""

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
PINNED_COMMIT = "51588e1d6f97875fe8de1a3621d28668bff83fcf"
DEFAULT_OUTPUT = ROOT / "results/set_associative_wam"
DEFAULT_RAW = Path("/tmp/set_associative_wam_raw")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    bad = [row["trace_name"] for row in rows if row["download_status"] != "verified_local"]
    if bad:
        raise SystemExit(f"unverified traces in manifest: {', '.join(bad)}")
    return rows


def run_one(job: dict[str, object]) -> dict[str, object]:
    output = Path(str(job["output"]))
    raw = Path(str(job["raw"]))
    output.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    event_path = raw / "events.bin"
    prediction_path = raw / "predictions.bin"
    json_path = output / "stats.json"
    stdout_path = output / "stdout.txt"
    stderr_path = output / "stderr.txt"
    command = [str(job["executable"]), "--hide-heartbeat", "--warmup-instructions", str(job["warmup"]), "--simulation-instructions", str(job["simulation"]), "--json", str(json_path), str(job["trace"])]
    environment = os.environ.copy()
    environment["WAM_DIAGNOSTIC_EVENT_PATH"] = str(event_path)
    environment["WAM_DIAGNOSTIC_PREDICTION_PATH"] = str(prediction_path)
    started = time.time()
    completed = subprocess.run(command, cwd=str(job["champsim_root"]), env=environment, text=True, capture_output=True, check=False)
    elapsed = time.time() - started
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    result = {
        "trace_name": job["trace_name"],
        "trace": str(job["trace"]),
        "warmup_instructions": job["warmup"],
        "simulation_instructions": job["simulation"],
        "returncode": completed.returncode,
        "status": "completed" if completed.returncode == 0 and json_path.exists() and event_path.exists() else "failed",
        "elapsed_seconds": elapsed,
        "json_path": str(json_path),
        "event_path": str(event_path),
        "prediction_path": str(prediction_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "executable": str(job["executable"]),
        "command": command,
    }
    (output / "run.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--champsim-root", type=Path, default=ROOT / "external/champsim")
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/champsim_real_eval/trace_manifest.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--executable", type=Path, default=ROOT / "external/champsim/bin/champsim_set_associative_wam")
    parser.add_argument("--warmup", type=int, default=5_000_000)
    parser.add_argument("--simulation", type=int, default=10_000_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    champsim_root = args.champsim_root.resolve()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=champsim_root, text=True).strip()
    if actual != PINNED_COMMIT:
        raise SystemExit(f"expected ChampSim {PINNED_COMMIT}, found {actual}")
    executable = args.executable.resolve()
    if not executable.exists():
        raise SystemExit(f"SetAssociativeWAM executable not found: {executable}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.manifest.resolve())
    jobs: list[dict[str, object]] = []
    for row in rows:
        trace_path = Path(row["local_path"]).resolve()
        stem = trace_path.name.removesuffix(".champsimtrace.xz")
        run_output = output / "runs" / stem
        run_raw = args.raw_root.resolve() / stem
        if args.resume and (run_output / "run.json").exists():
            try:
                if json.loads((run_output / "run.json").read_text(encoding="utf-8")).get("status") == "completed":
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        jobs.append({"trace_name": row["trace_name"], "trace": trace_path, "output": run_output, "raw": run_raw, "champsim_root": champsim_root, "executable": executable, "warmup": args.warmup, "simulation": args.simulation})
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            print(f"{result['status']} {result['trace_name']} {float(result['elapsed_seconds']):.1f}s", flush=True)

    all_results = []
    for metadata in sorted((output / "runs").glob("*/run.json")):
        try:
            all_results.append(json.loads(metadata.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    fields = ["trace_name", "status", "returncode", "warmup_instructions", "simulation_instructions", "elapsed_seconds", "json_path", "event_path", "prediction_path", "stdout_path", "stderr_path", "executable", "command"]
    with (output / "set_associative_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for result in all_results:
            writer.writerow({field: json.dumps(result[field]) if field == "command" else result.get(field, "") for field in fields})
    summary = {"status": "completed" if len(all_results) == len(rows) and all(result.get("status") == "completed" for result in all_results) else "partial_or_failed", "traces": len(rows), "completed": len(all_results), "warmup_instructions": args.warmup, "simulation_instructions": args.simulation, "event_definition": "every non-PREFETCH access callback received by the L2 WAM module", "raw_root": str(args.raw_root.resolve()), "champsim_commit": actual, "variant": "SetAssociativeWAM-4way-64sets-256total-entries"}
    (output / "set_associative_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
