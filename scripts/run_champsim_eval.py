#!/usr/bin/env python3
"""Run reproducible ChampSim smoke jobs and save one JSON per predictor."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINNED_COMMIT = "51588e1d6f97875fe8de1a3621d28668bff83fcf"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--champsim-root", type=Path, default=ROOT / "external/champsim")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/champsim_validation/smoke")
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--simulation", type=int, default=10_000)
    args = parser.parse_args()
    champsim = args.champsim_root.resolve()
    trace = args.trace.resolve()
    output = args.output.resolve()
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=champsim, text=True).strip() != PINNED_COMMIT:
        raise SystemExit("ChampSim checkout is not pinned to the recorded revision")
    if not trace.exists():
        raise SystemExit(f"trace not found: {trace}")
    output.mkdir(parents=True, exist_ok=True)
    executables = {"baseline": "champsim", "wam_h16": "champsim_wam_h16", "spp_dev": "champsim_spp_dev"}
    commands = {}
    for name, executable in executables.items():
        exe = champsim / "bin" / executable
        if not exe.exists():
            raise SystemExit(f"ChampSim executable missing: {exe}; run scripts/build_champsim.py first")
        json_output = output / f"{name}.json"
        command = [str(exe), "--hide-heartbeat", "--warmup-instructions", str(args.warmup), "--simulation-instructions", str(args.simulation), "--json", str(json_output), str(trace)]
        completed = subprocess.run(command, cwd=champsim, text=True, capture_output=True, check=True)
        (output / f"{name}.stdout").write_text(completed.stdout, encoding="utf-8")
        (output / f"{name}.stderr").write_text(completed.stderr, encoding="utf-8")
        commands[name] = command
    metadata = {"platform": "ChampSim", "revision": PINNED_COMMIT, "trace": str(trace), "warmup_instructions": args.warmup, "simulation_instructions": args.simulation, "commands": commands, "predictors": list(executables), "status": "smoke_completed", "paper_quality_trace": False, "reason": "deterministic synthetic instruction-level format smoke trace"}
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
