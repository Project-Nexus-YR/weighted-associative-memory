#!/usr/bin/env python3
"""Write the frozen ChampSim real-evaluation configuration and ledger."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/champsim_real_eval"
PINNED_COMMIT = "51588e1d6f97875fe8de1a3621d28668bff83fcf"
VCPKG_COMMIT = "6d7bf7ef2193e2d1c5798a5ff8811d533104c861"


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--config", type=Path, default=ROOT / "external/champsim/champsim_config.json")
    parser.add_argument("--warmup", type=int, default=5_000_000)
    parser.add_argument("--simulation", type=int, default=10_000_000)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = json.loads(args.config.read_text(encoding="utf-8"))
    champsim_root = ROOT / "external/champsim"
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=champsim_root, text=True).strip()
    vcpkg_actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=champsim_root / "vcpkg", text=True).strip()
    if actual != PINNED_COMMIT or vcpkg_actual != VCPKG_COMMIT:
        raise SystemExit(f"pinned dependency mismatch: ChampSim={actual}, vcpkg={vcpkg_actual}")
    predictors = {
        "NoPrefetch": {"binary": "champsim", "role": "negative_control", "state_bytes": 0},
        "NativeSPP": {"binary": "champsim_spp_dev", "role": "strong_native_candidate", "state_bytes": "native_unbounded_audit_separate"},
        "NativeIPStride": {"binary": "champsim_ip_stride", "role": "additional_native_candidate", "state_bytes": "native_unbounded_audit_separate"},
        "WAM-H16": {"binary": "champsim_wam_h16", "role": "primary_fixed_direct_wam", "state_bytes": 8448},
        "Hybrid-SPP+WAM": {"binary": "champsim_hybrid_spp_wam", "role": "cheap_online_selector", "state_bytes": "native_spp_plus_8448"},
    }
    config = {
        "evaluation": "ChampSim native real-trace evaluation",
        "created_on": "2026-08-13",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "champsim_commit": actual,
        "vcpkg_commit": vcpkg_actual,
        "base_config_path": str(args.config.resolve()),
        "base_config": base,
        "core_count": 1,
        "warmup_instructions": args.warmup,
        "simulation_instructions": args.simulation,
        "trace_source": "https://zenodo.org/records/10959705",
        "trace_format": "native ChampSim input_instr .champsimtrace.xz",
        "predictors": predictors,
        "primary_predictor": "WAM-H16",
        "strong_baseline_candidates": ["NativeSPP", "NativeIPStride"],
        "headline_budget_bytes": 32768,
        "budget_points_bytes": [16384, 32768, 65536],
        "wam_fixed_state_bytes": 8448,
        "storage_fairness": "WAM ledger is fixed and auditable; canonical native predictors are reported separately because their pre-existing module state is not redesigned or truncated for this experiment.",
        "memory_intensity_rule": "memory-bound if NoPrefetch baseline LLC demand MPKI >= 10; regular/irregular group comes from the frozen manifest class.",
        "regression_guard": "WAM or Hybrid regression is any IPC speedup below 0.98x versus NoPrefetch; headline success requires >=1.02x geomean and no regression worse than -2% on the target group.",
        "online_hybrid_rule": "Native SPP runs first. DirectWAM-H16 sidecar is enabled while recent primary usefulness is below 5% after a 64-demand minimum; the selector uses only observed callbacks in a 1024-demand bounded window.",
        "known_measurement_limits": [
            "ChampSim JSON exposes aggregate prefetch counts, not predicted-address streams; address-level overlap/disagreement and exact late/pollution counts are not claimed.",
            "The primary run is fixed and laptop-bounded at 5M warmup plus 10M simulation instructions per trace.",
            "One-million-instruction oracle windows and Direct-vs-RecursiveWAM are recorded as not run because no native RecursiveWAM implementation is present and standard JSON has no window event stream.",
        ],
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    storage_rows = [
        {"predictor": "NoPrefetch", "budget_point_bytes": 0, "declared_state_bytes": 0, "object_size_bytes": "0", "budget_fit": "yes", "accounting_basis": "no prefetcher state"},
        {"predictor": "WAM-H16", "budget_point_bytes": 16384, "declared_state_bytes": 8448, "object_size_bytes": "measure_with_compiler", "budget_fit": "yes", "accounting_basis": "256 x 32-byte entries + 16 x 16-byte pending contexts"},
        {"predictor": "WAM-H16", "budget_point_bytes": 32768, "declared_state_bytes": 8448, "object_size_bytes": "measure_with_compiler", "budget_fit": "yes", "accounting_basis": "256 x 32-byte entries + 16 x 16-byte pending contexts"},
        {"predictor": "WAM-H16", "budget_point_bytes": 65536, "declared_state_bytes": 8448, "object_size_bytes": "measure_with_compiler", "budget_fit": "yes", "accounting_basis": "256 x 32-byte entries + 16 x 16-byte pending contexts"},
        {"predictor": "NativeSPP", "budget_point_bytes": "canonical", "declared_state_bytes": "native_module", "object_size_bytes": "measure_with_compiler", "budget_fit": "separate", "accounting_basis": "canonical native comparator; no WAM redesign"},
        {"predictor": "NativeIPStride", "budget_point_bytes": "canonical", "declared_state_bytes": "native_module", "object_size_bytes": "measure_with_compiler", "budget_fit": "separate", "accounting_basis": "canonical native comparator; no WAM redesign"},
        {"predictor": "Hybrid-SPP+WAM", "budget_point_bytes": "canonical", "declared_state_bytes": "native_spp_plus_8448", "object_size_bytes": "measure_with_compiler", "budget_fit": "separate", "accounting_basis": "online sidecar composed with native SPP"},
    ]
    write_csv(output / "storage_accounting.csv", list(storage_rows[0]), storage_rows)
    print(json.dumps({"status": "frozen", "config": str(output / "config.json"), "storage": str(output / "storage_accounting.csv")}, indent=2))


if __name__ == "__main__":
    main()
