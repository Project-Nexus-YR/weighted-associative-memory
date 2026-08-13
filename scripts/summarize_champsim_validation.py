#!/usr/bin/env python3
"""Summarize ChampSim smoke output without implying paper-quality evidence."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "results/champsim_validation"
SMOKE = VALIDATION / "smoke"
PREDICTORS = ("baseline", "wam_h16", "spp_dev")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def smoke_metrics(name: str) -> dict[str, object]:
    data = json.loads((SMOKE / f"{name}.json").read_text(encoding="utf-8"))[0]["roi"]
    core = data["cores"][0]
    l2 = data["cpu0_L2C"]
    rfo = l2["RFO"]
    instructions = int(core["instructions"])
    cycles = int(core["cycles"])
    return {
        "predictor": name,
        "workload": "synthetic_smoke",
        "status": "smoke_only",
        "paper_quality_trace": False,
        "instructions": instructions,
        "cycles": cycles,
        "ipc": instructions / cycles if cycles else 0.0,
        "l2_rfo_misses": int(rfo["miss"][0]),
        "l2_mpki": 1000.0 * int(rfo["miss"][0]) / instructions if instructions else 0.0,
        "prefetch_issued": int(l2["prefetch issued"]),
        "useful_prefetch": int(l2["useful prefetch"]),
        "useless_prefetch": int(l2["useless prefetch"]),
    }


def main() -> None:
    VALIDATION.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((SMOKE / "run_metadata.json").read_text(encoding="utf-8"))
    trace = Path(metadata["trace"])
    write_csv(
        VALIDATION / "trace_manifest.csv",
        ["workload", "trace", "format", "has_pcs", "paper_quality_trace", "status", "notes"],
        [{"workload": "synthetic_smoke", "trace": str(trace), "format": "ChampSim input_instr native 64-byte", "has_pcs": True, "paper_quality_trace": False, "status": "smoke_completed", "notes": "deterministic format/build check only; no legal public workload claim"}],
    )
    rows = [smoke_metrics(name) for name in PREDICTORS]
    write_csv(
        VALIDATION / "per_workload.csv",
        list(rows[0]),
        rows,
    )
    write_csv(VALIDATION / "aggregate.csv", ["scope", "predictor", "workloads", "paper_quality_trace", "status", "mean_ipc", "mean_l2_mpki"], [{"scope": "smoke", "predictor": row["predictor"], "workloads": 1, "paper_quality_trace": False, "status": "not_for_claim", "mean_ipc": row["ipc"], "mean_l2_mpki": row["l2_mpki"]} for row in rows])
    write_csv(VALIDATION / "hybrid.csv", ["scope", "selector", "status", "paper_quality_trace", "note"], [{"scope": "all", "selector": "hybrid_oracle", "status": "not_run", "paper_quality_trace": False, "note": "requires paper-quality traces and at least two evaluated predictor streams"}])
    write_csv(VALIDATION / "disagreement.csv", ["scope", "predictor_a", "predictor_b", "status", "paper_quality_trace", "note"], [{"scope": "all", "predictor_a": "wam_h16", "predictor_b": "spp_dev", "status": "not_run", "paper_quality_trace": False, "note": "smoke trace has no meaningful predictor disagreement"}])
    write_csv(VALIDATION / "timeliness.csv", ["scope", "predictor", "status", "paper_quality_trace", "note"], [{"scope": "all", "predictor": "wam_h16", "status": "not_run", "paper_quality_trace": False, "note": "timeliness requires paper-quality workload traces with demand/prefetch event analysis"}])
    write_csv(
        VALIDATION / "predictor_storage.csv",
        ["predictor", "table_entries", "entry_bytes", "metadata_table_bytes", "bounded_queue_entries", "declared_fixed_state_bytes", "status", "note"],
        [
            {"predictor": "baseline", "table_entries": 0, "entry_bytes": 0, "metadata_table_bytes": 0, "bounded_queue_entries": 0, "declared_fixed_state_bytes": 0, "status": "native_baseline", "note": "no prefetcher state"},
            {"predictor": "wam_h16", "table_entries": 256, "entry_bytes": 32, "metadata_table_bytes": 8192, "bounded_queue_entries": 16, "declared_fixed_state_bytes": 8448, "status": "compiled", "note": "8 KiB table plus fixed delayed-training queue accounting"},
            {"predictor": "spp_dev", "table_entries": "n/a", "entry_bytes": "n/a", "metadata_table_bytes": "n/a", "bounded_queue_entries": "n/a", "declared_fixed_state_bytes": "n/a", "status": "native_module", "note": "not audited into the WAM fixed-budget ledger"},
        ],
    )
    (VALIDATION / "report.md").write_text(
        """# ChampSim validation status

## Scope

The pinned ChampSim checkout, unmodified no-prefetch baseline, DirectWAM-H16 module, and native `spp_dev` module all built and completed a deterministic 10,000-instruction smoke run after a 1,000-instruction warmup. The three smoke JSON files and raw stdout/stderr are under `smoke/`.

## Evidence boundary

The smoke trace is synthetic and exists only to verify the native instruction-record format, PC transport, configuration wiring, executable launch, and result capture. It is explicitly marked `paper_quality_trace=False` in every summary artifact. It must not be used to support a workload-speedup or predictor-quality claim. No legally obtained public ChampSim-compatible real trace was available in this phase, so the primary trace-based evaluation remains blocked.

## Smoke outcome

All three predictors produced valid ChampSim JSON. On this deliberately simple smoke input, the reported ROI metrics are identical: 10,000 instructions, 10,988 cycles, IPC 0.9101, and 597 L2 RFO misses. WAM-H16 reported zero predictions/issued prefetches and zero delayed updates because the smoke input generated only two qualifying L2 callbacks; this is a format/build result, not a negative scientific result.

## Decision

Classification: E — platform and integration validated, primary trace-based evaluation not yet complete. Do not start RTL implementation or claim paper readiness. The next gating action is to add legally obtained, ChampSim-compatible instruction traces containing PCs and real load/store behavior, then run the preregistered H8/H16/H32, 8/16/32/64 KiB, baseline, SPP, VLDP/GMC-faithful, hybrid, oracle, timeliness, traffic, and pollution matrix.

## Reproducibility

Run `python3 scripts/build_champsim.py`, then `python3 scripts/generate_champsim_smoke_trace.py --output results/champsim_validation/smoke/smoke_input.champsimtrace --instructions 20000`, `python3 scripts/run_champsim_eval.py ...`, and `python3 scripts/summarize_champsim_validation.py`.
""",
        encoding="utf-8",
    )
    print(json.dumps({"status": "summarized", "paper_quality_trace": False, "predictors": list(PREDICTORS)}, indent=2))


if __name__ == "__main__":
    main()
