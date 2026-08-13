#!/usr/bin/env python3
"""Create and verify the fixed public-trace manifest for the real evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_DIR = ROOT / "traces/champsim_public/spec2006"
OUTPUT = ROOT / "results/champsim_real_eval/trace_manifest.csv"
SOURCE = "https://zenodo.org/records/10959705"

TRACE_SPECS = [
    ("403.gcc-48B", "gcc", "control/compute", 69011360, "3854d99369d79df1bdbd25979b7902f6"),
    ("410.bwaves-945B", "bwaves", "irregular/memory-bound", 51500100, "a26dbb4baafd4c69ac3a7b67082fdd39"),
    ("416.gamess-875B", "gamess", "irregular/scientific", 103939692, "d3b7611d19dc8ece554e5e07b53f99a5"),
    ("429.mcf-217B", "mcf", "irregular/pointer-heavy", 220147452, "7efef8e0a78e35f826f90fa8ab2b3992"),
    ("434.zeusmp-10B", "zeusmp", "irregular/scientific", 151581576, "75d6872f86fec205b4aaf2a2aad5f3f4"),
    ("435.gromacs-111B", "gromacs", "regular/scientific", 213153172, "1d333a6a11ce8b9da0e885f6aaef4fc7"),
    ("447.dealII-3B", "dealII", "irregular/tree/index-like", 179356152, "3e896f58b42f8320ad2f49ed7323468e"),
    ("453.povray-576B", "povray", "control/compute", 68234140, "5cd8f7ffec0af9986fcffc668688f575"),
    ("454.calculix-104B", "calculix", "irregular/scientific", 36517968, "678dad7a69362d24a7a14566a7426327"),
    ("465.tonto-44B", "tonto", "control/compute", 106378076, "6287e65f57a6224f19b1261c91d0dc8d"),
]


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    trace_dir = args.trace_dir.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for stem, suite_name, workload_class, expected_size, expected_md5 in TRACE_SPECS:
        filename = f"{stem}.champsimtrace.xz"
        path = trace_dir / filename
        local = str(path) if path.exists() else ""
        status = "verified_local" if path.exists() and path.stat().st_size == expected_size and md5(path) == expected_md5 else ("size_or_checksum_mismatch" if path.exists() else "not_downloaded")
        rows.append({"trace_name": filename, "source": f"{SOURCE}/files/{filename}?download=1", "benchmark_suite": "SPEC CPU 2006", "workload_class": workload_class, "compressed_size": expected_size, "instruction_count_if_known": "unknown", "download_status": status, "checksum": expected_md5, "local_path": local, "notes": f"fixed representative selection; suite workload {suite_name}; native ChampSim input_instr trace"})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verified = sum(row["download_status"] == "verified_local" for row in rows)
    print(f"wrote {output} ({verified}/{len(rows)} verified local traces)")


if __name__ == "__main__":
    main()
