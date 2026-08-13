#!/usr/bin/env python3
"""Plot smoke-only ChampSim metrics with an explicit evidence boundary."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "results/champsim_validation"


def main() -> None:
    rows = list(csv.DictReader((VALIDATION / "per_workload.csv").open(encoding="utf-8")))
    names = [row["predictor"] for row in rows]
    ipc = [float(row["ipc"]) for row in rows]
    mpki = [float(row["l2_mpki"]) for row in rows]
    output = VALIDATION / "plots"
    output.mkdir(parents=True, exist_ok=True)
    colors = ["#777777", "#1f77b4", "#ff7f0e"]
    width, height = 900, 420
    max_ipc = max(ipc) or 1.0
    max_mpki = max(mpki) or 1.0
    bars = []
    for index, name in enumerate(names):
        x = 80 + index * 120
        bars.append(f'<rect x="{x}" y="{260 - 180 * ipc[index] / max_ipc:.1f}" width="70" height="{180 * ipc[index] / max_ipc:.1f}" fill="{colors[index]}"/><text x="{x + 35}" y="285" text-anchor="middle">{name}</text>')
    for index, name in enumerate(names):
        x = 530 + index * 120
        bars.append(f'<rect x="{x}" y="{260 - 180 * mpki[index] / max_mpki:.1f}" width="70" height="{180 * mpki[index] / max_mpki:.1f}" fill="{colors[index]}"/><text x="{x + 35}" y="285" text-anchor="middle">{name}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/><text x="450" y="30" text-anchor="middle" font-family="sans-serif" font-size="18">ChampSim smoke trace — synthetic format/build check only</text>
<line x1="50" y1="260" x2="430" y2="260" stroke="#333"/><line x1="500" y1="260" x2="880" y2="260" stroke="#333"/>
<text x="240" y="70" text-anchor="middle" font-family="sans-serif">IPC (not paper evidence)</text><text x="690" y="70" text-anchor="middle" font-family="sans-serif">L2 RFO MPKI (not paper evidence)</text>
{''.join(bars)}<text x="240" y="330" text-anchor="middle" font-family="sans-serif" font-size="12">10,000 instructions; 10,988 cycles</text><text x="690" y="330" text-anchor="middle" font-family="sans-serif" font-size="12">synthetic trace; do not use for claims</text></svg>'''
    path = output / "smoke_metrics.svg"
    path.write_text(svg, encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
