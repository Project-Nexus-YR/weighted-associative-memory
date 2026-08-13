"""Convert common external tracer text into one hexadecimal data address per line.

This intentionally accepts only explicit address tokens. It does not infer
addresses from arbitrary program output, preventing accidental fabricated
traces. For Valgrind Lackey, pass lines such as `` L 0x...`` or ``S ...``;
instruction records and malformed lines are skipped and counted.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

HEX = re.compile(r"0x[0-9a-fA-F]+")


def convert(source: Path, destination: Path, data_only: bool = True) -> tuple[int, int]:
    kept = skipped = 0
    with source.open("r", encoding="utf-8", errors="replace") as src, destination.open("w", encoding="utf-8") as dst:
        for line in src:
            token = line.strip()
            if not token:
                continue
            # Lackey data records begin with L/S/M. Pin/DynamoRIO exports may
            # be plain address lines; those are accepted as explicit addresses.
            if data_only and token[0] in {"I", "i"}:
                skipped += 1
                continue
            match = HEX.search(token)
            if not match:
                skipped += 1
                continue
            dst.write(match.group(0).lower() + "\n")
            kept += 1
    return kept, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    kept, skipped = convert(args.source, args.destination)
    print(f"kept={kept} skipped={skipped} output={args.destination}")


if __name__ == "__main__":
    main()
