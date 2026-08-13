#!/usr/bin/env python3
"""Generate a deterministic instruction-level ChampSim smoke trace.

This is only a format/build smoke test. It is deliberately not used as a
paper-quality workload or included in the validation claim.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

RECORD = struct.Struct("@QBB2B4B2Q4Q")


def write_trace(path: Path, instructions: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for index in range(instructions):
            ip = 0x400000 + (index % 32) * 4
            data = 0x10000000 + ((index * 16) % 8192)
            if index % 17 == 0:
                data = 0x20000000 + ((index * 64) % 65536)
            # ChampSim's native record is: IP, branch flags, two destination
            # registers, four source registers, two destination addresses,
            # and four source addresses.
            record = RECORD.pack(ip, 0, 0, 0, 0, 0, 0, 0, 0, data, 0, 0, 0, 0, 0)
            handle.write(record)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instructions", type=int, default=20_000)
    args = parser.parse_args()
    if args.instructions < 64:
        raise SystemExit("instructions must be at least 64")
    write_trace(args.output, args.instructions)
    print(f"wrote {args.instructions} instructions ({RECORD.size}-byte records) to {args.output}")


if __name__ == "__main__":
    main()
