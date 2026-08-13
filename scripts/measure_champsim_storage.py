#!/usr/bin/env python3
"""Measure compiled native module object sizes without changing the build."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--champsim-root", type=Path, default=ROOT / "external/champsim")
    args = parser.parse_args()
    champsim = args.champsim_root.resolve()
    code = r'''#include <cstddef>
#include <iostream>
#include "ip_stride.h"
#include "spp_dev.h"
#include "wam_h16.h"
#include "hybrid_spp_wam.h"
int main() {
  std::cout << "spp_dev=" << sizeof(spp_dev) << "\n";
  std::cout << "ip_stride=" << sizeof(ip_stride) << "\n";
  std::cout << "wam_h16=" << sizeof(wam_h16) << "\n";
  std::cout << "hybrid_spp_wam=" << sizeof(hybrid_spp_wam) << "\n";
}
'''
    include = champsim / "vcpkg_installed/arm64-osx/include"
    include_args = [
        f"-I{champsim / '.csconfig'}",
        f"-I{champsim / 'inc'}",
        f"-I{champsim / 'prefetcher/spp_dev'}",
        f"-I{champsim / 'prefetcher/ip_stride'}",
        f"-I{ROOT / 'champsim/prefetcher/wam_h16'}",
        f"-I{ROOT / 'champsim/prefetcher/hybrid_spp_wam'}",
        f"-I{include}",
    ]
    with tempfile.TemporaryDirectory(prefix="wam-storage-") as temp:
        executable = Path(temp) / "measure"
        command = ["clang++", "-std=c++17", *include_args, "-x", "c++", "-o", str(executable), "-"]
        completed = subprocess.run(command, input=code, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise SystemExit(completed.stderr)
        result = subprocess.check_output([str(executable)], text=True)
    values = {line.split("=", 1)[0]: int(line.split("=", 1)[1]) for line in result.splitlines() if "=" in line}
    print(json.dumps({"compiler": "clang++", "sizes": values}, indent=2))


if __name__ == "__main__":
    main()
