#!/usr/bin/env python3
"""Build pinned ChampSim baseline, DirectWAM-H16, and SPP smoke binaries."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINNED_COMMIT = "51588e1d6f97875fe8de1a3621d28668bff83fcf"
VCPKG_COMMIT = "6d7bf7ef2193e2d1c5798a5ff8811d533104c861"


def run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def capture(command: list[str], cwd: Path) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def ensure_makefile_patch(root: Path) -> None:
    makefile = root / "Makefile"
    text = makefile.read_text(encoding="utf-8")
    if "$(foreach build_id,$(build_ids),$(call get_base_objs,$(build_id))) $(base_module_objs)" in text:
        return
    patch = ROOT / "champsim/Makefile.test-deps.patch"
    run(["git", "apply", str(patch)], root)


def write_configs(validation: Path, baseline: dict) -> dict[str, Path]:
    configs = {
        "baseline": baseline,
        "wam_h16": {**baseline, "executable_name": "champsim_wam_h16", "L2C": {**baseline["L2C"], "prefetcher": "wam_h16"}},
        "spp_dev": {**baseline, "executable_name": "champsim_spp_dev", "L2C": {**baseline["L2C"], "prefetcher": "spp_dev"}},
    }
    paths = {}
    for name, config in configs.items():
        path = validation / f"{name}_config.json"
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        paths[name] = path
    return paths


def build_one(root: Path, config: Path, executable_name: str, include: Path, lib: Path, prefetcher_dir: Path | None) -> None:
    # ChampSim's generated environment is configuration-specific. Clean the
    # generated object/configuration graph between binaries so a prior target
    # cannot be linked with a different build hash.
    run(["make", "configclean", "SKIP_CHAMPSIM_TEST_DEPS=1"], root)
    configure = ["./config.sh"]
    if prefetcher_dir is not None:
        configure += ["--prefetcher-dir", str(prefetcher_dir)]
    configure += [str(config)]
    run(configure, root)
    run(
        [
            "make",
            "-j1",
            f"bin/{executable_name}",
            "SKIP_CHAMPSIM_TEST_DEPS=1",
            f"CPPFLAGS=-I.csconfig -I{include}",
            f"LDFLAGS=-L{lib} -L{lib / 'manual-link'}",
        ],
        root,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--champsim-root", type=Path, default=ROOT / "external/champsim")
    parser.add_argument("--config", type=Path, default=ROOT / "external/champsim/champsim_config.json")
    parser.add_argument("--validation-dir", type=Path, default=ROOT / "results/champsim_validation")
    args = parser.parse_args()
    root = args.champsim_root.resolve()
    if not (root / ".git").exists():
        raise SystemExit(f"ChampSim checkout not found: {root}")
    actual = capture(["git", "rev-parse", "HEAD"], root)
    if actual != PINNED_COMMIT:
        raise SystemExit(f"expected ChampSim {PINNED_COMMIT}, found {actual}")
    run(["git", "submodule", "update", "--init"], root)
    if capture(["git", "rev-parse", "HEAD"], root) != PINNED_COMMIT:
        raise SystemExit("ChampSim revision changed while initializing submodules")
    ensure_makefile_patch(root)
    vcpkg = root / "vcpkg/vcpkg"
    if not vcpkg.exists():
        run(["./vcpkg/bootstrap-vcpkg.sh", "-disableMetrics"], root)
    run(["./vcpkg/vcpkg", "install"], root)
    triplet = next((p for p in (root / "vcpkg_installed").iterdir() if p.is_dir() and p.name != "vcpkg"), None)
    if triplet is None:
        raise SystemExit("vcpkg installed triplet not found")
    # The pinned Makefile writes an absolute.options file before the package
    # triplet exists on a fresh checkout; regenerate it after vcpkg install so
    # -isystem receives a real include directory instead of consuming -MM.
    run(["make", "-B", "-rR", "absolute.options", f"TRIPLET_DIR={triplet}"], root)
    include = triplet / "include"
    lib = triplet / "lib"
    validation = args.validation_dir.resolve()
    validation.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(args.config.read_text(encoding="utf-8"))
    config_paths = write_configs(validation, baseline)
    prefetcher_dir = ROOT / "champsim/prefetcher"
    build_one(root, config_paths["baseline"], "champsim", include, lib, None)
    build_one(root, config_paths["wam_h16"], "champsim_wam_h16", include, lib, prefetcher_dir)
    build_one(root, config_paths["spp_dev"], "champsim_spp_dev", include, lib, None)
    environment = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": capture(["python3", "--version"], ROOT),
        "compiler": capture(["clang++", "--version"], ROOT).splitlines()[0],
        "champsim_commit": actual,
        "vcpkg_commit": capture(["git", "rev-parse", "HEAD"], root / "vcpkg"),
        "expected_vcpkg_commit": VCPKG_COMMIT,
        "source_config": str(args.config),
        "configs": {name: str(path) for name, path in config_paths.items()},
        "include": str(include),
        "library": str(lib),
        "build_flags": ["C++17", "-O3", "-Wall", "-Wextra", "-Wshadow", "-Wpedantic", "-Wconversion"],
        "executables": {name: str(root / "bin" / executable) for name, executable in {"baseline": "champsim", "wam_h16": "champsim_wam_h16", "spp_dev": "champsim_spp_dev"}.items()},
        "status": "built",
    }
    (validation / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(environment, indent=2))


if __name__ == "__main__":
    main()
