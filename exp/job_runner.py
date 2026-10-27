"""
Unified experiment job launcher for STL/MTL train, compile, and load sweeps.

Presets preserve the region/method grids from the former stl_*/mtl_* scripts.

Written by Will Solow, 2025
"""

from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

import numpy as np

from assets import CROP_NAMES


def run_module(job: list) -> str:
    mod, args = job
    print(f"\n=== Starting {mod} {' '.join(args)} ===", flush=True)
    process = subprocess.Popen(
        ["python", "-m", mod, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in process.stdout:
        print(f"[{mod}] {line}", end="")
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, mod)
    print(f"=== Finished {mod} ===\n", flush=True)
    return mod


def run_module_quiet(job: list) -> str:
    """Load-results style: suppress stderr, only announce successes."""
    mod, args = job
    process = subprocess.Popen(
        ["python", "-m", mod, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    if process.stdout:
        for line in process.stdout:
            print(line, end="")
    process.wait()
    if process.returncode == 0:
        print(f" {' '.join(args)} \n", flush=True)
    return mod


def get_subfolders(path: str) -> list[str]:
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]


def walk_geo(dpath: str, regions: list[str]):
    """Yield (region, station, site) triples under dpath."""
    for region in regions:
        stations = get_subfolders(f"{dpath}/{region}/")
        for station in stations:
            sites = get_subfolders(f"{dpath}/{region}/{station}")
            for site in sites:
                yield region, station, site


def run_pool(jobs: list, max_workers: int = 2, runner: Callable = run_module) -> None:
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(runner, job) for job in jobs]
        for fut in as_completed(futures):
            try:
                fut.result()
            except subprocess.CalledProcessError as e:
                print(f" {e.cmd} exited with code {e.returncode}", flush=True)


# ---------------------------------------------------------------------------
# Job builders (presets)
# ---------------------------------------------------------------------------


def _jobs_stl_train(file: str, config: str, dpath: str, dtype: str, seeds: int, regions: list[str]) -> list:
    """Full STL hierarchy: every site + station-All + region-All + All, per cultivar/seed."""
    jobs = []
    cultivars = np.append(CROP_NAMES[dtype], ["All"], axis=0)
    for seed in range(seeds):
        for cult in cultivars:
            for region in regions:
                stations = get_subfolders(f"{dpath}/{region}/")
                for station in stations:
                    sites = get_subfolders(f"{dpath}/{region}/{station}")
                    for site in sites:
                        jobs.append(
                            (
                                file,
                                [
                                    "--config",
                                    config,
                                    "--region",
                                    region,
                                    "--station",
                                    station,
                                    "--site",
                                    site,
                                    "--cultivar",
                                    cult,
                                    "--seed",
                                    str(seed),
                                ],
                            )
                        )
                    jobs.append(
                        (
                            file,
                            [
                                "--config",
                                config,
                                "--region",
                                region,
                                "--station",
                                station,
                                "--site",
                                "All",
                                "--cultivar",
                                cult,
                                "--seed",
                                str(seed),
                            ],
                        )
                    )
                jobs.append(
                    (
                        file,
                        [
                            "--config",
                            config,
                            "--region",
                            region,
                            "--station",
                            "All",
                            "--site",
                            "All",
                            "--cultivar",
                            cult,
                            "--seed",
                            str(seed),
                        ],
                    )
                )
            jobs.append(
                (
                    file,
                    [
                        "--config",
                        config,
                        "--region",
                        "All",
                        "--station",
                        "All",
                        "--site",
                        "All",
                        "--cultivar",
                        cult,
                        "--seed",
                        str(seed),
                    ],
                )
            )
    return jobs


def _jobs_mtl_train_sites(file: str, configs: list[str], dpath: str, seeds: int, regions: list[str]) -> list:
    """MTL site-level train (cultivar=All); matches former mtl_exp.py active grid."""
    jobs = []
    for c in configs:
        for seed in range(seeds):
            for region, station, site in walk_geo(dpath, regions):
                jobs.append(
                    (
                        file,
                        [
                            "--config",
                            c,
                            "--region",
                            region,
                            "--station",
                            station,
                            "--site",
                            site,
                            "--cultivar",
                            "All",
                            "--seed",
                            str(seed),
                        ],
                    )
                )
    return jobs


def _jobs_stl_compile(file: str, dpath: str, dtype: str, regions: list[str], run_root: str) -> list:
    jobs = []
    cultivars = np.append(CROP_NAMES[dtype], ["All"], axis=0)
    for cult in cultivars:
        for region, station, site in walk_geo(dpath, regions):
            jobs.append(
                (
                    file,
                    ["--start_dir", f"{run_root}/{region}/{station}/{site}/{cult}/"],
                )
            )
    return jobs


def _jobs_mtl_compile(file: str, dpath: str, regions: list[str], run_roots: list[str]) -> list:
    jobs = []
    for region, station, site in walk_geo(dpath, regions):
        for root in run_roots:
            jobs.append((file, ["--start_dir", f"{root}/{region}/{station}/{site}/"]))
    return jobs


def _jobs_stl_load(file: str, dpath: str, dtype: str, regions: list[str], run_root: str) -> list:
    return _jobs_stl_compile(file, dpath, dtype, regions, run_root)


def _jobs_mtl_load(file: str, dpath: str, regions: list[str], run_roots: list[str]) -> list:
    """Former mtl_load_results passed --config <run_dir> --stl."""
    jobs = []
    for region, station, site in walk_geo(dpath, regions):
        for root in run_roots:
            jobs.append(
                (
                    file,
                    ["--config", f"{root}/{region}/{station}/{site}/", "--stl"],
                )
            )
    return jobs


PRESETS: dict[str, dict] = {
    "stl_mthl_gd": {
        "mode": "train",
        "flavor": "stl",
        "file": "trainers.train_model",
        "config": "mthl/ch_gd",
        "dpath": "_data/processed_data/grape_coldhardiness",
        "dtype": "grape_coldhardiness_reg",
        "regions": ["MI", "NS", "NY"],
        "seeds": 5,
        "builder": "stl_train",
    },
    "mtl_bcov_deep": {
        "mode": "train",
        "flavor": "mtl",
        "file": "trainers.train_model",
        "config": "mthl/ch_deep",
        "dpath": "_data/processed_data/grape_coldhardiness",
        "dtype": "grape_coldhardiness_reg",
        "regions": ["MI", "NS", "NY"],
        "seeds": 5,
        "builder": "mtl_train",
        "configs": ["mthl/ch_deep"],
    },
    "stl_compile_mthl_gd": {
        "mode": "compile",
        "flavor": "stl",
        "file": "results.compile_runs",
        "dpath": "_data/processed_data/grape_coldhardiness",
        "dtype": "grape_coldhardiness_reg",
        "regions": ["NS", "NY"],
        "run_root": "_runs/MTHL/GrapeColdhardiness/GradientDescent",
        "builder": "stl_compile",
    },
    "mtl_compile_bcov": {
        "mode": "compile",
        "flavor": "mtl",
        "file": "results.compile_runs",
        "dpath": "_data/processed_data/grape_coldhardiness",
        "dtype": "grape_coldhardiness_reg",
        "regions": ["BCOV", "ONNP", "WA"],
        "run_roots": [
            "_runs/MTLHierarchy/GrapeColdhardiness/DeepMTL",
            "_runs/MTLHierarchy/GrapeColdhardiness/ParamMTL",
        ],
        "builder": "mtl_compile",
    },
    "stl_load_mthl_gd": {
        "mode": "load",
        "flavor": "stl",
        "file": "results.load_results",
        "dpath": "_data/processed_data/grape_coldhardiness",
        "dtype": "grape_coldhardiness_reg",
        "regions": ["NY", "NS"],
        "run_root": "_runs/MTHL/GrapeColdhardiness/GradientDescent",
        "builder": "stl_load",
    },
    "mtl_load_bcov": {
        "mode": "load",
        "flavor": "mtl",
        "file": "results.load_results",
        "dpath": "_data/processed_data/grape_coldhardiness",
        "dtype": "grape_coldhardiness_reg",
        "regions": ["BCOV", "ONNP", "WA"],
        "run_roots": [
            "_runs/MTLHierarchy/GrapeColdhardiness/DeepMTL",
            "_runs/MTLHierarchy/GrapeColdhardiness/ParamMTL",
        ],
        "builder": "mtl_load",
    },
}

# Map legacy module entry points → preset names
LEGACY_PRESETS = {
    "stl_exp": "stl_mthl_gd",
    "mtl_exp": "mtl_bcov_deep",
    "stl_results": "stl_compile_mthl_gd",
    "mtl_results": "mtl_compile_bcov",
    "stl_load_results": "stl_load_mthl_gd",
    "mtl_load_results": "mtl_load_bcov",
}


def build_jobs(preset: dict, overrides: argparse.Namespace) -> tuple[list, Callable]:
    file = overrides.file or preset.get("file")
    dpath = overrides.dpath or preset.get("dpath")
    dtype = overrides.dtype or preset.get("dtype")
    regions = overrides.regions or preset.get("regions")
    seeds = overrides.seeds if overrides.seeds is not None else preset.get("seeds", 5)
    config = overrides.config or preset.get("config")
    builder = preset["builder"]

    if builder == "stl_train":
        return _jobs_stl_train(file, config, dpath, dtype, seeds, regions), run_module
    if builder == "mtl_train":
        configs = [config] if overrides.config else preset.get("configs", [config])
        return _jobs_mtl_train_sites(file, configs, dpath, seeds, regions), run_module
    if builder == "stl_compile":
        return _jobs_stl_compile(file, dpath, dtype, regions, preset["run_root"]), run_module
    if builder == "mtl_compile":
        return _jobs_mtl_compile(file, dpath, regions, preset["run_roots"]), run_module
    if builder == "stl_load":
        return _jobs_stl_load(file, dpath, dtype, regions, preset["run_root"]), run_module_quiet
    if builder == "mtl_load":
        return _jobs_mtl_load(file, dpath, regions, preset["run_roots"]), run_module_quiet
    raise ValueError(f"Unknown builder `{builder}`")


def main(preset: str | None = None, argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Unified STL/MTL experiment job runner")
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=sorted(PRESETS.keys()),
        help="Named experiment recipe (region/method grid)",
    )
    parser.add_argument("--mode", type=str, choices=["train", "compile", "load"], default=None)
    parser.add_argument("--flavor", type=str, choices=["stl", "mtl"], default=None)
    parser.add_argument("--file", type=str, default=None, help="Python module to invoke")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dpath", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--regions", type=str, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)

    args = parser.parse_args(argv)

    preset_name = preset or args.preset
    if preset_name is None:
        # Infer from mode+flavor if a single matching preset exists, else require --preset
        if args.mode and args.flavor:
            matches = [k for k, v in PRESETS.items() if v["mode"] == args.mode and v["flavor"] == args.flavor]
            if len(matches) == 1:
                preset_name = matches[0]
            elif matches:
                parser.error(f"Ambiguous mode/flavor; pick --preset from {matches}")
            else:
                parser.error(f"No preset for mode={args.mode} flavor={args.flavor}")
        else:
            parser.error("Provide --preset (or --mode and --flavor)")

    p = dict(PRESETS[preset_name])
    jobs, runner = build_jobs(p, args)
    print(f"Running preset `{preset_name}` with {len(jobs)} jobs", flush=True)
    run_pool(jobs, max_workers=args.workers, runner=runner)


if __name__ == "__main__":
    main()
