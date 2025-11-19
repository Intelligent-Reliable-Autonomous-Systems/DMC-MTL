from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import argparse
import os
from model_engine.util import CROP_NAMES
import numpy as np


def run_module(job: list[str]) -> None:
    mod, args = job

    process = subprocess.Popen(
        ["python", "-m", mod, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,  # suppress errors
        text=True,
        bufsize=1,
    )

    if process.stdout:
        for line in process.stdout:
            print(line, end="")

    process.wait()

    # only print "Finished" if successful
    if process.returncode == 0:
        print(f" {' '.join(args)} \n", flush=True)

    # silently ignore failures
    return mod


def get_subfolders(path: str) -> list[str]:
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="plotters.load_results")
    parser.add_argument("--config", type=str, default="hierarchical_mtl/ch_gd")
    parser.add_argument("--dpath", type=str, default="_data/processed_data/grape_coldhardiness")
    parser.add_argument("--dtype", type=str, default="grape_coldhardiness_reg")

    args = parser.parse_args()

    jobs = []
    for region in ["BCOV", "ONNP", "WA"]:
        stations = get_subfolders(f"{args.dpath}/{region}/")
        for station in stations:
            sites = get_subfolders(f"{args.dpath}/{region}/{station}")
            for site in sites:
                jobs.append(
                    (
                        args.file,
                        [
                            "--config",
                            f"_runs/MTLHierarchy/GrapeColdhardiness/DeepMTL/{region}/{station}/{site}/",
                            "--stl"
                        ],
                    )
                )

                jobs.append(
                    (
                        args.file,
                        [
                            "--config",
                            f"_runs/MTLHierarchy/GrapeColdhardiness/ParamMTL/{region}/{station}/{site}/",
                            "--stl"
                        ],
                    )
                )


    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_module, job) for job in jobs]
        for fut in as_completed(futures):
            try:
                fut.result()
            except subprocess.CalledProcessError as e:
                print(f" {e.cmd} exited with code {e.returncode}", flush=True)


if __name__ == "__main__":
    main()
