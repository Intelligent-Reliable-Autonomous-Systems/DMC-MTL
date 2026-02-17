import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from assets import CROP_NAMES


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
    parser.add_argument("--file", type=str, default="results.load_results")
    parser.add_argument("--config", type=str, default="hierarchical_mtl/ch_gd")
    parser.add_argument("--dpath", type=str, default="_data/processed_data/grape_coldhardiness")
    parser.add_argument("--dtype", type=str, default="grape_coldhardiness_reg")

    args = parser.parse_args()

    jobs = []
    cultivars = CROP_NAMES[args.dtype]
    cultivars = np.append(cultivars, ["All"], axis=0)
    for cult in cultivars:
        for region in ["NY", "NS"]:
            stations = get_subfolders(f"{args.dpath}/{region}/")
            for station in stations:
                sites = get_subfolders(f"{args.dpath}/{region}/{station}")
                for site in sites:
                    jobs.append(
                        (
                            args.file,
                            [
                                "--config",
                                f"_runs/MTHL/GrapeColdhardiness/GradientDescent/{region}/{station}/{site}/{cult}/",
                                "--per",
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
