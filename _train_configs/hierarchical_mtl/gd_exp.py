
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import argparse
import os 
from model_engine.util import CROP_NAMES
import numpy as np


def run_module(job: list[str]) -> None:
    mod, args = job
    print(f"\n=== Starting {mod} {' '.join(args)} ===", flush=True)

    # Stream stdout/stderr directly
    process = subprocess.Popen(
        ["python", "-m", mod, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Print lines live
    for line in process.stdout:
        print(f"[{mod}] {line}", end="")

    process.wait()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, mod)

    print(f"=== Finished {mod} ===\n", flush=True)
    return mod

def get_subfolders(path:str) -> list[str]:
     return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="trainers.train_model")
    parser.add_argument("--config", type=str, default="hierarchical_mtl/ch_gd")
    parser.add_argument("--dpath", type=str, default="_data/processed_data/grape_coldhardiness")
    parser.add_argument("--dtype", type=str, default="grape_coldhardiness_reg")

    args = parser.parse_args()

    jobs = []
    cultivars = CROP_NAMES[args.dtype]
    cultivars = np.append(cultivars, ["All"], axis=0)
    for seed in range(5):
        for cult in cultivars:
            for region in ["BCOV", "ONNP", "WA"]:
                stations = get_subfolders(f"{args.dpath}/{region}/")
                for station in stations:
                    sites = get_subfolders(f"{args.dpath}/{region}/{station}")
                    for site in sites:
                        jobs.append((args.file, ["--config", args.config, "--region", region, "--station", station, "--site", site, "--cultivar", cult, "--seed", str(seed)]) )
                    jobs.append((args.file, ["--config", args.config, "--region", region, "--station", station, "--site", "All", "--cultivar", cult, "--seed", str(seed)]) )
                jobs.append((args.file, ["--config", args.config, "--region", region, "--station", "All", "--site", "All", "--cultivar", cult, "--seed", str(seed)]) )
            jobs.append((args.file, ["--config", args.config, "--region", "All", "--station", "All", "--site", "All", "--cultivar", cult, "--seed", str(seed)]) )
                


    
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_module, job) for job in jobs]
        for fut in as_completed(futures):
            try:
                fut.result()
            except subprocess.CalledProcessError as e:
                print(f" {e.cmd} exited with code {e.returncode}", flush=True)


if __name__ == "__main__":
    main()
