"""
per_stage_error.py

Plot the per-stage error of different phenology models
using box plots

Written by, Will Solow 2025
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from model_engine.util import CROP_NAMES
import sys

def load_named_pickles(biomodel: str, folder_paths: list[str], target_name: str, exclude_multi: bool = False):
    """
    Load all pickle files matching a given name in all subdirectories.
    """
    results = {}

    for root in folder_paths:
        root = Path(f"./_runs/PaperExperiments/{biomodel}/{root}")
        for pkl_file in root.rglob(target_name):
            try:
                # Get relative subdirectory name
                subdir = "/".join(pkl_file.parent.parts[-2:])
                if "All" in subdir and exclude_multi:
                    continue
                with open(pkl_file, "rb") as f:
                    results[subdir] = pickle.load(f)
            except Exception as e:
                pass

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--biomodel",
        type=str,
        default="Phenology",
        choices=["Phenology", "ColdHardiness", "wofost"],
    )
    args = parser.parse_args()

    # Load Models
    agg_cultivar_models = ["ParamMTL"]
    per_cultivar_models = ["STL"]
    mtl_model = load_named_pickles(args.biomodel, agg_cultivar_models, "results_per_cultivars.pkl")
    stl_model = load_named_pickles(
        args.biomodel, per_cultivar_models, "results_agg_cultivars.pkl", exclude_multi=True
    )
    sorted_keys = np.argsort(list(mtl_model.keys()))
    mtl_array = np.array(list(mtl_model.values()))
    sorted_keys = np.argsort(list(stl_model.keys()))

    stl_array = np.array(list(stl_model.values()))[sorted_keys][np.newaxis, :]

    # Compute avg RMSE
    print(mtl_array)
    print(mtl_array.shape)
    mtl_array = mtl_array[0,:,-1,:]
    stl_array = stl_array[0,:,-1,:]
    mtl_array = np.delete(mtl_array,1, axis=0) # delete cab franc
    stl_array = np.delete(stl_array,1, axis=0)

    # Per run error
    mask = mtl_array < stl_array
    perc_mean = np.mean(mask,axis=(0,-1))
    perc_std = np.std(mask,axis=(0,-1))

    avg_better = (stl_array - mtl_array)[mask]
    avg_worse = (mtl_array - stl_array)[~mask]

    print(f"Percent Improve: {perc_mean} +/- {perc_std}")
    print(f"Avg Improve: {np.mean(avg_better)} +/- {np.std(avg_better)}")
    print(f"Avg Unimprove: {np.mean(avg_worse)} +/- {np.std(avg_worse)}")

    mtl_array = np.mean(mtl_array,axis=-1)
    stl_array = np.mean(stl_array,axis=-1)

    mask = mtl_array < stl_array
    perc_mean = np.mean(mask)
    perc_std = np.std(mask)

    avg_better = (stl_array - mtl_array)[mask]
    avg_worse = (mtl_array - stl_array)[~mask]

    print(f"Percent Improve: {perc_mean} +/- {perc_std}")
    print(f"Avg Improve: {np.mean(avg_better)} +/- {np.std(avg_better)}")
    print(f"Avg Unimprove: {np.mean(avg_worse)} +/- {np.std(avg_worse)}")


if __name__ == "__main__":
    main()
