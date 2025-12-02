"""
window.py

Plot the impact of weather window on performance

Written by Will Solow, 2025
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from model_engine.util import CROP_NAMES
from plotters.plotting_functions import C_AGG, C_PER
from matplotlib.patches import Patch


def load_named_pickles(folder_paths: list[str], target_name: str, exclude: bool = False):
    """
    Load all pickle files matching a given name in all subdirectories.
    """
    results = {}

    for root in folder_paths:
        root = Path(f"./_runs/{root}")
        for pkl_file in root.rglob(target_name):
            try:
                # Get relative subdirectory name
                subdir = "/".join(pkl_file.parent.parts[-6:])
                if "DeepMTL" in subdir and exclude:
                    continue
                with open(pkl_file, "rb") as f:
                    results[subdir] = pickle.load(f)
            except Exception as e:
                pass

    return results


def main():

    pheno_models = load_named_pickles(["RTMC/FineTuneColdHardiness"], "results_per_cultivars.pkl", exclude=True)

    pheno_sorted_keys = np.argsort(list(pheno_models.keys()))
    pheno_array = np.array(list(pheno_models.values()))[pheno_sorted_keys]
    print(np.array(list(pheno_models.keys()))[pheno_sorted_keys])
    pheno_array[pheno_array == 0] = np.nan
    pheno_mean = np.nanmean(pheno_array, axis=(1, 2, 3, 4, -1))[:,-10:-1]
    pheno_std = np.nanstd(pheno_array, axis=(1, 2, 3, 4, -1))[:,-10:-1]

    data = pheno_mean[0]

    fig, ax = plt.subplots(figsize=(5, 3))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    x = np.arange(9)
    for i in range(data.shape[0]):
        if i < 4 or i == 8:
            continue
        ax.plot(data[i], marker='o',label=f"Day {30*i}-{30*(i+1)}")


    ax.set_xticks(x, labels=[0, 30, 60, 90, 120, 150, 180, 210, 240])
    ax.set_xlabel("Error Data up to Day k")
    ax.set_ylabel("Root Mean Sqaured Error")


    ax.set_title("Cold-Hardiness RTMC with in season data")
    ax.legend(loc="lower left")
    
    plt.savefig("plotters/figs/rtmc_day.png", bbox_inches="tight")
    plt.close()





if __name__ == "__main__":
    main()
