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
                if "ParamMTL" in subdir and exclude:
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
    pheno_mean = np.nanmean(pheno_array, axis=(1, 2, 3, 4, -1))[:,-1, -1]
    pheno_std = np.nanstd(pheno_array, axis=(1, 2, 3, 4, -1))[:,-1, -1]

    ones = np.array([4, 8, 12, 0])
    twos = ones + 2
    fives = ones + 3
    tens = ones + 1

    data = [pheno_mean[ones], pheno_mean[twos], pheno_mean[fives], pheno_mean[tens]]

    spacing = 2 
    pos= 0
    fig, ax = plt.subplots(figsize=(5, 3))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for group in data:
        for i,box in enumerate(group):
            ax.bar(pos, height=box, width=1, color=colors[i])
            pos += 1
        pos += spacing  # add group spacing

    ax.set_xticks([1.5, 7.5, 13.5, 19.5], labels=[1, 2, 5, 10])
    ax.set_xlabel("Per cultivar seasons of initial data")
    ax.set_ylabel("Root Mean Sqaured Error")
    legend_patches = [
        Patch(facecolor="tab:blue", label="Extra 1"),
        Patch(facecolor="tab:orange", label="Extra 2"),
        Patch(facecolor="tab:green", label="Extra 5"),
        Patch(facecolor="tab:red", label="Extra 10"),
    ]

    ax.legend(handles=legend_patches)
    ax.set_title("Cold-Hardiness RTMC with Additional Data")
    
    plt.savefig("plotters/figs/rtmc.png", bbox_inches="tight")
    plt.close()





if __name__ == "__main__":
    main()
