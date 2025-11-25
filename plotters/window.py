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

def load_named_pickles(folder_paths: list[str], target_name: str, exclude_multi: bool = False):
    """
    Load all pickle files matching a given name in all subdirectories.
    """
    results = {}

    for root in folder_paths:
        root = Path(f"./_runs/PaperExperiments/{root}")
        for pkl_file in root.rglob(target_name):
            try:
                # Get relative subdirectory name
                subdir = "/".join(pkl_file.parent.parts[-6:])
                if "All" in subdir and exclude_multi:
                    continue
                with open(pkl_file, "rb") as f:
                    results[subdir] = pickle.load(f)
            except Exception as e:
                pass

    return results

def main():

    pheno_models = load_named_pickles(["PhenologyWindow"], "results_per_cultivars.pkl")
    ch_models = load_named_pickles(["ColdHardinessWindow"], "results_per_cultivars.pkl")

    pheno_sorted_keys = np.argsort(list(pheno_models.keys())) 
    pheno_array = np.array(list(pheno_models.values()))[pheno_sorted_keys][np.array([0, 3, 1, 2, 4, 5])]

    ch_sorted_keys = np.argsort(list(ch_models.keys()))  
    ch_array = np.array(list(ch_models.values()))[ch_sorted_keys][np.array([0, 3, 1, 2, 4, 5])]

    pheno_mean = np.mean(pheno_array,axis=(-3, -1))[:,-1]
    pheno_std = np.std(pheno_array,axis=(-3, -1))[:,-1]
    ch_mean  = np.mean(ch_array,axis=(-3, -1))[:,-1]
    ch_std  = np.std(ch_array,axis=(-3, -1))[:,-1]

    fig, ax = plt.subplots(figsize=(5,3))
    ax1_color = "tab:blue"
    ax.plot(np.arange(6), pheno_mean, color=ax1_color, label="Phenology")
    ax.set_xlabel("Window Size")
    ax.set_xticks(np.arange(6), labels=[1, 5, 10, 20, 50, "All"])
    ax.set_ylabel("RMSE in Days (Phenology)", color=ax1_color)
    ax.tick_params(axis='y', labelcolor=ax1_color)

    ax2_color = "tab:red"
    ax2 = ax.twinx()
    ax2.plot(np.arange(6), ch_mean, color=ax2_color, label="Cold-Hardiness")
    ax2.set_ylabel(r"RMSE in $^\circ$C (Cold-Hardiness)", color=ax2_color)
    ax2.tick_params(axis='y', labelcolor=ax2_color)
    ax.set_title("Weather Window Size Effect on Error")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    ax.legend(handles1 + handles2, labels1 + labels2, loc="lower left")

    plt.savefig("plotters/figs/window.png",bbox_inches="tight")
    plt.close()




if __name__ == "__main__":
    main()