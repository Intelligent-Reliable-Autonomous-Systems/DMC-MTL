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

def load_named_pickles(folder_paths: list[str], target_name: str, include: str=None):
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
                if not (include in subdir):
                    continue
                with open(pkl_file, "rb") as f:
                    results[subdir] = pickle.load(f)
            except Exception as e:
                pass

    return results

def main():

    pheno_deep_models = load_named_pickles(["PhenologyLimited"], "results_per_cultivars.pkl", include="Deep")
    pheno_param_models = load_named_pickles(["PhenologyLimited"], "results_per_cultivars.pkl", include="Param")
    ch_deep_models = load_named_pickles(["ColdHardinessLimited"], "results_per_cultivars.pkl", include="Deep")
    ch_param_models = load_named_pickles(["ColdHardinessLimited"], "results_per_cultivars.pkl", include="Param")

    ind_arr = np.array([0, 5, 8, 9, 1, 2])
    pheno_deep_sorted_keys = np.argsort(list(pheno_deep_models.keys())) 
    pheno_deep_array = np.array(list(pheno_deep_models.values()))[pheno_deep_sorted_keys][ind_arr]
    pheno_param_sorted_keys = np.argsort(list(pheno_param_models.keys())) 
    pheno_param_array = np.array(list(pheno_param_models.values()))[pheno_param_sorted_keys][ind_arr]
    
    ch_deep_sorted_keys = np.argsort(list(ch_deep_models.keys()))  
    ch_deep_array = np.array(list(ch_deep_models.values()))[ch_deep_sorted_keys][ind_arr]
    #ch_param_sorted_keys = np.argsort(list(ch_param_models.keys()))  
    #ch_param_array = np.array(list(ch_param_models.values()))[ch_param_sorted_keys][ind_arr]

    pheno_deep_array[pheno_deep_array == 0] = np.nan
    pheno_param_array[pheno_param_array == 0] = np.nan
    ch_deep_array[ch_deep_array == 0] = np.nan
    #ch_param_array[ch_param_array == 0] = np.nan

    pheno_deep_mean = np.nanmean(pheno_deep_array,axis=(-3, -1))[:,-1]
    pheno_deep_std = np.nanstd(pheno_deep_array,axis=(-3, -1))[:,-1]
    pheno_param_mean = np.nanmean(pheno_param_array,axis=(-3, -1))[:,-1]
    pheno_param_std = np.nanstd(pheno_param_array,axis=(-3, -1))[:,-1]

    ch_deep_mean  = np.nanmean(ch_deep_array,axis=(-3, -1))[:,-1]
    ch_deep_std  = np.nanstd(ch_deep_array,axis=(-3, -1))[:,-1]
    #ch_param_mean  = np.nanmean(ch_param_array,axis=(1,2,3,4, -1))[:,-1]
    #ch_param_std  = np.nanstd(ch_param_array,axis=(1,2,3,4, -1))[:,-1]

    fig, ax = plt.subplots(figsize=(5,3))
    ax1_color = "tab:blue"
    ax.plot(np.arange(6), pheno_deep_mean, color=ax1_color, marker="o", label="Pheno Reg. MTL")
    ax.plot(np.arange(6), pheno_param_mean, color=ax1_color, marker="s", label="Pheno Param MTL")
    ax.set_xlabel("Seasons of Data per Cultivar")
    ax.set_xticks(np.arange(6), labels=[1, 2, 3, 5, 10, 15])
    ax.set_ylabel("RMSE in Days (Phenology)", color=ax1_color)
    ax.tick_params(axis='y', labelcolor=ax1_color)

    ax2_color = "tab:red"
    ax2 = ax.twinx()
    ax2.plot(np.arange(6), ch_deep_mean, color=ax2_color, marker="o",label="CH Reg. MTL")
    #ax2.plot(np.arange(6), ch_param_mean, color=ax2_color, marker="s", label="CH DMC-MTL")
    ax2.set_ylabel(r"RMSE in $^\circ$C (Cold-Hardiness)", color=ax2_color)
    ax2.tick_params(axis='y', labelcolor=ax2_color)
    ax.set_title("Effect of Per Cultivar Data Availability")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    ax.legend(handles1 + handles2, labels1 + labels2, loc="lower left")

    plt.savefig("plotters/figs/data.png",bbox_inches="tight")
    plt.close()




if __name__ == "__main__":
    main()