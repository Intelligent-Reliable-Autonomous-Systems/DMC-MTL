"""
window.py

Plot the impact of weather window on performance

Written by Will Solow, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import utils


def main():

    ch_models_rtmc = utils.load_named_pickles(["RTMC/FineTuneColdHardiness"], "results_per_cultivars.pkl", include="Param")
    ch_models_dmc = utils.load_named_pickles(["RTMC/ColdHardiness"], "results_per_cultivars.pkl", include="Param")
    
    param_keys_rtmc =  np.char.find(np.asarray(list(ch_models_rtmc.keys())), "Param") >= 0
    param_keys_rtmc = np.sort(np.asarray(list(ch_models_rtmc.keys()))[param_keys_rtmc])

    #ch_array_rtmc = np.array([ch_models_rtmc[k] for k in param_keys_rtmc if ch_models_rtmc[k].shape[4] == 30])
    ch_array_rtmc = np.array([ch_models_rtmc[k] for k in param_keys_rtmc])

    param_keys_dmc =  np.char.find(np.asarray(list(ch_models_dmc.keys())), "Param") >= 0
    param_keys_dmc = np.sort(np.asarray(list(ch_models_dmc.keys()))[param_keys_dmc])
    ch_array_dmc = np.array([ch_models_dmc[k] for k in param_keys_dmc])

    ch_array_rtmc[ch_array_rtmc == 0] = np.nan
    ch_array_dmc[ch_array_dmc == 0] = np.nan

    print(ch_array_dmc.shape)
    print(ch_array_rtmc.shape)

    ch_mean_rtmc = np.nanmean(ch_array_rtmc, axis=(1, 2, 3, 4, -1))[:, -1, -1]
    ch_std_rtmc = np.nanstd(ch_array_rtmc, axis=(1, 2, 3, 4, -1))[:, -1, -1]
    ch_mean_dmc = np.nanmean(ch_array_dmc, axis=(1, 2, 3, 4, -1))[:, -1]
    ch_std_dmc = np.nanstd(ch_array_dmc, axis=(1, 2, 3, 4, -1))[:, -1]
    
    print(ch_mean_dmc.shape)
    print(ch_mean_rtmc.shape)

    fig, ax = plt.subplots(figsize=(5, 3))

    ax.plot(ch_mean_dmc[[0,2,3,1]], label="Base DMC-MTL",marker='o')
    ax.plot(ch_mean_rtmc[[4, 7, -1, 1]], label="In-Season Adapation",marker='o')
    #ax.plot(ch_mean_rtmc[[0,2,3,1]], label="In-Season Adapation",marker='o')

    ax.set_xticks(np.arange(4), labels=[1,2,5,10])
    ax.set_xlabel("Years of Per-Cultivar Data")
    ax.set_ylabel(r"Root Mean Sqaured Error ($^\circ$C)")

    ax.legend()
    ax.set_title("In-Season Adaptation of DMC-MTL")

    plt.savefig("plotters/figs/rtmc_new.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
