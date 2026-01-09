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

    ch_models_rtmc = utils.load_named_pickles(
        ["RTMC/FineTuneColdHardiness"], "results_per_cultivars.pkl", include="All"
    )
    ch_models_dmc = utils.load_named_pickles(["RTMC/ColdHardiness"], "results_per_cultivars.pkl", include="All")

    pheno_models_rtmc = utils.load_named_pickles(["RTMC/FineTunePhenology"], "results_per_cultivars.pkl", include="All")
    pheno_models_dmc = utils.load_named_pickles(["RTMC/Phenology"], "results_per_cultivars.pkl", include="All")

    ch_param_keys_rtmc = np.char.find(np.asarray(list(ch_models_rtmc.keys())), "Param") >= 0
    ch_param_keys_rtmc = np.sort(np.asarray(list(ch_models_rtmc.keys()))[ch_param_keys_rtmc])

    pheno_param_keys_rtmc = np.char.find(np.asarray(list(pheno_models_rtmc.keys())), "Param") >= 0
    pheno_param_keys_rtmc = np.sort(np.asarray(list(pheno_models_rtmc.keys()))[pheno_param_keys_rtmc])

    ch_array_rtmc = np.array([ch_models_rtmc[k] for k in ch_param_keys_rtmc])
    pheno_array_rtmc = np.array([pheno_models_rtmc[k] for k in pheno_param_keys_rtmc])

    ch_param_keys_dmc = np.char.find(np.asarray(list(ch_models_dmc.keys())), "Param") >= 0
    ch_param_keys_dmc = np.sort(np.asarray(list(ch_models_dmc.keys()))[ch_param_keys_dmc])
    pheno_param_keys_dmc = np.char.find(np.asarray(list(pheno_models_dmc.keys())), "Param") >= 0
    pheno_param_keys_dmc = np.sort(np.asarray(list(pheno_models_dmc.keys()))[pheno_param_keys_dmc])

    ch_array_dmc = np.array([ch_models_dmc[k] for k in ch_param_keys_dmc])
    pheno_array_dmc = np.array([pheno_models_dmc[k] for k in pheno_param_keys_dmc])

    ch_array_rtmc[ch_array_rtmc == 0] = np.nan
    ch_array_dmc[ch_array_dmc == 0] = np.nan

    pheno_array_rtmc[pheno_array_rtmc == 0] = np.nan
    pheno_array_dmc[pheno_array_dmc == 0] = np.nan

    ch_mean_rtmc = np.nanmean(ch_array_rtmc, axis=(1, 2, 3, 4, -1))[:, -1]
    ch_mean_dmc = np.nanmean(ch_array_dmc, axis=(1, 2, 3, 4, -1))[:, -1]

    pheno_mean_rtmc = np.nanmean(pheno_array_rtmc, axis=(1, 2, 3, 4, -1))[:, -1]
    pheno_mean_dmc = np.nanmean(pheno_array_dmc, axis=(1, 2, 3, 4, -1))[:, -1]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax2 = ax.twinx()

    ax.plot(pheno_mean_dmc[[0, 2, 3, 1]], label="Pheno: Base DMC-MTL", marker="o", c="tab:red")
    ax.plot(pheno_mean_rtmc[[1, 2, 0, 3]], label="Pheno: In-Season Adapation", marker="s", c="tab:red", linestyle="--")

    ax2.plot(ch_mean_dmc[[0, 2, 3, 1]], label="CH: Base DMC-MTL", marker="o", c="tab:blue")
    ax2.plot(ch_mean_rtmc[[1, 2, 0, 3]], label="CH: In-Season Adapation", marker="s", c="tab:blue", linestyle="--")

    ax.set_xticks(np.arange(4), labels=[1, 2, 5, 10])
    ax.set_xlabel("Years of Per-Cultivar Data")
    ax2.set_ylabel(r"CH: Root Mean Sqaured Error ($^\circ$C)")
    ax.set_ylabel(r"Pheno: Root Mean Sqaured Error (Days)")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    ax.legend(h1 + h2, l1 + l2, loc="best")

    ax.set_title("In-Season Adaptation of DMC-MTL")

    plt.savefig("plotters/figs/rtmc_new.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
