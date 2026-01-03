"""
window.py

Plot the impact of weather window on performance

Written by Will Solow, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import utils


def main():

    pheno_deep_models = utils.load_named_pickles(["PaperExperiments/PhenologyLimited"], "results_per_cultivars.pkl", include="Deep")
    pheno_param_models = utils.load_named_pickles(["PaperExperiments/PhenologyLimited"], "results_per_cultivars.pkl", include="Param")
    ch_deep_models = utils.load_named_pickles(["PaperExperiments/ColdHardinessLimited"], "results_per_cultivars.pkl", include="Deep")
    ch_param_models = utils.load_named_pickles(["PaperExperiments/ColdHardinessLimited"], "results_per_cultivars.pkl", include="Param")

    pheno_gd_models = utils.load_named_pickles(["PaperExperiments/PhenologyLimited"], "results_agg_cultivars.pkl", include="GD")
    pheno_hybrid_models = utils.load_named_pickles(["PaperExperiments/PhenologyLimited"], "results_agg_cultivars.pkl", include="Hybrid")
    ch_gd_models = utils.load_named_pickles(["PaperExperiments/ColdHardinessLimited"], "results_agg_cultivars.pkl", include="GD")
    ch_hybrid_models = utils.load_named_pickles(["PaperExperiments/ColdHardinessLimited"], "results_agg_cultivars.pkl", include="Hybrid")

    inds = [0, 3, 4, 5, 1, 2]
    pheno_deep_sorted_keys = np.argsort(list(pheno_deep_models.keys()))[inds]
    pheno_deep_array = np.array(list(pheno_deep_models.values()))[pheno_deep_sorted_keys]
    pheno_param_sorted_keys = np.argsort(list(pheno_param_models.keys()))[inds]
    pheno_param_array = np.array(list(pheno_param_models.values()))[pheno_param_sorted_keys]

    pheno_gd_sorted_keys = np.argsort(list(pheno_gd_models.keys()))
    pheno_gd_sorted_keys = np.concatenate((pheno_gd_sorted_keys[:31], pheno_gd_sorted_keys[93:], pheno_gd_sorted_keys[31:93]))
    pheno_gd_array = np.array(list(pheno_gd_models.values()))[pheno_gd_sorted_keys].reshape(6,31,12,5)
    pheno_hybrid_sorted_keys = np.argsort(list(pheno_hybrid_models.keys()))
    pheno_hybrid_sorted_keys = np.concatenate((pheno_hybrid_sorted_keys[:31], pheno_hybrid_sorted_keys[93:], pheno_hybrid_sorted_keys[31:93]))
    pheno_hybrid_array = np.array(list(pheno_hybrid_models.values()))[pheno_hybrid_sorted_keys].reshape(6,31,12,5)

    ch_deep_sorted_keys = np.argsort(list(ch_deep_models.keys()))[inds]
    ch_deep_array = np.array(list(ch_deep_models.values()))[ch_deep_sorted_keys]
    ch_param_sorted_keys = np.argsort(list(ch_param_models.keys()))[inds]
    ch_param_array = np.array(list(ch_param_models.values()))[ch_param_sorted_keys]

    ch_gd_sorted_keys = np.argsort(list(ch_gd_models.keys()))
    ch_gd_sorted_keys = np.concatenate((ch_gd_sorted_keys[:20], ch_gd_sorted_keys[60:], ch_gd_sorted_keys[20:60]))
    ch_gd_array = np.array(list(ch_gd_models.values()))[ch_gd_sorted_keys].reshape(6, 20, 3, 5)
    ch_hybrid_sorted_keys = np.argsort(list(ch_hybrid_models.keys()))
    ch_hybrid_sorted_keys = np.concatenate((ch_hybrid_sorted_keys[:20], ch_hybrid_sorted_keys[60:], ch_hybrid_sorted_keys[20:60]))
    ch_hybrid_array = np.array(list(ch_hybrid_models.values()))[ch_hybrid_sorted_keys].reshape(6, 20, 3, 5)


    pheno_deep_array[pheno_deep_array == 0] = np.nan
    pheno_param_array[pheno_param_array == 0] = np.nan
    ch_deep_array[ch_deep_array == 0] = np.nan
    ch_param_array[ch_param_array == 0] = np.nan

    pheno_gd_array[pheno_gd_array == 0] = np.nan
    pheno_hybrid_array[pheno_hybrid_array == 0] = np.nan
    ch_gd_array[ch_gd_array == 0] = np.nan
    ch_hybrid_array[ch_hybrid_array == 0] = np.nan

    pheno_deep_mean = np.nanmean(pheno_deep_array, axis=(-3, -1))[:, -1]
    pheno_deep_std = np.nanstd(pheno_deep_array, axis=(-3, -1))[:, -1]
    pheno_param_mean = np.nanmean(pheno_param_array, axis=(-3, -1))[:, -1]
    pheno_param_std = np.nanstd(pheno_param_array, axis=(-3, -1))[:, -1]

    pheno_gd_mean = np.nanmean(pheno_gd_array, axis=(-3, -1))[:, -1]
    pheno_gd_std = np.nanstd(pheno_gd_array, axis=(-3, -1))[:, -1]
    pheno_hybrid_mean = np.nanmean(pheno_hybrid_array, axis=(-3, -1))[:, -1]
    pheno_hybrid_std = np.nanstd(pheno_hybrid_array, axis=(-3, -1))[:, -1]

    ch_deep_mean = np.nanmean(ch_deep_array, axis=(-3, -1))[:, -1]
    ch_deep_std = np.nanstd(ch_deep_array, axis=(-3, -1))[:, -1]
    ch_param_mean = np.nanmean(ch_param_array, axis=(1, 2, 3, 4, -1))[:, -1]
    ch_param_std = np.nanstd(ch_param_array, axis=(1, 2, 3, 4, -1))[:, -1]

    ch_gd_mean = np.nanmean(ch_gd_array, axis=(-3, -1))[:, -1]
    ch_gd_std = np.nanstd(ch_gd_array, axis=(-3, -1))[:, -1]
    ch_hybrid_mean = np.nanmean(ch_hybrid_array, axis=(-3, -1))[:, -1]
    ch_hybrid_std = np.nanstd(ch_hybrid_array, axis=(-3, -1))[:, -1]

    fig, ax = plt.subplots(figsize=(5, 3))

    ax2_color = "tab:red"
    ax2 = ax.twinx()
    ax2.plot(np.arange(6), ch_deep_mean, color=ax2_color, marker="o", label="CH Reg. MTL")
    ax2.plot(np.arange(6), ch_param_mean, color=ax2_color, marker="s", label="CH DMC-MTL")
    ax2.plot(np.arange(6), ch_gd_mean, color=ax2_color, marker="*", label="CH Bio. Model")
    #ax2.plot(np.arange(6), ch_hybrid_mean, color=ax2_color, marker="d", label="CH Temp-Hybrid")

    ax1_color = "tab:blue"
    ax.plot(np.arange(6), pheno_deep_mean, color=ax1_color, marker="o", label="Pheno Reg. MTL")
    ax.plot(np.arange(6), pheno_param_mean, color=ax1_color, marker="s", label="Pheno DMC-MTL")
    ax.plot(np.arange(6), pheno_gd_mean, color=ax1_color, marker="*", label="Pheno Bio. Model")
    #ax.plot(np.arange(6), pheno_hybrid_mean, color=ax1_color, marker="d", label="Pheno Temp-Hybrid")
    ax.set_xlabel("Seasons of Data per Cultivar")
    ax.set_xticks(np.arange(6), labels=[1, 2, 3, 5, 10, 15])
    ax.set_ylabel("RMSE in Days (Phenology)", color=ax1_color)
    ax.tick_params(axis="y", labelcolor=ax1_color)

    ax2.set_ylabel(r"RMSE in $^\circ$C (Cold-Hardiness)", color=ax2_color)
    ax2.tick_params(axis="y", labelcolor=ax2_color)
    ax.set_title("Effect of Per Cultivar Data Availability")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    plt.savefig("plotters/figs/data.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
