"""
window.py

Plot the impact of weather window on performance

Written by Will Solow, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import utils


def main():

    pheno_models = utils.load_named_pickles(["PaperExperiments/PhenologyWindow"], "results_per_cultivars.pkl")
    ch_models = utils.load_named_pickles(["PaperExperiments/ColdHardinessWindow"], "results_per_cultivars.pkl")
    ff_pheno_models = utils.load_named_pickles(["PaperExperiments/Phenology/ParamFFMTL"], "results_per_cultivars.pkl")
    ff_ch_models = utils.load_named_pickles(["PaperExperiments/ColdHardiness/ParamFFMTL"], "results_per_cultivars.pkl")

    pheno_sorted_keys = np.argsort(list(pheno_models.keys()))
    pheno_array = np.array(list(pheno_models.values()))[pheno_sorted_keys][np.array([0, 3, 1, 2, 4, 5])]
    pheno_ff_array = np.array(list(ff_pheno_models.values())).squeeze()
    ch_sorted_keys = np.argsort(list(ch_models.keys()))
    ch_array = np.array(list(ch_models.values()))[ch_sorted_keys][np.array([0, 3, 1, 2, 4, 5])]
    ch_ff_array = np.array(list(ff_ch_models.values()))

    ch_ff_array[ch_ff_array == 0] = np.nan
    pheno_mean = np.mean(pheno_array, axis=(-3, -1))[:, -1]
    pheno_ff_mean = np.mean(pheno_ff_array, axis=(-3, -1))[-1]
    pheno_std = np.std(pheno_array, axis=(-3, -1))[:, -1]
    ch_mean = np.mean(ch_array, axis=(-3, -1))[:, -1]
    ch_ff_mean = np.nanmean(ch_ff_array, axis=(0, 1, 2, 3, 4, 6))[-1]
    ch_std = np.std(ch_array, axis=(-3, -1))[:, -1]

    pheno_mean = np.append(pheno_ff_mean, pheno_mean)
    ch_mean = np.append(ch_ff_mean, ch_mean)

    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax1_color = "tab:blue"
    ax.plot(np.arange(7), pheno_mean, color=ax1_color, label="Phenology", marker="o")
    ax.set_xlabel("Window Size")
    ax.set_xticks(np.arange(7), labels=["FF", 1, 5, 10, 20, 50, "All"])
    ax.set_ylabel("RMSE in Days (Phenology)")

    ax2_color = "tab:red"
    ax2 = ax.twinx()
    ax2.plot(np.arange(7), ch_mean, color=ax2_color, label="Cold-Hardiness", marker="s")
    ax2.set_ylabel(r"RMSE in $^\circ$C (Cold-Hardiness)")
    # ax.set_title("Weather Window Size Effect on Error")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper center")

    plt.savefig("plotters/figs/window.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
