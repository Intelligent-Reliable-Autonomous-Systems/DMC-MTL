"""
window.py

Plot the impact of weather window on performance

Written by Will Solow, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import utils


def main():
    # This is slightly wrong. Need to exclude true and "DeepMTL" in subdir
    pheno_models = utils.load_named_pickles(["RTMC/FineTuneColdHardiness"], "results_per_cultivars.pkl", exclude_multi=True) 

    pheno_sorted_keys = np.argsort(list(pheno_models.keys()))
    pheno_array = np.array(list(pheno_models.values()))[pheno_sorted_keys]
    print(np.array(list(pheno_models.keys()))[pheno_sorted_keys])
    pheno_array[pheno_array == 0] = np.nan
    pheno_mean = np.nanmean(pheno_array, axis=(1, 2, 3, 4, -1))[:, -10:-1]
    pheno_std = np.nanstd(pheno_array, axis=(1, 2, 3, 4, -1))[:, -10:-1]

    data = pheno_mean[0]

    fig, ax = plt.subplots(figsize=(5, 3))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    x = np.arange(9)
    for i in range(data.shape[0]):
        if i < 4 or i == 8:
            continue
        ax.plot(data[i], marker="o", label=f"Day {30*i}-{30*(i+1)}")

    ax.set_xticks(x, labels=[0, 30, 60, 90, 120, 150, 180, 210, 240])
    ax.set_xlabel("Error Data up to Day k")
    ax.set_ylabel("Root Mean Sqaured Error")

    ax.set_title("Cold-Hardiness RTMC with in season data")
    ax.legend(loc="lower left")

    plt.savefig("plotters/figs/rtmc_day.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
