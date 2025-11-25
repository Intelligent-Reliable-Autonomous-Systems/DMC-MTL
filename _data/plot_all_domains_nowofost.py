"""
plot_ch_pheno_data_merged.py

Plots a visualization of three cultivars and the weather
data for a given growing season. We merge the weather into one season

Written by Will Solow, 2025
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
from plotters.plotting_functions import C_AGG, C_PER, C_SHAPES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pheno_cultivars",
        default=["Chardonnay", "Pinot_Noir", "Melon", "Aligote", "Barbera"],
        type=list,
        help="Path to Config",
    )
    parser.add_argument(
        "--ch_cultivars",
        default=["Merlot", "Malbec", "Concord", "Riesling", "Syrah"],
        type=list,
        help="Path to Config",
    )
    parser.add_argument("--yr", type=int, default=1)
    args = parser.parse_args()

    # Define cultivars and load last year of data
    pheno_yr = []
    ch_yr = []
    wf_yr = []
    for c in args.pheno_cultivars:
        with open(f"_data/processed_data/grape_phenology/WA/Roza2/Prosser/WA_Prosser_grape_phenology_{c}.pkl", "rb") as f:
            data = pkl.load(f)[-args.yr]
            pheno_yr.append(data)
        f.close()
    for c in args.ch_cultivars:
        with open(f"_data/processed_data/grape_coldhardiness/WA/Roza2/Prosser/WA_Prosser_grape_coldhardiness_{c}.pkl", "rb") as f:
            data = pkl.load(f)[-args.yr]
            ch_yr.append(data)
        f.close()

    fig, ax = plt.subplots(ncols=2, figsize=(12, 1.5))
    pheno_xticks = np.asarray([0, 31, 59, 90, 121, 151, 182, 212, 243])
    pheno_xlabels = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
    ]
    ch_xticks = np.asarray([0, 30, 61, 92, 122, 153, 181, 212, 243])
    ch_xlabels = [
        "Sep",
        "Oct",
        "Nov",
        "Dec",
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
    ]

    lb_spacing = 0.3
    handle_txt_pad = 0.3
    border_ax_pad = 0.3
    handle_length = 0.6
    fontsize = 11.5
    # Phenology
    pheno_lines = []
    for i in range(len(args.pheno_cultivars)):
        ln = ax[0].plot(
            np.arange(len(pheno_yr[i])),
            pheno_yr[i].loc[:, "PHENOLOGY"],
            label=args.pheno_cultivars[i].replace("_", " "),
            c=C_PER[i],
            alpha=0.9,
        )
        pheno_lines.append(ln)
    ax[0].set_ylim(ymin=-0.05)
    ax[0].set_xlim(xmax=250)
    ax[0].set_yticks([0, 1, 2, 3], ["Ecodorm", "Bud Break", "Bloom", "Veraison"], rotation=0)
    ax[0].set_xticks(pheno_xticks, pheno_xlabels)
    ax[0].legend(
        loc="upper left",
        handletextpad=handle_txt_pad,
        borderaxespad=border_ax_pad,
        labelspacing=lb_spacing,
        handlelength=handle_length,
        fontsize=fontsize,
    )

    ch_lines = []
    for i in range(len(args.ch_cultivars)):
        ln = ax[1].scatter(
            np.arange(len(ch_yr[i])),
            ch_yr[i].loc[:, "LTE50"],
            label=args.ch_cultivars[i].replace("_", " "),
            c=C_PER[i],
            marker=C_SHAPES[i],
            s=4,
        )
        ch_lines.append(ln)

    ax[1].set_xticks(ch_xticks, ch_xlabels)
    ax[1].set_ylabel(r"LTE50 $^\circ$C")
    legend1 = ax[1].legend(
        loc="lower left",
        handles=ch_lines[:3],
        handletextpad=handle_txt_pad,
        borderaxespad=border_ax_pad,
        labelspacing=lb_spacing,
        handlelength=handle_length,
        fontsize=fontsize,
    )
    ax[1].add_artist(legend1)
    ax[1].legend(
        loc="lower right",
        handles=ch_lines[3:],
        handletextpad=handle_txt_pad,
        borderaxespad=border_ax_pad,
        labelspacing=lb_spacing,
        handlelength=handle_length,
        fontsize=fontsize,
    )


    ax[0].text(0.99, 0.91, f"(a)", ha="right", va="top", transform=ax[0].transAxes, fontsize=fontsize + 1)
    ax[1].text(0.99, 0.91, f"(b)", ha="right", va="top", transform=ax[1].transAxes, fontsize=fontsize + 1)

    # ax[0].set_facecolor("#F5F5F5")
    # ax[1].set_facecolor("#F5F5F5")
    # ax[2].set_facecolor("#F5F5F5")
    # fig.patch.set_facecolor("#F5F5F5")  #
    # ax[0].set_title("Visualization of Per-Domain Observations")
    plt.savefig("_data/figs/visualization_of_data.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()