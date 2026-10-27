"""
plot_all_domains.py

Plots phenology + cold hardiness (and optionally WOFOST wheat) for a season.

Written by Will Solow, 2025
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

from plotters.plotting_functions import C_PER, C_SHAPES

PHENO_PATH = "_data/processed_data/grape_phenology/WA/Roza2/Prosser/WA_Prosser_grape_phenology_{c}.pkl"
CH_PATH = "_data/processed_data/grape_coldhardiness/WA/Roza2/Prosser/WA_Prosser_grape_coldhardiness_{c}.pkl"
WOFOST_PATH = "_data/processed_data/wofost_{c}.pkl"


def _load_last_year(path: str, yr: int):
    with open(path, "rb") as f:
        return pkl.load(f)[-yr]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pheno_cultivars",
        default=["Chardonnay", "Pinot_Noir", "Melon", "Aligote", "Barbera"],
        type=list,
    )
    parser.add_argument(
        "--ch_cultivars",
        default=["Merlot", "Malbec", "Concord", "Riesling", "Syrah"],
        type=list,
    )
    parser.add_argument(
        "--wheat_cultivars",
        default=["Winter_Wheat_101", "Winter_Wheat_105", "Winter_Wheat_107", "Apache", "Bermude"],
        type=list,
    )
    parser.add_argument("--yr", type=int, default=1)
    parser.add_argument(
        "--include-wofost",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include WOFOST wheat panel (default: off)",
    )
    args = parser.parse_args()

    pheno_yr = [_load_last_year(PHENO_PATH.format(c=c), args.yr) for c in args.pheno_cultivars]
    ch_yr = [_load_last_year(CH_PATH.format(c=c), args.yr) for c in args.ch_cultivars]
    wf_yr = []
    if args.include_wofost:
        wf_yr = [_load_last_year(WOFOST_PATH.format(c=c), args.yr) for c in args.wheat_cultivars]

    if args.include_wofost:
        fig, ax = plt.subplots(nrows=3, figsize=(6, 5.25))
    else:
        fig, ax = plt.subplots(ncols=2, figsize=(12, 1.5))

    pheno_xticks = np.asarray([0, 31, 59, 90, 121, 151, 182, 212, 243])
    pheno_xlabels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
    ch_xticks = np.asarray([0, 30, 61, 92, 122, 153, 181, 212, 243])
    ch_xlabels = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May"]
    wf_xticks = pheno_xticks
    wf_xlabels = pheno_xlabels

    lb_spacing = 0.3
    handle_txt_pad = 0.3
    border_ax_pad = 0.3
    handle_length = 0.6
    fontsize = 11.5

    for i in range(len(args.pheno_cultivars)):
        ax[0].plot(
            np.arange(len(pheno_yr[i])),
            pheno_yr[i].loc[:, "PHENOLOGY"],
            label=args.pheno_cultivars[i].replace("_", " "),
            c=C_PER[i],
            alpha=0.9,
        )
    ax[0].set_ylim(ymin=-0.05)
    if not args.include_wofost:
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

    ax[0].text(0.99, 0.91, "(a)", ha="right", va="top", transform=ax[0].transAxes, fontsize=fontsize + 1)
    ax[1].text(0.99, 0.91, "(b)", ha="right", va="top", transform=ax[1].transAxes, fontsize=fontsize + 1)

    if args.include_wofost:
        for i in range(len(args.wheat_cultivars)):
            ax[2].plot(
                np.arange(len(wf_yr[i])),
                wf_yr[i].loc[:, "WSO"],
                label=args.wheat_cultivars[i].replace("_", " ").replace("Winter ", ""),
                c=C_PER[i],
                alpha=0.9,
            )
        ax[2].set_ylim(ymin=-5)
        ax[2].set_ylabel("Yield (kg/ha)")
        ax[2].set_xticks(wf_xticks, wf_xlabels)
        ax[2].legend(
            loc="upper left",
            handletextpad=handle_txt_pad,
            borderaxespad=border_ax_pad,
            labelspacing=lb_spacing,
            handlelength=handle_length,
            fontsize=fontsize,
        )
        ax[2].text(0.99, 0.91, "(c)", ha="right", va="top", transform=ax[2].transAxes, fontsize=fontsize + 1)

    plt.savefig("_data/figs/visualization_of_data.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
