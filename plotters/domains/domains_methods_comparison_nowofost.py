"""
domains_methods_comparison_nowofost.py

Phenology + cold-hardiness overlay (no wofost panel).
"""

import matplotlib.pyplot as plt
import numpy as np

from plotters.domains.domains_common import PANEL_CONFIGS, gen_results, load_models, plot_overlay


def main():
    cfg = PANEL_CONFIGS["nowofost"]
    panels = cfg["panels"]
    loaded = []
    for p in panels:
        models = load_models(p["fpaths"])
        true, pred = gen_results(models, yr=77)
        loaded.append((true, pred, p["names"]))

    fig, ax = plt.subplots(2, figsize=(6, 3.5))
    pheno_xticks = np.asarray([0, 31, 59, 90, 121, 151, 182, 212, 243])
    pheno_xlabels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
    ch_xticks = np.asarray([0, 30, 61, 92, 122, 153, 181, 212, 243])
    ch_xlabels = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May"]

    plot_overlay(ax, loaded[0][0], loaded[0][1], loaded[0][2], i=0)
    plot_overlay(ax, loaded[1][0], loaded[1][1], loaded[1][2], i=1, scatter_obs=True)

    fontsize = 12
    ax[0].set_yticks([0, 1, 2, 3], ["Ecodorm", "Budbreak", "Bloom", "Veraison"])
    ax[0].set_ylim(ymin=-0.02)
    ax[0].set_xlim([0, 250])
    ax[0].set_xticks(pheno_xticks, pheno_xlabels)
    fig.legend(loc="upper center", ncols=3, fontsize=fontsize)

    ax[1].set_ylabel(r"LTE50 $^\circ$C")
    ax[1].set_xticks(ch_xticks, ch_xlabels)
    ax[1].set_xlim([0, len(loaded[1][0][0])])
    ax[1].set_ylim(bottom=-40)

    plt.subplots_adjust(hspace=0.2)
    ax[0].text(
        0.05, 0.2, "(a)", ha="left", va="top", transform=ax[0].transAxes, fontsize=fontsize + 1, fontweight="bold"
    )
    ax[1].text(
        0.05, 0.2, "(b)", ha="left", va="top", transform=ax[1].transAxes, fontsize=fontsize + 1, fontweight="bold"
    )

    plt.savefig(cfg["outfile"], bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
