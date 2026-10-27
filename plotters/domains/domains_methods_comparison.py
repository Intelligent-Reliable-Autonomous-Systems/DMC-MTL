"""
domains_methods_comparison.py

Overlay of models across phenology / cold-hardiness / wofost domains.
"""

import matplotlib.pyplot as plt
import numpy as np

from plotters.domains.domains_common import PANEL_CONFIGS, gen_results, load_models, plot_overlay


def main():
    cfg = PANEL_CONFIGS["full"]
    panels = cfg["panels"]
    loaded = []
    for p, yr in zip(panels, [18, -1, 2]):
        models = load_models(p["fpaths"])
        true, pred = gen_results(models, yr=yr)
        loaded.append((true, pred, p["names"]))

    fig, ax = plt.subplots(3, figsize=(6, 5.25))
    pheno_xticks = np.asarray([0, 31, 59, 90, 121, 151, 182, 212, 243])
    pheno_xlabels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
    ch_xticks = np.asarray([0, 30, 61, 92, 122, 153, 181, 212, 243])
    ch_xlabels = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May"]
    wf_xticks = pheno_xticks
    wf_xlabels = pheno_xlabels

    for i, (true, pred, names) in enumerate(loaded):
        plot_overlay(ax, true, pred, names, i=i)

    fontsize = 11.5
    ax[0].set_title("Comparison of Model Outputs Across Domains")
    ax[0].set_yticks([0, 1, 2, 3], ["Ecodorm", "Budbreak", "Bloom", "Veraison"])
    ax[0].set_ylim(ymin=-0.02)
    ax[0].set_xlim([0, len(loaded[0][0][0])])
    ax[0].set_xticks(pheno_xticks, pheno_xlabels)
    ax[0].legend(loc="upper left", fontsize=fontsize)

    ax[1].set_ylabel(r"LTE50 $^\circ$C")
    ax[1].set_xticks(ch_xticks, ch_xlabels)
    ax[1].set_xlim([0, len(loaded[1][0][0])])
    ax[1].set_ylim(bottom=-40)
    ax[1].legend(loc="lower right", fontsize=fontsize)

    ax[2].set_ylabel("Yield (kg/ha)")
    ax[2].set_xticks(wf_xticks, wf_xlabels)
    ax[2].set_xlim([0, len(loaded[2][0][0])])
    ax[2].set_ylim(ymin=-6)
    ax[2].legend(loc="upper left", fontsize=fontsize)

    for i, lab in enumerate(["(a)", "(b)", "(c)"]):
        ax[i].text(0.05, 0.2, lab, ha="left", va="top", transform=ax[i].transAxes, fontsize=fontsize + 1)

    plt.savefig(cfg["outfile"], bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
