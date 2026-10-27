"""
domains_methods_comparison_journal.py

Cold-hardiness overlay with weather panel for journal figure.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from plotters.domains_common import gen_results, load_models
from plotters.plotting_functions import C_AGG, C_PER, C_SHAPES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=0, type=int)
    args = parser.parse_args()

    ch_fpaths = [
        "_runs/Journal/DMC/All/All/All/Region_QCWITXPA-/param_mtl__1779223682",
        "_runs/Journal/GrapeHardiNet/All/All/All/All/deep_mtl__1779915557",
    ]
    ch_model_names = ["Hybrid Model", "DL Model"]
    ch_models = load_models(ch_fpaths, args=args)
    ch_true, ch_pred, weather_arr = gen_results(ch_models, yr=12, with_weather=True)

    fig, ax = plt.subplots(2, figsize=(6, 5))
    ch_xticks = np.asarray([0, 30, 61, 92, 122, 153, 181, 212, 243])
    ch_xlabels = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May"]

    ax[0].plot(weather_arr[0][:, 2])
    for j in range(len(ch_true)):
        output = ch_pred[j][:, 0] if ch_pred[j].shape[-1] == 3 else ch_pred[j]
        x = np.arange(len(output))
        ax[1].plot(x, output, label=ch_model_names[j], c=C_PER[j], zorder=3)
        if j == len(ch_true) - 1:
            ax[1].scatter(x, ch_true[0], label="True Obs.", s=10, c=C_AGG[1], marker=C_SHAPES[0], zorder=1)

    fontsize = 12
    fig.legend(loc="upper center", ncols=3, fontsize=fontsize)
    ax[0].set_xticks([])
    ax[0].set_ylabel(r"Daily Average Temperature ($^\circ$C)")
    ax[0].set_xlim([0, len(ch_true[0]) - 10])
    ax[1].set_ylabel(r"LTE50 ($^\circ$C)")
    ax[1].set_xticks(ch_xticks, ch_xlabels)
    ax[1].set_xlim([0, len(ch_true[0])])
    ax[1].set_ylim(bottom=-30)
    plt.subplots_adjust(hspace=0.2)
    plt.savefig("plotters/figs/domain_methods_comparison_journal.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
