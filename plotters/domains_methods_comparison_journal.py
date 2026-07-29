"""
domains_methods_comarison.py

Plots overlay of models to demonstrate why DMC-MTL methods are better

Written by Will Solow, 2025
"""

import argparse
import os
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from omegaconf import OmegaConf

import utils
from plotters.plot_utils import gen_batch_data
from plotters.plotting_functions import C_AGG, C_PER, C_SHAPES
from train_algs.base.process_data import unnormalize


def load_models(fpaths: list[str], name: str = "rnn_model_best.pt", args: Namespace = None) -> list[torch.nn.Module]:
    """
    Load prediction models from a list of configation strings
    """
    calibrators = []
    for f in fpaths:
        config = OmegaConf.load(f"{f}/config.yaml")
        config = OmegaConf.merge(utils.Args, config)
        data = utils.load_data_from_config(config, args)
        fpath = fpath = f"{os.getcwd()}/{f}"
        c = utils.load_model_from_config(config, data)
        c.load_model(f"{fpath}", name=name)
        calibrators.append(c)

    return calibrators


def gen_results(calibrators: list[torch.nn.Module], yr: int = -1, split: str = "train") -> tuple[list, list]:
    """
    Generate results from a list of prediction models
    """
    model_true = []
    model_pred = []
    weather_arr = []

    for calibrator in calibrators:
        cultivars = calibrator.cultivars[split][yr] if calibrator.cultivars is not None else None
        regions = calibrator.regions[split][yr] if calibrator.regions is not None else None
        stations = calibrator.stations[split][yr] if calibrator.stations is not None else None
        sites = calibrator.sites[split][yr] if calibrator.sites is not None else None

        true, output, params = gen_batch_data(
            calibrator,
            calibrator.data[split][yr],
            calibrator.dates[split][yr],
            calibrator.val[split][yr],
            cultivars=cultivars,
            regions=regions,
            stations=stations,
            sites=sites,
        )
        model_true.append(true)
        model_pred.append(output)
        weather_arr.append(unnormalize(calibrator.data[split][yr], calibrator.drange).cpu().numpy())
    return model_true, model_pred, weather_arr


def plot_results(
    ax: Axes,
    model_true: list[np.ndarray],
    model_pred: list[np.ndarray],
    weather_arr: list[np.ndarray],
    model_names: list[str],
    i: int = 0,
    label: bool = True,
) -> None:
    """
    Plot the results on a given axis
    """
    ax[0].plot(weather_arr[0][:, 2])
    # ax[0].fill_between(np.arange(weather_arr[0][:, 3].shape[0]), weather_arr[0][:, 3], weather_arr[0][:, 4], alpha=0.6)
    for j in range(len(model_true)):
        if model_pred[j].shape[-1] == 3:  # Handle GCHN 3 outputs
            output = model_pred[j][:, 0]
        else:
            output = model_pred[j]

        x = np.arange(len(output))
        ax[1].plot(x, output, label=f"{model_names[j]}" if label else None, c=C_PER[j], zorder=3)

        if j == len(model_true) - 1:
            ax[1].scatter(
                x, model_true[0], label="True Obs." if label else None, s=10, c=C_AGG[1], marker=C_SHAPES[0], zorder=1
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        default=0,
        type=int,
    )
    args = parser.parse_args()
    ch_fpaths = [
        "_runs/Journal/DMC/All/All/All/Region_QCWITXPA-/param_mtl__1779223682",
        "_runs/Journal/GrapeHardiNet/All/All/All/All/deep_mtl__1779915557",
        # "_runs/PaperExperiments/ColdHardiness/PINNMTL/All/pinn_mtl__1746895757"
    ]

    ch_model_names = ["Hybrid Model", "DL Model"]

    ch_models = load_models(ch_fpaths, args=args)

    ch_true, ch_pred, weather_arr = gen_results(ch_models, yr=12)  # 31

    fig, ax = plt.subplots(2, figsize=(6, 5))

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

    plot_results(ax, ch_true, ch_pred, weather_arr, ch_model_names, i=1, label=True)

    lb_spacing = 0.3
    handle_txt_pad = 0.3
    border_ax_pad = 0.3
    handle_length = 0.6
    fontsize = 12

    fig.legend(
        loc="upper center",
        ncols=3,
        handletextpad=handle_txt_pad,
        borderaxespad=border_ax_pad,
        labelspacing=lb_spacing,
        handlelength=handle_length,
        fontsize=fontsize,
    )
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
