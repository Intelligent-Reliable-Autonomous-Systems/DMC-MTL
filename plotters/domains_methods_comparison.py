"""
domains_methods_comarison.py

Plots overlay of models to demonstrate why DMC-MTL methods are better

Written by Will Solow, 2025
"""

import os
import numpy as np
import argparse
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
import utils
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from plotters.plot_utils import gen_batch_data, C_AGG, C_PER, C_SHAPES

from model_engine.util import PHENOLOGY_INT


def load_models(fpaths: list[str], name: str = "rnn_model_best.pt") -> list[torch.nn.Module]:
    """
    Load prediction models from a list of configation strings
    """
    calibrators = []
    for f in fpaths:
        config = OmegaConf.load(f"{f}/config.yaml")
        config = OmegaConf.merge(utils.Args, config)
        data = utils.load_data_from_config(config)
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

    return model_true, model_pred


def plot_results(
    ax: Axes, model_true: list[np.ndarray], model_pred: list[np.ndarray], model_names: list[str], i: int = 0
) -> None:
    """
    Plot the results on a given axis
    """
    for j in range(len(model_true)):

        if model_pred[j].shape[-1] == len(PHENOLOGY_INT):  # Handle categorical classification
            output = torch.tensor(output)
            probs = F.softmax(output, dim=-1)
            output = torch.argmax(probs, dim=-1)
        elif model_pred[j].shape[-1] == 3:  # Handle GCHN 3 outputs
            output = model_pred[j][:, 0]
        else:
            output = model_pred[j]

        x = np.arange(len(output))
        if i == 1:
            ax[i].plot(x, output, label=f"{model_names[j]}", c=C_PER[j])
            if j == len(model_true) - 1:
                ax[i].scatter(x, model_true[0], label="True Obs.", s=10, c=C_AGG[1], marker=C_SHAPES[0])

        else:
            ax[i].plot(x, output, label=f"{model_names[j]}", c=C_PER[j])
            if j == len(model_true) - 1:
                ax[i].plot(x, model_true[0], label="True Obs.", c=C_AGG[1])


def main():

    pheno_fpaths = [
        "_runs/PaperExperiments/Phenology/ParamMTL/Multi/param_zscore__1746932177",
        # "_runs/PaperExperiments/Phenology/PINN/Multi/pinn_mtl__1747258573",
        "_runs/PaperExperiments/Phenology/DeepClass/Multi/class_mtl__1747001360",
    ]
    ch_fpaths = [
        "_runs/PaperExperiments/ColdHardiness/ParamMTL/Multi/paramMTL__1746844099",
        "_runs/PaperExperiments/ColdHardiness/DeepGCHNMTL/Multi/gchn_zscore__1746873252",
        # "_runs/PaperExperiments/ColdHardiness/PINNMTL/Multi/pinn_mtl__1746895757"
    ]
    wf_fpaths = [
        "_runs/PaperExperiments/wofost/ParamMTL/Multi/param_mtl__1749924131",
        "_runs/PaperExperiments/wofost/DeepMTL/Multi/deepMTL__1749324822",
        # "_runs/PaperExperiments/wofost/PINNMTL/Multi/pinn_mtl__1749331372"
    ]

    pheno_model_names = ["DMC-MTL", "Deep-MTL"]
    ch_model_names = ["DMC-MTL", "Deep-MTL"]
    wf_model_names = ["DMC-MTL", "Deep-MTL"]

    pheno_models = load_models(pheno_fpaths)
    ch_models = load_models(ch_fpaths)
    wf_models = load_models(wf_fpaths)

    pheno_true, pheno_pred = gen_results(pheno_models, yr=18)
    ch_true, ch_pred = gen_results(ch_models)
    wf_true, wf_pred = gen_results(wf_models, yr=2)

    fig, ax = plt.subplots(3, figsize=(6, 5.25))

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

    wf_xticks = np.asarray([0, 31, 59, 90, 121, 151, 182, 212, 243])
    wf_xlabels = [
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

    plot_results(ax, pheno_true, pheno_pred, pheno_model_names, i=0)
    plot_results(ax, ch_true, ch_pred, ch_model_names, i=1)
    plot_results(ax, wf_true, wf_pred, wf_model_names, i=2)

    lb_spacing = 0.3
    handle_txt_pad = 0.3
    border_ax_pad = 0.3
    handle_length = 0.6
    fontsize = 11.5

    ax[0].set_title("Comparison of Model Outputs Across Domains")
    ax[0].set_yticks(
        [0, 1, 2, 3],
        ["Ecodorm", "Budbreak", "Bloom", "Veraison"],
    )
    ax[0].set_ylim(ymin=-0.02)
    ax[0].set_xlim([0, len(pheno_true[0])])
    ax[0].set_xticks(pheno_xticks, pheno_xlabels)
    ax[0].legend(
        loc="upper left",
        handletextpad=handle_txt_pad,
        borderaxespad=border_ax_pad,
        labelspacing=lb_spacing,
        handlelength=handle_length,
        fontsize=fontsize,
    )

    ax[1].set_ylabel(r"LTE50 $^\circ$C")
    ax[1].set_xticks(ch_xticks, ch_xlabels)
    ax[1].set_xlim([0, len(ch_true[0])])
    ax[1].set_ylim(bottom=-40)
    ax[1].legend(
        loc="lower right",
        handletextpad=handle_txt_pad,
        borderaxespad=border_ax_pad,
        labelspacing=lb_spacing,
        handlelength=handle_length,
        fontsize=fontsize,
    )

    ax[2].set_ylabel("Yield (kg/ha)")
    ax[2].set_xticks(wf_xticks, wf_xlabels)
    ax[2].set_xlim([0, len(wf_true[0])])
    ax[2].set_ylim(ymin=-6)
    ax[2].legend()
    ax[2].legend(
        loc="upper left",
        handletextpad=handle_txt_pad,
        borderaxespad=border_ax_pad,
        labelspacing=lb_spacing,
        handlelength=handle_length,
        fontsize=fontsize,
    )

    ax[0].text(0.05, 0.2, f"(a)", ha="left", va="top", transform=ax[0].transAxes, fontsize=fontsize + 1)
    ax[1].text(0.05, 0.2, f"(b)", ha="left", va="top", transform=ax[1].transAxes, fontsize=fontsize + 1)
    ax[2].text(0.05, 0.2, f"(c)", ha="left", va="top", transform=ax[2].transAxes, fontsize=fontsize + 1)

    plt.savefig("plotters/figs/domain_methods_comparison.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
