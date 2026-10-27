"""
Shared helpers for domains_methods_comparison* figure scripts.

Written by Will Solow, 2025
"""

from __future__ import annotations

import os
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.axes import Axes
from omegaconf import OmegaConf

import utils
from model_engine.util import PHENOLOGY_INT
from plotters.plot_utils import gen_batch_data
from plotters.plotting_functions import C_AGG, C_PER, C_SHAPES


def load_models(
    fpaths: list[str], name: str = "rnn_model_best.pt", args: Namespace | None = None
) -> list[torch.nn.Module]:
    calibrators = []
    for f in fpaths:
        config = OmegaConf.load(f"{f}/config.yaml")
        config = OmegaConf.merge(utils.Args, config)
        data = utils.load_data_from_config(config, args) if args is not None else utils.load_data_from_config(config)
        fpath = f"{os.getcwd()}/{f}"
        c = utils.load_model_from_config(config, data)
        c.load_model(fpath, name=name)
        calibrators.append(c)
    return calibrators


def gen_results(
    calibrators: list[torch.nn.Module], yr: int = -1, split: str = "train", *, with_weather: bool = False
):
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
        if with_weather:
            weather_arr.append(calibrator.data[split][yr].detach().cpu().numpy())
    if with_weather:
        return model_true, model_pred, weather_arr
    return model_true, model_pred


def plot_overlay(
    ax: Axes,
    model_true: list[np.ndarray],
    model_pred: list[np.ndarray],
    model_names: list[str],
    i: int = 0,
    *,
    scatter_obs: bool = False,
) -> None:
    for j in range(len(model_true)):
        pred = model_pred[j]
        if pred.shape[-1] == len(PHENOLOGY_INT):
            output = torch.tensor(pred)
            probs = F.softmax(output, dim=-1)
            output = torch.argmax(probs, dim=-1).numpy()
        elif pred.shape[-1] == 3:
            output = pred[:, 0]
        else:
            output = pred
        x = np.arange(len(output))
        ax[i].plot(x, output, label=f"{model_names[j]}", c=C_PER[j])
        if j == len(model_true) - 1:
            if scatter_obs or i == 1:
                ax[i].scatter(x, model_true[0], label="True Obs.", s=10, c=C_AGG[1], marker=C_SHAPES[0])
            else:
                ax[i].plot(x, model_true[0], label="True Obs.", c=C_AGG[1])


# Panel configs for the three historical figure variants
PANEL_CONFIGS = {
    "full": {
        "panels": [
            {
                "key": "pheno",
                "fpaths": [
                    "_runs/PaperExperiments/Phenology/ParamMTL/Multi/param_zscore__1746932177",
                    "_runs/PaperExperiments/Phenology/DeepClass/Multi/class_mtl__1747001360",
                ],
                "names": ["DMC-MTL", "Deep-MTL"],
            },
            {
                "key": "ch",
                "fpaths": [
                    "_runs/PaperExperiments/ColdHardiness/ParamMTL/Multi/paramMTL__1746844099",
                    "_runs/PaperExperiments/ColdHardiness/DeepGCHNMTL/Multi/gchn_zscore__1746873252",
                ],
                "names": ["DMC-MTL", "Deep-MTL"],
            },
            {
                "key": "wf",
                "fpaths": [
                    "_runs/PaperExperiments/wofost/ParamMTL/Multi/param_mtl__1749924131",
                    "_runs/PaperExperiments/wofost/DeepMTL/Multi/deepMTL__1749324822",
                ],
                "names": ["DMC-MTL", "Deep-MTL"],
            },
        ],
        "outfile": "plotters/figs/domain_methods_comparison.pdf",
        "include_wofost": True,
    },
    "nowofost": {
        "panels": [
            {
                "key": "pheno",
                "fpaths": [
                    "_runs/PaperExperiments/Phenology/ParamMTL/All/param_zscore__1746932177",
                    "_runs/PaperExperiments/Phenology/DeepClass/All/class_mtl__1747001360",
                ],
                "names": ["DMC-MTL", "Deep-MTL"],
            },
            {
                "key": "ch",
                "fpaths": [
                    "_runs/PaperExperiments/ColdHardiness/ParamMTL/All/paramMTL__1746844099",
                    "_runs/PaperExperiments/ColdHardiness/DeepGCHNMTL/All/gchn_zscore__1746873252",
                ],
                "names": ["DMC-MTL", "Deep-MTL"],
            },
        ],
        "outfile": "plotters/figs/domain_methods_comparison_nowofost.png",
        "include_wofost": False,
    },
}
