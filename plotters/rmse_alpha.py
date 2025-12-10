"""
rmse_alpha.py

Plot the percentage of cultivars under a given RMSE
as a function of threshold tolerance (RMSE)

Written by, Will Solow 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from model_engine.util import CROP_NAMES
from plotters.plotting_functions import C_AGG, C_PER
import utils
from pathlib import Path
import pickle


def load_named_pickles(folder_paths: list[str], target_name: str, exclude_multi: bool = False):
    """
    Load all pickle files matching a given name in all subdirectories.
    """
    results = {}

    for root in folder_paths:
        root = Path(f"./_runs/PaperExperiments/{root}")
        for pkl_file in root.rglob(target_name):
            try:
                # Get relative subdirectory name
                subdir = "/".join(pkl_file.parent.parts[-2:])
                if "Multi" in subdir and exclude_multi:
                    continue
                with open(pkl_file, "rb") as f:
                    results[subdir] = pickle.load(f)
            except Exception as e:
                pass

    return results


def main():

    num_cultivars = len(CROP_NAMES["grape_phenology_"])

    per_cultivar_models = [
        "Phenology/ParamMTL",
        "Phenology/DeepClass",
        "Phenology/PINN",
    ]
    agg_cultivar_models = ["Phenology/StationaryModel"]  # STL, exclude Multi

    mtl_models = load_named_pickles(per_cultivar_models, "results_per_cultivars.pkl")
    bio_models = load_named_pickles(agg_cultivar_models, "results_agg_cultivars.pkl", exclude_multi=True)

    mtl_array = np.array(list(mtl_models.values()))
    bio_array = np.array(list(bio_models.values())).reshape(len(agg_cultivar_models), num_cultivars, 8, 5)

    mtl_rmse = np.mean(mtl_array, axis=-1)[:, :, -1]
    bio_rmse = np.mean(bio_array, axis=-1)[:, :, -1]

    count_below = lambda arr, thresh: (arr < thresh).sum(axis=-1)

    thresholds = np.arange(start=0, stop=33, step=1)
    tick_size = 12
    mtl_rmse_alpha = np.array([count_below(mtl_rmse, t) / num_cultivars for t in thresholds])
    bio_rmse_alpha = np.array([count_below(bio_rmse, t) / num_cultivars for t in thresholds])

    fig, ax = plt.subplots(figsize=(6, 3))
    per_cultivar_names = ["DMC-MTL", "Deep-MTL", "PINN-MTL"]
    for i, m in enumerate(per_cultivar_names[: len(per_cultivar_models)]):
        ax.plot(thresholds, mtl_rmse_alpha[:, i], label=m, marker="x", c=C_PER[i])

    agg_cultivar_names = ["GDD Model"]
    for j, m in enumerate(agg_cultivar_names[: len(agg_cultivar_models)]):
        ax.plot(thresholds, bio_rmse_alpha[:, j], label=m, marker="x", c=C_AGG[j])

    ax.set_xlabel("Confidence Threshold: RMSE in Days", fontsize=tick_size)
    ax.set_ylabel(r"%" + " of Cultivars Satisfying the Desired\nPrediction Confidence Threshold", fontsize=tick_size)
    ax.set_title("Model Confidence: Per Cultivar Error")
    ax.legend(fontsize=12)
    ax.tick_params(axis="both", labelsize=tick_size)
    plt.savefig("plotters/figs/rmse_alpha.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
