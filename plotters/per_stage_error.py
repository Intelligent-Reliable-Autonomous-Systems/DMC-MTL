"""
per_stage_error.py

Plot the per-stage error of different phenology models
using box plots

Written by, Will Solow 2025
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from model_engine.util import CROP_NAMES
from plotters.plotting_functions import C_AGG, C_PER


def load_named_pickles(biomodel: str, folder_paths: list[str], target_name: str, exclude_multi: bool = False):
    """
    Load all pickle files matching a given name in all subdirectories.
    """
    results = {}

    for root in folder_paths:
        root = Path(f"./_runs/PaperExperiments/{biomodel}/{root}")
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--biomodel",
        type=str,
        default="Phenology",
        choices=["Phenology", "ColdHardiness", "wofost"],
    )
    args = parser.parse_args()

    num_cultivars = len(CROP_NAMES["grape_phenology"])

    # Load Models
    agg_cultivar_models = ["ParamMTL"]
    per_cultivar_models = ["StationaryModel"]
    max_pheno_models = ["ParamMTL"]
    mtl_model = load_named_pickles("Phenology", agg_cultivar_models, "results_per_cultivars.pkl")
    bio_models = load_named_pickles("Phenology", per_cultivar_models, "results_agg_cultivars.pkl", exclude_multi=True)
    mtl_array = np.array(list(mtl_model.values()))

    bio_array = np.array(list(bio_models.values())).reshape(len(agg_cultivar_models), num_cultivars, 8, 5)

    # Compute avg RMSE
    mtl_rmse = np.mean(mtl_array, axis=-1)[:, :, -4:]
    bio_rmse = np.mean(bio_array, axis=-1)[:, :, -4:]

    widths = 0.5
    fig, ax = plt.subplots(figsize=(6, 3))
    per_cultivar_names = ["DMC-MTL", "Classification", "Regression", "PINN", "DMC-Agg"]
    offsets = [-0.3, 0.3]
    scale = 1.5

    # Plot Param MTL
    for i, m in enumerate(per_cultivar_names[: len(per_cultivar_models)]):
        box = ax.boxplot(
            mtl_rmse[i], label=m, showfliers=False, positions=np.arange(4) * scale + offsets[i], widths=widths
        )
        for element in ["boxes", "whiskers", "caps", "medians"]:
            for item in box[element]:
                item.set_color(C_PER[i])



    for element in ["boxes", "whiskers", "caps", "medians"]:
        for item in box[element]:
            item.set_color(C_PER[i + 1])
    tick_size = 12
    # Plot GDD Model
    agg_cultivar_names = ["GDD Model", "DMC-STL"]
    for j, m in enumerate(agg_cultivar_names[: len(agg_cultivar_models)]):
        box = ax.boxplot(bio_rmse[j], label=m, showfliers=False, positions=np.arange(4) * scale, widths=widths)
        for element in ["boxes", "whiskers", "caps", "medians"]:
            for item in box[element]:
                item.set_color(C_AGG[j])
    ax.set_ylabel("RMSE in Days", fontsize=tick_size)
    ax.set_xticks(np.arange(4) * scale, ["Bud Break", "Bloom", "Veraison", "Cumulative"])
    ax.tick_params(axis="both", labelsize=tick_size)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, fontsize=12)
    ax.set_title("Per-Stage Cultivar Average Cultivar Error")
    plt.savefig("plotters/figs/per_stage_error.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
