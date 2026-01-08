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

    num_cultivars_pheno = len(CROP_NAMES["grape_phenology_"])
    num_cultivars_ch = len(CROP_NAMES["grape_coldhardiness_ferg"])

    per_cultivar_pheno_models = [
        "Phenology/ParamMTL",
        "Phenology/DeepClass",
        "Phenology/PINN",
    ]
    agg_cultivar_pheno_models = ["Phenology/StationaryModel"]  # STL, exclude Multi

    per_cultivar_ch_models = [
        "ColdHardiness/ParamMTL",
        "ColdHardiness/DeepGCHNMTL",
        "ColdHardiness/PINNMTL",
    ]
    agg_cultivar_ch_models = ["ColdHardiness/StationaryModel"]  # STL, exclude Multi

    pheno_mtl_models = load_named_pickles(per_cultivar_pheno_models, "results_per_cultivars.pkl")
    pheno_bio_models = load_named_pickles(agg_cultivar_pheno_models, "results_agg_cultivars.pkl", exclude_multi=True)

    ch_mtl_models = load_named_pickles(per_cultivar_ch_models, "results_per_cultivars.pkl")
    ch_bio_models = load_named_pickles(agg_cultivar_ch_models, "results_agg_cultivars.pkl", exclude_multi=True)

    pheno_mtl_array = np.array(list(pheno_mtl_models.values()))
    pheno_bio_array = np.array(list(pheno_bio_models.values())).reshape(
        len(agg_cultivar_pheno_models), num_cultivars_pheno, 8, 5
    )

    ch_mtl_array = np.array(list(ch_mtl_models.values()))
    ch_bio_array = np.array(list(ch_bio_models.values())).reshape(
        len(agg_cultivar_pheno_models), num_cultivars_ch, 2, 5
    )

    pheno_mtl_rmse = np.mean(pheno_mtl_array, axis=-1)[:, :, -1]
    pheno_bio_rmse = np.mean(pheno_bio_array, axis=-1)[:, :, -1]

    ch_mtl_rmse = np.mean(ch_mtl_array, axis=-1)[:, :, -1]
    ch_bio_rmse = np.mean(ch_bio_array, axis=-1)[:, :, -1]

    count_below = lambda arr, thresh: (arr < thresh).sum(axis=-1)

    thresholds_pheno = np.arange(start=0, stop=33, step=1)
    thresholds_ch = np.arange(start=0, stop=2.7, step=0.1)
    font_size = 12
    pheno_mtl_rmse_alpha = np.array([count_below(pheno_mtl_rmse, t) / num_cultivars_pheno for t in thresholds_pheno])
    pheno_bio_rmse_alpha = np.array([count_below(pheno_bio_rmse, t) / num_cultivars_pheno for t in thresholds_pheno])

    ch_mtl_rmse_alpha = np.array([count_below(ch_mtl_rmse, t) / num_cultivars_ch for t in thresholds_ch])
    ch_bio_rmse_alpha = np.array([count_below(ch_bio_rmse, t) / num_cultivars_ch for t in thresholds_ch])

    fig, ax = plt.subplots(2, figsize=(6, 4))
    per_cultivar_names = ["DMC-MTL", "Deep-MTL", "PINN-MTL"]
    for i, m in enumerate(per_cultivar_names[: len(per_cultivar_pheno_models)]):
        ax[0].plot(thresholds_pheno, pheno_mtl_rmse_alpha[:, i], label=m, marker="x", c=C_PER[i])
        ax[1].plot(thresholds_ch, ch_mtl_rmse_alpha[:, i], marker="x", c=C_PER[i])

    agg_cultivar_names = ["GDD Model"]
    for j, m in enumerate(agg_cultivar_names[: len(agg_cultivar_pheno_models)]):
        ax[0].plot(thresholds_pheno, pheno_bio_rmse_alpha[:, j], label=m, marker="x", c=C_AGG[j])
        ax[1].plot(thresholds_ch, ch_bio_rmse_alpha[:, j], marker="x", c=C_AGG[j])

    # ax[0].set_xlabel("Confidence Threshold: RMSE in Days", fontsize=font_size)
    ax[1].set_xlabel(
        r"Confidence Threshold: $\mathbf{(a)}$ RMSE in days, $\mathbf{(b)}$ RMSE in $^\circ$C", fontsize=font_size
    )
    fig.supylabel(r"%" + " of Cultivars Satisfying the Desired\nPrediction Confidence Threshold", fontsize=font_size)
    fig.subplots_adjust(left=0.15)
    # fig.subplots_adjust(hspace=0.3)
    # ax[0].legend(fontsize=12)
    # ax[1].legend(fontsize=12, loc="lower right")
    fig.legend(
        loc="upper center",
        # bbox_to_anchor=(0.5, 1.05),
        ncol=4,
        frameon=True,
        fontsize=font_size,
        columnspacing=0.8,  # default ≈ 2.0
        handletextpad=0.4,
        handlelength=1.2,
    )

    fig.subplots_adjust(top=0.92)
    ax[0].tick_params(axis="both", labelsize=font_size)
    ax[1].tick_params(axis="both", labelsize=font_size)
    ax[0].set_yticks([0, 0.5, 1.0])
    ax[1].set_yticks([0, 0.5, 1.0])
    ax[0].text(
        0.05, 0.3, f"(a)", ha="left", va="top", transform=ax[0].transAxes, fontsize=font_size + 1, fontweight="bold"
    )
    ax[1].text(
        0.05, 0.3, f"(b)", ha="left", va="top", transform=ax[1].transAxes, fontsize=font_size + 1, fontweight="bold"
    )
    plt.savefig("plotters/figs/rmse_alpha.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
