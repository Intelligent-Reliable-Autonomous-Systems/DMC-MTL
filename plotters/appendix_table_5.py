"""
per_stage_error.py

Plot the per-stage error of different phenology models
using box plots

Written by, Will Solow 2025
"""

import argparse
import numpy as np
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--biomodel",
        type=str,
        default="Phenology",
        choices=["Phenology", "ColdHardiness", "wofost"],
    )
    args = parser.parse_args()

    # Load Models
    agg_cultivar_models = ["ParamMTL"]
    per_cultivar_models = ["STL"]
    mtl_model = utils.load_named_pickles(args.biomodel, agg_cultivar_models, "results_per_cultivars.pkl")
    stl_model = utils.load_named_pickles(
        args.biomodel, per_cultivar_models, "results_agg_cultivars.pkl", exclude_multi=True
    )
    sorted_keys = np.argsort(list(mtl_model.keys()))
    mtl_array = np.array(list(mtl_model.values()))
    sorted_keys = np.argsort(list(stl_model.keys()))

    stl_array = np.array(list(stl_model.values()))[sorted_keys][np.newaxis, :]

    # Compute avg RMSE

    mtl_array = mtl_array[0, :, -1, :]
    stl_array = stl_array[0, :, -1, :]
    if args.biomodel == "ColdHardiness":
        mtl_array = np.delete(mtl_array, 1, axis=0)  # delete cab franc
        stl_array = np.delete(stl_array, 1, axis=0)

    # Per run error
    mask = mtl_array < stl_array
    perc_mean = np.mean(mask, axis=(0, -1))
    perc_std = np.std(mask, axis=(0, -1))

    avg_better = (stl_array - mtl_array)[mask]
    avg_worse = (mtl_array - stl_array)[~mask]

    print(f"Percent Improve: {perc_mean} +/- {perc_std}")
    print(f"Avg Improve: {np.mean(avg_better)} +/- {np.std(avg_better)}")
    print(f"Avg Unimprove: {np.mean(avg_worse)} +/- {np.std(avg_worse)}")

    mtl_array = np.mean(mtl_array, axis=-1)
    stl_array = np.mean(stl_array, axis=-1)
    print(mtl_array)
    print(stl_array)

    mask = mtl_array < stl_array
    perc_mean = np.mean(mask)
    perc_std = np.std(mask)

    avg_better = (stl_array - mtl_array)[mask]
    avg_worse = (mtl_array - stl_array)[~mask]

    print(f"Percent Improve: {perc_mean} +/- {perc_std}")
    print(f"Avg Improve: {np.mean(avg_better)} +/- {np.std(avg_better)}")
    print(f"Avg Unimprove: {np.mean(avg_worse)} +/- {np.std(avg_worse)}")


if __name__ == "__main__":
    main()
