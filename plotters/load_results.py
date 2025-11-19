"""
load_results.py

Loads data from pickle and prints statistics

Written by Will Solow, 2025
"""

from pathlib import Path
import pickle
import argparse
import numpy as np

from model_engine.util import CROP_NAMES


def load_named_pickles(folder_paths: list[str], target_name: str, exclude_multi: bool = False):
    """
    Load all pickle files matching a given name in all subdirectories.
    """
    results = {}

    for root in folder_paths:
        root = Path(f"./{root}")
        for pkl_file in root.rglob(target_name):
            try:
                # Get relative subdirectory name
                subdir = "/".join(pkl_file.parent.parts[-2:])
                if "All" in subdir and exclude_multi:
                    continue
                with open(pkl_file, "rb") as f:
                    results[subdir] = pickle.load(f)
            except Exception as e:
                pass

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ParamMTL", help="Name of model")
    parser.add_argument("--prefix", type=str, help="prefix of results file", default="")
    parser.add_argument("--stl", action="store_true", help="If to toggle printing for STL variant")
    parser.add_argument("--per", action="store_false", help="If to load per cultivar or aggregate file")
    parser.add_argument("--cult", action="store_true", help="If to print per cultivar results")
    parser.add_argument("--total", action="store_true", help="If to print only final error")
    parser.add_argument("--obs", action="store_true", help="If to load sparse observation data")
    args = parser.parse_args()

    per_cultivar_models = [args.config]  # "DeepClass", "DeepRegression", "PINN", "STL/Multi"
    fpath = args.prefix + ("results_agg_cultivars.pkl" if not args.obs else "results_obs_agg_cultivars.pkl")
    fpath = fpath.replace("agg", "per") if args.per else fpath
    multi_flag = True if args.stl else False
    mtl_models = load_named_pickles(per_cultivar_models, fpath, exclude_multi=multi_flag)

    sorted_keys = np.argsort(list(mtl_models.keys()))
    mtl_arr = np.array(list(mtl_models.values()))[sorted_keys]
    mtl_arr = np.where(mtl_arr < 0, np.nan, mtl_arr)
    mtl_arr = np.transpose(mtl_arr, (0, 1, 2, 4, 3)) if args.obs else mtl_arr
    mtl_arr = np.where(mtl_arr == 0, np.nan, mtl_arr) # Replace 0.0s with nan
    if args.per:
        if args.cult:
            mean = np.round(np.nanmean(mtl_arr, axis=-1), decimals=2).squeeze()
            std = np.round(np.nanstd(mtl_arr, axis=-1), decimals=2).squeeze()
        else:
            mean = np.round(np.nanmean(mtl_arr, axis=(1, -1)), decimals=2).squeeze()
            std = np.round(np.nanstd(mtl_arr, axis=(1, -1)), decimals=2).squeeze()
    else:
        mean = np.round(np.nanmean(mtl_arr, axis=-1), decimals=2).squeeze()
        std = np.round(np.nanstd(mtl_arr, axis=-1), decimals=2).squeeze()
    all_str = ""
    if args.cult:
        if mean.ndim == 3 and not args.stl:
            for k in range(mean.shape[0]):
                for j in range(mean.shape[2]):
                    if args.total and j != mean.shape[2] - 1:
                        continue
                    for i in range(mean.shape[1]):
                        all_str += f"{mean[k,i,j]} +/- {std[k,i,j]}, "
                    all_str += "\n"
        else:
            if mean.ndim == 3:
                for k in range(mean.shape[0]):
                    for j in range(mean.shape[2]):
                        if args.total and j != mean.shape[2] - 1:
                            continue
                        for i in range(mean.shape[1]):
                            all_str += f"{mean[k,i,j]} +/- {std[k,i,j]}, "
                        all_str += "\n"
                all_mean = np.round(np.nanmean(mtl_arr, axis=(0, -1)), decimals=2)
                all_std = np.round(np.nanstd(mtl_arr, axis=(0, -1)), decimals=2)
                for j in range(all_mean.shape[1]):
                    for i in range(len(all_mean)):
                        all_str += f"{all_mean[i,j]} +/- {all_std[i,j]}, "
                    all_str += "\n"
            else:
                for i in range(len(mean)):
                    for j in range(len(mean[i])):
                        all_str += f"{mean[i,j]} +/- {std[i,j]}, "
                    all_str += "\n"
                all_mean = np.round(np.nanmean(mtl_arr, axis=(0, -1)), decimals=2)
                all_std = np.round(np.nanstd(mtl_arr, axis=(0, -1)), decimals=2)
                for i in range(len(all_mean)):
                    all_str += f"{all_mean[i]} +/- {all_std[i]}, "
    else:
        if len(mean.shape) == 2:
            for j in range(mean.shape[1]):
                for i in range(mean.shape[0]):
                    all_str += f"{mean[i,j]} +/- {std[i,j]}, "
                all_str += "\n"
        else:
            for i in range(len(mean)):
                all_str += f"{mean[i]} +/- {std[i]}, "
    print(all_str)


if __name__ == "__main__":
    main()
