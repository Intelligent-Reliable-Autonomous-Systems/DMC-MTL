"""
load_results.py

Loads data from pickle and prints statistics

Written by Will Solow, 2025
"""

from pathlib import Path
import pickle
import argparse
from argparse import Namespace
import numpy as np

from model_engine.util import CROP_NAMES


def load_named_pickles(folder_paths: list[str], target_name: str, args: Namespace):
    """
    Load all pickle files matching a given name in all subdirectories.
    """
    results = {}

    for root in folder_paths:
        root = Path(f"./{root}")
        for pkl_file in root.rglob(target_name):
            try:
                # Get relative subdirectory name
                subdir = "/".join(pkl_file.parent.parts[-4:])
                if "All" in pkl_file.parent.parts and args.stl:  # subdir
                    continue
                #print(subdir)
                if args.station and ("All" in pkl_file.parent.parts[-3] or "All" not in pkl_file.parent.parts[-2]):
                    continue
                if args.site and ("All" in pkl_file.parent.parts[-3] or "All" in pkl_file.parent.parts[-2]):
                    continue
                print(pkl_file)
                with open(pkl_file, "rb") as f:
                    results[subdir] = pickle.load(f)
            except Exception as e:
                pass
    return results


def compute_str_ndim5(mtl_arr: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """
    Make print string for when we are printing MTL result over regions
    """
    all_str = ""

    cult_mean = np.round(np.nanmean(mtl_arr, axis=(-3, -1)), decimals=2).squeeze()
    cult_std = np.round(np.nanstd(mtl_arr, axis=(-3, -1)), decimals=2).squeeze()
    site_mean = np.round(np.nanmean(mtl_arr, axis=(-4, -3, -1)), decimals=2).squeeze()
    site_std = np.round(np.nanstd(mtl_arr, axis=(-4, -3, -1)), decimals=2).squeeze()
    stat_mean = np.round(np.nanmean(mtl_arr, axis=(-5, -4, -3, -1)), decimals=2).squeeze()
    stat_std = np.round(np.nanstd(mtl_arr, axis=(-5, -4, -3, -1)), decimals=2).squeeze()
    reg_mean = np.round(np.nanmean(mtl_arr, axis=(-6, -5, -4, -3, -1)), decimals=2).squeeze()
    reg_std = np.round(np.nanstd(mtl_arr, axis=(-6, -5, -4, -3, -1)), decimals=2).squeeze()

    for j in range(mean.shape[0]):
        if (np.isnan(mean[j])).all():
            continue
        for k in range(mean.shape[1]):
            if (np.isnan(mean[j, k])).all():
                continue
            for l in range(mean.shape[2]):
                if (np.isnan(mean[j, k, l])).all():
                    continue
                for m in range(mean.shape[3]):
                    if (np.isnan(mean[j, k, l, m])).all():
                        continue
                    all_str += "        "
                    for i in range(mean.shape[4]):
                        all_str += f"{mean[j,k,l,m,i]} +/- {std[j,k,l,m,i]}, "
                    all_str += "\n"
                all_str += "      "
                for i in range(mean.shape[4]):
                    all_str += f"{cult_mean[j,k,l,i]} +/- {cult_std[j,k,l,i]}, "
                all_str += "\n"
                all_str += "\n"
            all_str += "    "
            for i in range(mean.shape[4]):
                all_str += f"{site_mean[j,k,i]} +/- {site_std[j,k,i]}, "
            all_str += "\n"
            all_str += "\n"
        all_str += "  "
        for i in range(mean.shape[4]):
            all_str += f"{stat_mean[j,i]} +/- {stat_std[j,i]}, "
        all_str += "\n"
        all_str += "\n"
    for i in range(mean.shape[4]):
        all_str += f"{reg_mean[i]} +/- {reg_std[i]}, "
    all_str += "\n"

    return all_str


def compute_str_ndim6(mtl_arr: np.ndarray, mean: np.ndarray, std: np.ndarray):
    all_str = ""
    for i in range(mtl_arr.shape[0]):
        all_str += compute_str_ndim5(mtl_arr[i], mean[i], std[i])
        all_str += "\n"
    all_mean = np.round(np.nanmean(mtl_arr, axis=(-7, -6, -5, -4, -3, -1)), decimals=2).squeeze()
    all_std = np.round(np.nanstd(mtl_arr, axis=(-7, -6, -5, -4, -3, -1)), decimals=2).squeeze()
    for i in range(all_mean.shape[0]):
        all_str += f"{all_mean[i]} +/- {all_std[i]}, "
    return all_str


def compute_str_stl_ndim3(mtl_arr: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """Compute print string for when loading cultivar level STL models"""
    all_str = ""
    for i in range(mean.shape[0]):
        if (np.isnan(mean[i])).all():
            continue
        for j in range(mean.shape[1]):
            if (np.isnan(mean[i, j])).all():
                continue
            for k in range(mean.shape[2]):
                all_str += f"{mean[i,j,k]} +/- {std[i,j,k]}, "
            all_str += "\n"
    all_mean = np.round(np.nanmean(mtl_arr, axis=(0, 1)), decimals=2)
    all_std = np.round(np.nanstd(mtl_arr, axis=(0, 1)), decimals=2)
    for j in range(all_mean.shape[1]):
        for i in range(all_mean.shape[0]):
            all_str += f"{all_mean[i,j]} +/- {all_std[i,j]}, "
        all_str += "\n"

    return all - str


def compute_str_stl_ndim2(mtl_arr: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """Compute print string for when loading single STL model"""
    all_str = ""
    for i in range(mean.shape[0]):
        if (np.isnan(mean[i])).all():
            continue
        for j in range(mean.shape[1]):
            all_str += f"{mean[i,j]} +/- {std[i,j]}, "
        all_str += "\n"
    all_mean = np.round(np.nanmean(mtl_arr, axis=(0, -1)), decimals=2)
    all_std = np.round(np.nanstd(mtl_arr, axis=(0, -1)), decimals=2)
    for i in range(all_mean.shape[0]):
        all_str += f"{all_mean[i]} +/- {all_std[i]}, "
    return all_str


def compute_all_mean(mtl_arr: np.ndarray, start: int, end: int):
    """
    Compute all mean of array with start and ends to split stations
    """
    all_str = ""
    all_mean = np.round(np.nanmean(mtl_arr[start:end], axis=(0, 1, -1)), decimals=2).squeeze()
    all_std = np.round(np.nanstd(mtl_arr[start:end], axis=(0, 1, -1)), decimals=2).squeeze()
    all_str += "\n"
    for i in range(all_mean.shape[0]):
        all_str += f"{all_mean[i]} +/- {all_std[i]}, "
    return all_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_dir", type=str, default="", help="Path to results .pkls")
    parser.add_argument("--prefix", type=str, help="prefix of results file", default="")
    parser.add_argument("--stl", action="store_true", help="If to toggle printing for STL variant")
    parser.add_argument("--station", action="store_true", help="If to toggle MTL printing variant by station")
    parser.add_argument("--site", action="store_true", help="If to toggle MTL printint variant by site")
    parser.add_argument("--cult2", action="store_true", help="If to toggle MTL printint variant by cult")
    parser.add_argument("--per", action="store_false", help="If to load per cultivar or aggregate file")
    parser.add_argument("--cult", action="store_true", help="If to print per cultivar results")
    args = parser.parse_args()

    fpath = args.prefix + "results_agg_cultivars.pkl"
    fpath = fpath.replace("agg", "per") if args.per else fpath
    mtl_models = load_named_pickles([args.start_dir], fpath, args)

    sorted_keys = np.argsort(list(mtl_models.keys()))  # Reorder based on alphabetical
    mtl_arr = np.array(list(mtl_models.values()))[sorted_keys]
    mtl_arr = np.where(mtl_arr == 0, np.nan, mtl_arr)  # Replace 0.0s with nan
    # Take average over runs and cultivars
    print(mtl_arr.shape)
    if args.per:
        if args.cult:
            mean = np.round(np.nanmean(mtl_arr, axis=-1), decimals=2).squeeze()
            std = np.round(np.nanstd(mtl_arr, axis=-1), decimals=2).squeeze()
        else:
            mean = np.round(np.nanmean(mtl_arr, axis=(-3, -1)), decimals=2).squeeze()
            std = np.round(np.nanstd(mtl_arr, axis=(-3, -1)), decimals=2).squeeze()
    else:
        mean = np.round(np.nanmean(mtl_arr, axis=-1), decimals=2).squeeze()
        std = np.round(np.nanstd(mtl_arr, axis=-1), decimals=2).squeeze()

    print(mean.shape)
    if args.cult:
        if args.stl:
            if mean.ndim == 3:
                all_str = compute_str_stl_ndim3(mtl_arr, mean, std)
            else:  # For when we are printing a single task cultivar model
                all_str = compute_str_stl_ndim2(mtl_arr, mean, std)
        elif args.site:
            all_str = ""
            for i in range(mean.shape[0]):
                for j in range(mean.shape[1]):
                    if (np.isnan(mean[i, j])).all():
                        continue
                    for k in range(mean.shape[2]):
                        all_str += f"{mean[i,j,k]} +/- {std[i,j,k]}, "
                    all_str += "\n"
                all_str += "\n"
            all_str += compute_all_mean(mtl_arr, 0, mtl_arr.shape[0])
            all_str += compute_all_mean(mtl_arr, 0, 4)
            all_str += compute_all_mean(mtl_arr, 4, 9)

        else:
            if mean.ndim == 6:
                all_str = compute_str_ndim6(mtl_arr, mean, std)
            elif mean.ndim == 5:  # For when we have region/station/site/cultivar
                all_str = compute_str_ndim5(mtl_arr, mean, std)
            elif mean.ndim == 3:  # For when we are loading multiple single task models
                for k in range(mean.shape[0]):
                    for j in range(mean.shape[2]):
                        for i in range(mean.shape[1]):
                            all_str += f"{mean[k,i,j]} +/- {std[k,i,j]}, "
                        all_str += "\n"
            else:  # For when we are printing a single task cultivar model
                all_str = compute_str_stl_ndim2(mtl_arr, mean, std)

    else:  # For when we are not loading individual cultivar data
        all_str = ""
        if mean.ndim == 4:  # For when we have region/station/site
            for j in range(mean.shape[0]):
                for k in range(mean.shape[1]):
                    for l in range(mean.shape[2]):
                        if (np.isnan(mean[j, k, l])).all():
                            continue
                        for i in range(mean.shape[3]):
                            all_str += f"{mean[j,k,l,i]} +/- {std[j,k,l,i]}, "
                        all_str += "\n"
                    all_str += "\n"
                all_str += "\n"
        elif mean.ndim == 3:  # For when we have region/station
            for j in range(mean.shape[0]):
                for k in range(mean.shape[1]):
                    if (np.isnan(mean[j, k])).all():
                        continue
                    for i in range(mean.shape[2]):
                        all_str += f"{mean[j,k,i]} +/- {std[j,k,i]}, "
                    all_str += "\n"
                all_str += "\n"
        elif mean.ndim == 2 and args.stl:  # For when we have multiple stl models
            for j in range(mean.shape[0]):
                for i in range(mean.shape[1]):
                    all_str += f"{mean[j,i]} +/- {std[j,i]}, "
                all_str += "\n"
            all_mean = np.round(np.nanmean(mtl_arr, axis=(0, 1, -1)), decimals=2)
            all_std = np.round(np.nanstd(mtl_arr, axis=(0, 1, -1)), decimals=2)
            for i in range(all_mean.shape[0]):
                all_str += f"{all_mean[i]} +/- {all_std[i]}, "
        else:  # For when we have just train/test/val
            for i in range(mean.shape[0]):
                all_str += f"{mean[i]} +/- {std[i]}, "
    print(all_str)


if __name__ == "__main__":
    main()
