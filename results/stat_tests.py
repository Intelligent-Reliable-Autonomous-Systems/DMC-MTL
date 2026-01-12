"""
Docstring for results.stat_tests

Perform the Student T-Test and Wilcoxon Signed Rank Test on
model runs

Written by Will Solow, 2026
"""

import argparse
import numpy as np
from scipy.stats import ttest_rel, ttest_ind, wilcoxon

from results.load_results import load_named_pickles

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1", type=str, default="", help="Path to results .pkls")
    parser.add_argument("--model2", type=str, default="", help="Path to results .pkls")
    parser.add_argument("--prefix", type=str, help="prefix of results file", default="")
    parser.add_argument("--stl", action="store_true", help="If to toggle printing for STL variant")
    parser.add_argument("--station", action="store_true", help="If to toggle MTL printing variant by station")
    parser.add_argument("--site", action="store_true", help="If to toggle MTL printint variant by site")
    parser.add_argument("--cult", action="store_true", help="If to toggle MTL printint variant by cult")
    parser.add_argument("--per", action="store_false", help="If to load per cultivar or aggregate file")
    parser.add_argument("--per_cult", action="store_false", help="If to print per cultivar results")
    parser.add_argument("--rtmc", action="store_true", help="If to print RTMC data")
    args = parser.parse_args()

    fpath = args.prefix + "results_agg_cultivars.pkl"
    fpath = fpath.replace("agg", "per") if args.per else fpath
    mtl_models_1 = load_named_pickles([args.model1], fpath, args)
    mtl_models_2 = load_named_pickles([args.model2], fpath, args)

    sorted_keys_1 = np.argsort(list(mtl_models_1.keys()))  # Reorder based on alphabetical
    mtl_arr_1 = np.array(list(mtl_models_1.values()))[sorted_keys_1]
    mtl_arr_1 = np.where(mtl_arr_1 == 0, np.nan, mtl_arr_1)  # Replace 0.0s with nan
    sorted_keys_2 = np.argsort(list(mtl_models_2.keys()))  # Reorder based on alphabetical
    mtl_arr_2 = np.array(list(mtl_models_2.values()))[sorted_keys_2]
    mtl_arr_2 = np.where(mtl_arr_2 == 0, np.nan, mtl_arr_2)  # Replace 0.0s with nan

    if args.per:
        if args.per_cult:
            mean_1 = mtl_arr_1
            mean_2 = mtl_arr_2
        elif args.rtmc:
            mean_1 = np.round(np.nanmean(mtl_arr_1, axis=(-4)), decimals=2).squeeze()
            std_1 = np.round(np.nanstd(mtl_arr_1, axis=(-4)), decimals=2).squeeze()
            mean_2 = np.round(np.nanmean(mtl_arr_2, axis=(-4)), decimals=2).squeeze()
            std_2 = np.round(np.nanstd(mtl_arr_2, axis=(-4)), decimals=2).squeeze()
        else:
            mean_1 = np.round(np.nanmean(mtl_arr_1, axis=(-3)), decimals=2).squeeze()
            std_1 = np.round(np.nanstd(mtl_arr_1, axis=(-3)), decimals=2).squeeze()
            mean_2 = np.round(np.nanmean(mtl_arr_1, axis=(-3)), decimals=2).squeeze()
            std_2 = np.round(np.nanstd(mtl_arr_1, axis=(-3)), decimals=2).squeeze()
    else:
        mean_1 = mtl_arr_1
        mean_2 = mtl_arr_2

    # Ignore training data
    mean_1 = mean_1[:,:,:,:,:,-1,:]
    mean_2 = mean_2[:,:,:,:,:,-1,:]

    mean_1 = mean_1.reshape(-1,5)
    mean_2 = mean_2.reshape(-1,5)
    mask_1 = ~np.isnan(mean_1).any(axis=1)
    mask_2 = ~np.isnan(mean_2).any(axis=1)
    filt_mean_1 = mean_1[mask_1]
    filt_mean_2 = mean_2[mask_2]

    _, tt_pval = ttest_rel(filt_mean_1, filt_mean_2, axis=None)
    _, w_pval = wilcoxon(filt_mean_1.flatten(), filt_mean_2.flatten(), axis=None, alternative="two-sided")
    print("t-test P-Value: ", tt_pval)
    print("Wilcoxon P-Value: ", w_pval)

if __name__ == "__main__":
    main()

