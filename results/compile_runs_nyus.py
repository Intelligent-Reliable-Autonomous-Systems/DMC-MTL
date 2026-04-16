"""
compile_runs.py

Compiles RNN data across multiple runs and takes the average RMSE,
saving data to file

Written by Will Solow, 2025
"""

import argparse
import os
import pickle as pkl
from argparse import Namespace

import numpy as np
import pandas as pd
import torch.nn as nn
from omegaconf import DictConfig

import utils
from assets import CROP_NAMES, REGIONS, SITES, STATIONS
from plotters.plot_utils import compute_total_RMSE
from train_algs.NYUS_2 import NYUS2


def gen_batch_data(
    calibrator: nn.Module,
    input_data: list[pd.DataFrame],
    val_data: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates model output from a single batch
    """
    output = calibrator.forward(input_data)

    true_data = [v.squeeze() for v in val_data]

    output_data = [o.squeeze() for o in output]

    return true_data, output_data, None


def gen_all_data_and_plot_nyus(
    config: DictConfig,
    fpath: str,
    args: Namespace,
    calibrator: nn.Module,
    true_data: list[np.ndarray],
    output_data: list[np.ndarray],
    true_cultivar_data: list[np.ndarray],
    output_cultivar_data: list[np.ndarray],
    name: str = "train",
    all_inds: list[np.ndarray] = None,
    cult_inds: list[np.ndarray] = None,
) -> None:
    """
    Generates data for train and testing data and plots accordingly for RNNs
    """
    if name == "train":
        n = 0
    elif name == "val":
        n = 1
    else:  # test
        n = 2

    true_arr = []
    output_arr = []
    for i in range(0, len(calibrator.data[name]), calibrator.batch_size):
        print(f"Batch: {i}:{i + calibrator.batch_size}")
        cultivars = (
            calibrator.cultivars[name][i : i + calibrator.batch_size] if calibrator.cultivars is not None else None
        )
        regions = calibrator.regions[name][i : i + calibrator.batch_size] if calibrator.regions is not None else None
        stations = calibrator.stations[name][i : i + calibrator.batch_size] if calibrator.stations is not None else None
        sites = calibrator.sites[name][i : i + calibrator.batch_size] if calibrator.sites is not None else None

        true, output, params = gen_batch_data(
            calibrator,
            calibrator.data[name][i : i + calibrator.batch_size],
            calibrator.val[name][i : i + calibrator.batch_size],
        )
        true_arr.append(true)
        output_arr.append(output)

        ind_val_data = [v.squeeze() for v in calibrator.val[name][i : i + calibrator.batch_size]]
        inds = [~np.isnan(v) for v in ind_val_data]

        [true_data[n].append(true[k][inds[k]]) for k in range(len(true))]
        [output_data[n].append(output[k][inds[k]]) for k in range(len(output))]
        (
            [all_inds[n].append(np.argwhere(inds[k]).flatten()) for k in range(len(inds))]
            if all_inds is not None
            else None
        )

        cm = calibrator.cult_mapping
        rm = calibrator.reg_mapping
        sm = calibrator.stat_mapping
        sim = calibrator.site_mapping

        for k in range(len(true)):
            ck = int(cultivars[k].item()) if cultivars is not None else 0
            rk = int(regions[k].item()) if regions is not None else 0
            sk = int(stations[k].item()) if stations is not None else 0
            sik = int(sites[k].item()) if sites is not None else 0

            true_cultivar_data[rm[rk]][sm[sk]][sim[sik]][cm[ck]][n].append(true[k][inds[k]])
            output_cultivar_data[rm[rk]][sm[sk]][sim[sik]][cm[ck]][n].append(output[k][inds[k]])
            (
                cult_inds[rm[rk]][sm[sk]][sim[sik]][cm[ck]][n].append(np.argwhere(inds[k]).flatten())
                if cult_inds is not None
                else None
            )

        if args.break_early:
            break


def main():

    argparser = argparse.ArgumentParser(description="Plotting script for model")
    argparser.add_argument("--start_dir", type=str, default="_runs/")
    argparser.add_argument("--break_early", action="store_false", help="If to break early when making data")
    argparser.add_argument("--save", action="store_true", help="If to save the plots")
    argparser.add_argument("--config", type=str, default="", help="To be filled in at runtime")
    argparser.add_argument("--prefix", type=str, default=None, help="Prefix of files to search for")
    argparser.add_argument("--num_runs", type=int, default=5)
    argparser.add_argument("--data_fname", type=str, default="rmse.txt")
    argparser.add_argument("--rnn_name", type=str, default="rnn_model_best.pt")
    argparser.add_argument("--all", action="store_true", help="If to compute over all region data")
    np.set_printoptions(precision=2)
    args = argparser.parse_args()

    config_dirs = utils.find_config_yaml_dirs(start_dir=args.start_dir)

    # Setup total storage
    all_avg_ch = np.zeros((3, args.num_runs))
    args.config = f"{args.start_dir}/{config_dirs[0]}"
    config0, _, _ = utils.load_config_data_fpath(args, includes="nyus")

    all_cultivar_avg_ch = np.zeros(
        (
            len(REGIONS),
            len(STATIONS),
            len(SITES),
            len(CROP_NAMES[config0.DataConfig.dtype]),
            3,
            args.num_runs,
        )
    )

    for i, config in enumerate(config_dirs):
        print(config)
        if args.prefix is not None:
            if args.prefix not in config:
                continue

        args.config = f"{args.start_dir}/{config}"
        fpath = f"{os.getcwd()}/{args.config}"

        config, data, fpath = utils.load_config_data_fpath(args, includes="nyus")
        args.cultivar = config.DataConfig.cultivar
        config.DConfig.batch_size = 72
        calibrator = NYUS2(config, data)

        # Setup per run storage
        true_data = [[], [], []]
        output_data = [[], [], []]
        true_cultivar_data = [
            [
                [[[[], [], []] for _ in range(calibrator.num_cultivars)] for _ in range(calibrator.num_sites)]
                for _ in range(calibrator.num_stations)
            ]
            for _ in range(calibrator.num_regions)
        ]
        output_cultivar_data = [
            [
                [[[[], [], []] for _ in range(calibrator.num_cultivars)] for _ in range(calibrator.num_sites)]
                for _ in range(calibrator.num_stations)
            ]
            for _ in range(calibrator.num_regions)
        ]
        # Generate all data
        [
            gen_all_data_and_plot_nyus(
                config,
                fpath,
                args,
                calibrator,
                true_data,
                output_data,
                true_cultivar_data,
                output_cultivar_data,
                name=n,
            )
            for n in ["train", "test"]
        ]

        if config.DataConfig.val_set:
            gen_all_data_and_plot_nyus(
                config,
                fpath,
                args,
                calibrator,
                true_data,
                output_data,
                true_cultivar_data,
                output_cultivar_data,
                name="val",
            )

        # Store data for averaging

        if "coldhardiness" in config.DataConfig.dtype:
            total_rmse, _ = compute_total_RMSE(true_data[0], output_data[0])
            val_total_rmse, _ = compute_total_RMSE(true_data[1], output_data[1])
            test_total_rmse, _ = compute_total_RMSE(true_data[2], output_data[2])
            all_avg_ch[:, i] = np.array([total_rmse, val_total_rmse, test_total_rmse])

        for r in range(calibrator.num_regions):
            for s in range(calibrator.num_stations):
                for si in range(calibrator.num_sites):
                    for k in range(calibrator.num_cultivars):
                        if (
                            len(true_cultivar_data[r][s][si][k][0]) == 0
                            and len(true_cultivar_data[r][s][si][k][2]) == 0
                            and (len(true_cultivar_data[r][s][si][k][1]) == 0 and config.DataConfig.val_set)
                        ):
                            continue

                        if "coldhardiness" in config.DataConfig.dtype:
                            cultivar_train_rmse, _ = compute_total_RMSE(
                                true_cultivar_data[r][s][si][k][0],
                                output_cultivar_data[r][s][si][k][0],
                            )
                            cultivar_val_rmse, _ = compute_total_RMSE(
                                true_cultivar_data[r][s][si][k][1],
                                output_cultivar_data[r][s][si][k][1],
                            )
                            cultivar_test_rmse, _ = compute_total_RMSE(
                                true_cultivar_data[r][s][si][k][2],
                                output_cultivar_data[r][s][si][k][2],
                            )

                            all_cultivar_avg_ch[r, s, si, k, :, i] = np.array(
                                [
                                    cultivar_train_rmse,
                                    cultivar_val_rmse,
                                    cultivar_test_rmse,
                                ]
                            )

    # Save Data
    prefix = ""

    if "coldhardiness" in config.DataConfig.dtype:
        with open(f"{args.start_dir}/{prefix}results_agg_cultivars.pkl", "wb") as f:
            pkl.dump(all_avg_ch, f)
        f.close()
        with open(f"{args.start_dir}/{prefix}results_per_cultivars.pkl", "wb") as f:
            pkl.dump(all_cultivar_avg_ch, f)
        f.close()


if __name__ == "__main__":
    main()
