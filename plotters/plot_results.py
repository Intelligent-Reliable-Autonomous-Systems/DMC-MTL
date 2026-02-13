"""
plot_results.py

Main plotting function for RNN-based methods
Handles the generation and processing of data on a per batch basis
and then calls correct plot functions

Written by Will Solow, 2025
"""

import argparse
import os

import numpy as np

import utils
from plotters.plot_utils import (
    gen_all_data_and_plot,
    gen_all_data_and_plot_error,
    gen_all_data_and_plot_transfer,
)
from plotters.plotting_functions import plot_loss, plot_stats


def main():
    argparser = argparse.ArgumentParser(description="Plotting script for model")
    argparser.add_argument("--config", type=str, required=True, help="Path to the config file")
    argparser.add_argument("--break_early", action="store_false", help="If to break early when making data")
    argparser.add_argument("--save", action="store_false", help="If to save the plots")
    argparser.add_argument("--fname", type=str, default="data.txt", help="File to save data to")
    argparser.add_argument("--rnn_name", type=str, default="rnn_model_best.pt")
    argparser.add_argument("--transfer_rate", type=float, default=0.0)
    argparser.add_argument(
        "--mode",
        default="default",
        choices=["default", "error", "transfer"],
        help="Plot mode: default, transfer, error",
    )
    np.set_printoptions(precision=2)

    args = argparser.parse_args()
    fpath = f"{os.getcwd()}/{args.config}"

    config, data, fpath = utils.load_config_data_fpath(args)
    config.DConfig.batch_size = 12
    # Plot tensorboard outputs
    try:
        plot_loss(fpath, config)
        plot_stats(fpath, config)
    except:
        pass
    # config.DConfig.batch_size = 4
    calibrator = utils.load_model_from_config(config, data)
    calibrator.load_model(f"{fpath}", name=args.rnn_name)
    calibrator.eval()

    # Setup Storage
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

    if args.mode == "default":
        plot_func = gen_all_data_and_plot
    elif args.mode == "error":
        plot_func = gen_all_data_and_plot_error
    elif args.mode == "transfer":
        plot_func = gen_all_data_and_plot_transfer

    # Generate all data
    plot_func(
        config,
        fpath,
        args,
        calibrator,
        true_data,
        output_data,
        true_cultivar_data,
        output_cultivar_data,
        name="train",
        transfer_rate=args.transfer_rate,
    )

    plot_func(
        config,
        fpath,
        args,
        calibrator,
        true_data,
        output_data,
        true_cultivar_data,
        output_cultivar_data,
        name="test",
        transfer_rate=args.transfer_rate,
    )
    if config.DataConfig.val_set:
        plot_func(
            config,
            fpath,
            args,
            calibrator,
            true_data,
            output_data,
            true_cultivar_data,
            output_cultivar_data,
            name="val",
            transfer_rate=args.transfer_rate,
        )


if __name__ == "__main__":
    main()
