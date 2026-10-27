"""
plot_results.py

Main plotting entry for RNN-based methods (single run).

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
from results.eval_common import empty_prediction_buffers


def main():
    parser = argparse.ArgumentParser(description="Plotting script for model")
    parser.add_argument("--config", type=str, required=True, help="Path to the run directory")
    parser.add_argument("--break_early", action="store_true", help="Stop after first batch")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True, help="Save PNGs (default on)")
    parser.add_argument("--rnn_name", type=str, default="rnn_model_best.pt")
    parser.add_argument("--transfer_rate", type=float, default=0.0)
    parser.add_argument("--transfer_type", type=str, default="direct")
    parser.add_argument(
        "--mode",
        default="default",
        choices=["default", "error", "transfer"],
        help="Plot mode: default, transfer, error",
    )
    np.set_printoptions(precision=2)
    args = parser.parse_args()

    fpath = f"{os.getcwd()}/{args.config}"
    config, data, fpath = utils.load_config_data_fpath(args)
    config.DConfig.batch_size = 12

    try:
        plot_loss(fpath, config)
        plot_stats(fpath, config)
    except Exception:
        pass

    calibrator = utils.load_model_from_config(config, data)
    calibrator.load_model(fpath, name=args.rnn_name)
    calibrator.eval()

    true_data, output_data, true_cultivar_data, output_cultivar_data = empty_prediction_buffers(calibrator)

    if args.mode == "default":
        plot_func = gen_all_data_and_plot
    elif args.mode == "error":
        plot_func = gen_all_data_and_plot_error
    else:
        plot_func = gen_all_data_and_plot_transfer

    for name in ["train", "test"]:
        plot_func(
            config,
            fpath,
            args,
            calibrator,
            true_data,
            output_data,
            true_cultivar_data,
            output_cultivar_data,
            name=name,
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
