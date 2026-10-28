"""
compile_runs_nyus.py

Compiles NYUS metrics across multiple runs and saves pickles
(same RMSE / MAE / P90 / P95 / smax layout as compile_runs.py).

Written by Will Solow, 2025
"""

import argparse
import os
from argparse import Namespace

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from torch import nn

import utils
from results.eval_common import (
    add_compile_args,
    alloc_result_arrays,
    apply_split_scores,
    empty_prediction_buffers,
    iter_run_dirs,
    save_result_pickles,
    score_per_cultivar,
    score_splits,
)
from train_algs.NYUS_2 import NYUS2


def gen_batch_data(
    calibrator: nn.Module,
    input_data: list[pd.DataFrame],
    val_data: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates model output from a single batch."""
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
    """Generates data for train/val/test splits for NYUS models."""
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
    parser = argparse.ArgumentParser(description="Compile NYUS RMSE/MAE/P90/P95 across runs")
    add_compile_args(parser)
    parser.add_argument("--all", action="store_true", help="If to compute over all region data")
    parser.add_argument("--data_fname", type=str, default="rmse.txt")
    np.set_printoptions(precision=2)
    args = parser.parse_args()

    config_dirs = iter_run_dirs(args.start_dir, args.prefix)
    if not config_dirs:
        raise SystemExit(f"No run dirs found under {args.start_dir}")

    args.config = f"{args.start_dir}/{config_dirs[0]}"
    config0, _, _ = utils.load_config_data_fpath(args, includes="nyus")
    arrays = alloc_result_arrays(config0.DataConfig.dtype, args.num_runs)
    last_dtype = config0.DataConfig.dtype

    run_i = 0
    for config_rel in config_dirs:
        print(config_rel)
        args.config = f"{args.start_dir}/{config_rel}"
        fpath = f"{os.getcwd()}/{args.config}"

        config, data, fpath = utils.load_config_data_fpath(args, includes="nyus")
        args.cultivar = config.DataConfig.cultivar
        config.DConfig.batch_size = 72
        calibrator = NYUS2(config, data)
        last_dtype = config.DataConfig.dtype

        true_data, output_data, true_cultivar_data, output_cultivar_data = empty_prediction_buffers(calibrator)

        for n in ["train", "test"]:
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

        scores = score_splits(config, true_data, output_data, fpath)
        apply_split_scores(arrays, scores, run_i)
        score_per_cultivar(
            config,
            calibrator,
            true_cultivar_data,
            output_cultivar_data,
            fpath,
            arrays,
            run_i,
            with_uq=True,
        )
        run_i += 1

    file_prefix = args.synth_test + "_" if args.synth_test is not None else ""
    save_result_pickles(
        args.start_dir,
        arrays,
        last_dtype,
        file_prefix=file_prefix,
        transfer_type=args.transfer_type,
        verbose_level=args.verbose_level,
        with_uq=True,
        noise=getattr(args, "noise", None),
        forecast_noise=getattr(args, "forecast_noise", None),
        forecast_k=getattr(args, "forecast_k", None),
    )


if __name__ == "__main__":
    main()
