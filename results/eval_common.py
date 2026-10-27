"""
Shared evaluation helpers for compile_runs / plot_results / transfer compilers.

Written by Will Solow, 2025
"""

from __future__ import annotations

import argparse
import os
import pickle as pkl
from typing import Any, Optional

import numpy as np

import utils
from assets import CROP_NAMES, REGIONS, SITES, STATIONS
from plotters.plot_utils import compute_total_RMSE, compute_total_uq
from plotters.plotting_functions import compute_rmse_plot


def add_compile_args(parser: argparse.ArgumentParser, *, transfer: bool = False) -> argparse.ArgumentParser:
    """Shared CLI flags for multi-run compile scripts."""
    parser.add_argument("--start_dir", type=str, default="_runs/")
    parser.add_argument("--break_early", action="store_true", help="Stop after first batch when generating data")
    parser.add_argument("--save", action="store_true", help="Save per-batch plots")
    parser.add_argument("--config", type=str, default="", help="Filled per run at runtime")
    parser.add_argument("--prefix", type=str, default=None, help="Substring filter on run dir names")
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--synth_test", type=str, default=None, help="Prefix of synthetic testing data")
    parser.add_argument("--rnn_name", type=str, default="rnn_model_best.pt")
    if not transfer:
        parser.add_argument("--transfer_rate", type=float, default=0.0)
        parser.add_argument("--num_transfer_years", type=int, default=5)
        parser.add_argument("--transfer_type", type=str, default="direct")
        parser.add_argument("--verbose_level", type=str, default="verb")
    return parser


def iter_run_dirs(start_dir: str, prefix: Optional[str] = None) -> list[str]:
    """Return relative run dirs under start_dir that contain config.yaml, optionally filtered."""
    dirs = utils.find_config_yaml_dirs(start_dir=start_dir)
    if prefix is None:
        return dirs
    return [d for d in dirs if prefix in d]


def empty_prediction_buffers(calibrator) -> tuple[list, list, list, list]:
    """Allocate nested true/output buffers shared by compile_runs and plot_results."""
    true_data: list = [[], [], []]
    output_data: list = [[], [], []]
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
    return true_data, output_data, true_cultivar_data, output_cultivar_data


def load_run_calibrator(
    args: argparse.Namespace,
    *,
    batch_size: int = 72,
    num_transfer_years: Optional[int] = None,
    rnn_name: Optional[str] = None,
):
    """Load config/data/model for args.config and load checkpoint."""
    config, data, fpath = utils.load_config_data_fpath(args)
    args.cultivar = config.DataConfig.cultivar
    config.DConfig.batch_size = batch_size
    if num_transfer_years is not None:
        config.DConfig.num_transfer_years = num_transfer_years
    calibrator = utils.load_model_from_config(config, data)
    calibrator.load_model(fpath, name=rnn_name or args.rnn_name)
    calibrator.eval()
    return config, data, fpath, calibrator


def alloc_result_arrays(dtype: str, num_runs: int, *, use_config_dtype: bool = True) -> dict[str, np.ndarray]:
    """Allocate aggregate and per-cultivar metric arrays for a dtype."""
    n_crops = len(CROP_NAMES[dtype])
    arrays = {
        "all_avg_pheno": np.zeros((12, num_runs)),
        "all_avg_ch": np.zeros((3, num_runs)),
        "all_uq_ch": np.zeros((3, num_runs)),
        "all_avg_wf": np.zeros((3, num_runs)),
        "all_cultivar_avg_pheno": np.zeros((len(REGIONS), len(STATIONS), len(SITES), n_crops, 12, num_runs)),
        "all_cultivar_avg_ch": np.zeros((len(REGIONS), len(STATIONS), len(SITES), n_crops, 3, num_runs)),
        "all_cultivar_uq_ch": np.zeros((len(REGIONS), len(STATIONS), len(SITES), n_crops, 3, num_runs)),
        "all_cultivar_avg_wf": np.zeros((len(REGIONS), len(STATIONS), len(SITES), n_crops, 3, num_runs)),
    }
    return arrays


def _pheno_row(train_avg, val_avg, test_avg) -> np.ndarray:
    return np.concatenate(
        (
            train_avg[1:-1],
            [np.sum(train_avg[1:-1])],
            val_avg[1:-1],
            [np.sum(val_avg[1:-1])],
            test_avg[1:-1],
            [np.sum(test_avg[1:-1])],
        )
    )


def score_splits(config, true_data, output_data, fpath: str) -> dict[str, Any]:
    """Score train/val/test aggregates; returns keys depending on dtype."""
    dtype = config.DataConfig.dtype
    out: dict[str, Any] = {}
    if "phenology" in dtype:
        train_avg, _ = compute_rmse_plot(config, true_data[0], output_data[0], fpath, save=False)
        val_avg, _ = compute_rmse_plot(config, true_data[1], output_data[1], fpath, name="val", save=False)
        test_avg, _ = compute_rmse_plot(config, true_data[2], output_data[2], fpath, name="test", save=False)
        out["agg"] = _pheno_row(train_avg, val_avg, test_avg)
        out["kind"] = "phenology"
    elif "coldhardiness" in dtype:
        total_rmse, _ = compute_total_RMSE(true_data[0], output_data[0])
        val_total_rmse, _ = compute_total_RMSE(true_data[1], output_data[1])
        test_total_rmse, _ = compute_total_RMSE(true_data[2], output_data[2])
        out["agg"] = np.array([total_rmse, val_total_rmse, test_total_rmse])
        out["kind"] = "coldhardiness"
        try:
            total_uq, _ = compute_total_uq(true_data[0], output_data[0])
            val_total_uq, _ = compute_total_uq(true_data[1], output_data[1])
            test_total_uq, _ = compute_total_uq(true_data[2], output_data[2])
            out["uq"] = np.array([total_uq, val_total_uq, test_total_uq])
        except Exception:
            pass
    elif "wofost" in dtype:
        total_rmse, _ = compute_total_RMSE(true_data[0], output_data[0])
        val_total_rmse, _ = compute_total_RMSE(true_data[1], output_data[1])
        test_total_rmse, _ = compute_total_RMSE(true_data[2], output_data[2])
        out["agg"] = np.array([total_rmse, val_total_rmse, test_total_rmse])
        out["kind"] = "wofost"
    else:
        raise ValueError(f"Unrecognized dtype `{dtype}`")
    return out


def score_per_cultivar(
    config,
    calibrator,
    true_cultivar_data,
    output_cultivar_data,
    fpath: str,
    arrays: dict[str, np.ndarray],
    run_i: int,
    *,
    with_uq: bool = True,
) -> None:
    """Fill per-cultivar metric arrays for one run index."""
    dtype = config.DataConfig.dtype
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
                    if "phenology" in dtype:
                        tr, _ = compute_rmse_plot(
                            config,
                            true_cultivar_data[r][s][si][k][0],
                            output_cultivar_data[r][s][si][k][0],
                            fpath,
                            save=False,
                        )
                        va, _ = compute_rmse_plot(
                            config,
                            true_cultivar_data[r][s][si][k][1],
                            output_cultivar_data[r][s][si][k][1],
                            fpath,
                            name="val",
                            save=False,
                        )
                        te, _ = compute_rmse_plot(
                            config,
                            true_cultivar_data[r][s][si][k][2],
                            output_cultivar_data[r][s][si][k][2],
                            fpath,
                            name="test",
                            save=False,
                        )
                        arrays["all_cultivar_avg_pheno"][r, s, si, k, :, run_i] = _pheno_row(tr, va, te)
                    elif "coldhardiness" in dtype:
                        tr, _ = compute_total_RMSE(
                            true_cultivar_data[r][s][si][k][0], output_cultivar_data[r][s][si][k][0]
                        )
                        va, _ = compute_total_RMSE(
                            true_cultivar_data[r][s][si][k][1], output_cultivar_data[r][s][si][k][1]
                        )
                        te, _ = compute_total_RMSE(
                            true_cultivar_data[r][s][si][k][2], output_cultivar_data[r][s][si][k][2]
                        )
                        arrays["all_cultivar_avg_ch"][r, s, si, k, :, run_i] = np.array([tr, va, te])
                        if with_uq:
                            try:
                                uq_tr, _ = compute_total_uq(
                                    true_cultivar_data[r][s][si][k][0], output_cultivar_data[r][s][si][k][0]
                                )
                                uq_va, _ = compute_total_uq(
                                    true_cultivar_data[r][s][si][k][1], output_cultivar_data[r][s][si][k][1]
                                )
                                uq_te, _ = compute_total_uq(
                                    true_cultivar_data[r][s][si][k][2], output_cultivar_data[r][s][si][k][2]
                                )
                                arrays["all_cultivar_uq_ch"][r, s, si, k, :, run_i] = np.array([uq_tr, uq_va, uq_te])
                            except Exception:
                                pass
                    elif "wofost" in dtype:
                        tr, _ = compute_total_RMSE(
                            true_cultivar_data[r][s][si][k][0], output_cultivar_data[r][s][si][k][0]
                        )
                        va, _ = compute_total_RMSE(
                            true_cultivar_data[r][s][si][k][1], output_cultivar_data[r][s][si][k][1]
                        )
                        te, _ = compute_total_RMSE(
                            true_cultivar_data[r][s][si][k][2], output_cultivar_data[r][s][si][k][2]
                        )
                        arrays["all_cultivar_avg_wf"][r, s, si, k, :, run_i] = np.array([tr, va, te])


def apply_split_scores(arrays: dict[str, np.ndarray], scores: dict[str, Any], run_i: int) -> None:
    kind = scores["kind"]
    if kind == "phenology":
        arrays["all_avg_pheno"][:, run_i] = scores["agg"]
    elif kind == "coldhardiness":
        arrays["all_avg_ch"][:, run_i] = scores["agg"]
        if "uq" in scores:
            arrays["all_uq_ch"][:, run_i] = scores["uq"]
    elif kind == "wofost":
        arrays["all_avg_wf"][:, run_i] = scores["agg"]


def save_result_pickles(
    start_dir: str,
    arrays: dict[str, np.ndarray],
    dtype: str,
    *,
    file_prefix: str = "",
    transfer_type: Optional[str] = None,
    verbose_level: Optional[str] = None,
    with_uq: bool = True,
) -> None:
    """
    Write agg/per pickles. Naming:
    - If transfer_type is set: results_*_{transfer_type}[_{verbose}].pkl (compile_runs style)
    - Else: {file_prefix}results_*.pkl (transfer strategy style)
    """
    os.makedirs(start_dir, exist_ok=True)

    def _dump(name: str, obj) -> None:
        with open(f"{start_dir}/{name}", "wb") as f:
            pkl.dump(obj, f)

    if transfer_type is not None:
        if "phenology" in dtype:
            _dump(f"{file_prefix}results_agg_cultivars_{transfer_type}.pkl", arrays["all_avg_pheno"])
            _dump(f"{file_prefix}results_per_cultivars_{transfer_type}.pkl", arrays["all_cultivar_avg_pheno"])
        elif "coldhardiness" in dtype:
            verb = verbose_level or "verb"
            _dump(f"{file_prefix}results_agg_cultivars_{transfer_type}_{verb}.pkl", arrays["all_avg_ch"])
            _dump(f"{file_prefix}results_per_cultivars_{transfer_type}_{verb}.pkl", arrays["all_cultivar_avg_ch"])
            if with_uq:
                _dump(f"{file_prefix}uq_agg_cultivars_{transfer_type}_{verb}.pkl", arrays["all_uq_ch"])
                _dump(f"{file_prefix}uq_per_cultivars_{transfer_type}_{verb}.pkl", arrays["all_cultivar_uq_ch"])
        elif "wofost" in dtype:
            _dump(f"{file_prefix}results_agg_cultivars_{transfer_type}.pkl", arrays["all_avg_wf"])
            _dump(f"{file_prefix}results_per_cultivars_{transfer_type}.pkl", arrays["all_cultivar_avg_wf"])
    else:
        if "phenology" in dtype:
            _dump(f"{file_prefix}results_agg_cultivars.pkl", arrays["all_avg_pheno"])
            _dump(f"{file_prefix}results_per_cultivars.pkl", arrays["all_cultivar_avg_pheno"])
        elif "coldhardiness" in dtype:
            _dump(f"{file_prefix}results_agg_cultivars.pkl", arrays["all_avg_ch"])
            _dump(f"{file_prefix}results_per_cultivars.pkl", arrays["all_cultivar_avg_ch"])
        elif "wofost" in dtype:
            _dump(f"{file_prefix}results_agg_cultivars.pkl", arrays["all_avg_wf"])
            _dump(f"{file_prefix}results_per_cultivars.pkl", arrays["all_cultivar_avg_wf"])


def append_masked_predictions(
    true_data,
    output_data,
    true_cultivar_data,
    output_cultivar_data,
    calibrator,
    true,
    output,
    inds,
    a_inds,
    *,
    split_idx: int = 2,
    name: str = "test",
) -> None:
    """Append masked series into aggregate + cultivar trees (transfer compilers).

    Resolves hierarchy IDs from ``a_inds`` then delegates to
    ``plotters.plot_utils.append_masked_batch``.
    """
    import torch
    from plotters.plot_utils import append_masked_batch

    idx = a_inds.to(torch.int32)
    cultivars = calibrator.cultivars[name][idx] if calibrator.cultivars is not None else None
    regions = calibrator.regions[name][idx] if calibrator.regions is not None else None
    stations = calibrator.stations[name][idx] if calibrator.stations is not None else None
    sites = calibrator.sites[name][idx] if calibrator.sites is not None else None
    append_masked_batch(
        true_data,
        output_data,
        true_cultivar_data,
        output_cultivar_data,
        calibrator,
        true,
        output,
        inds,
        cultivars,
        regions,
        stations,
        sites,
        split_idx,
    )


def parse_withold(config) -> tuple[list, list, list, list]:
    wd = config.DataConfig.withold
    wd_keys = list(wd.keys())
    regions = wd["Region"] if "Region" in wd_keys else [None]
    stations = wd["Station"] if "Station" in wd_keys else [None]
    sites = wd["Site"] if "Site" in wd_keys else [None]
    cultivars = wd["Cultivar"] if "Cultivar" in wd_keys else [None]
    return regions, stations, sites, cultivars
