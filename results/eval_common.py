"""
Shared evaluation helpers for compile_runs / plot_results / transfer compilers.

Written by Will Solow, 2025
"""

from __future__ import annotations

import argparse
import os
import pickle as pkl
from typing import Any

import numpy as np

import utils
from assets import CROP_NAMES, REGIONS, SITES, STATIONS
from plotters.plot_utils import (
    compute_error_percentile,
    compute_seasonal_max_error,
    compute_total_MAE,
    compute_total_RMSE,
    compute_total_uq,
)
from plotters.plotting_functions import compute_rmse_plot

EXTRA_METRICS = ("mae", "p90", "p95", "smax")


def _point_extra_metrics(true, model) -> dict[str, float]:
    """MAE / P90 / P95 / seasonal-max for one true/pred list pair."""
    mae, _ = compute_total_MAE(true, model)
    p90, _ = compute_error_percentile(true, model, q=90.0)
    p95, _ = compute_error_percentile(true, model, q=95.0)
    smax, _ = compute_seasonal_max_error(true, model)
    return {"mae": mae, "p90": p90, "p95": p95, "smax": smax}


def _split_extra_metrics(true_data, output_data) -> dict[str, np.ndarray]:
    """Train/val/test arrays for each extra metric."""
    per_split = [_point_extra_metrics(true_data[i], output_data[i]) for i in range(3)]
    return {key: np.array([per_split[i][key] for i in range(3)]) for key in EXTRA_METRICS}


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
    parser.add_argument(
        "--all", action=argparse.BooleanOptionalAction, default=False, help="If to compute over all region data"
    )
    if not transfer:
        parser.add_argument("--transfer_rate", type=float, default=0.0)
        parser.add_argument("--num_transfer_years", type=int, default=5)
        parser.add_argument("--transfer_type", type=str, default="direct")
        parser.add_argument("--verbose_level", type=str, default="verb")
        parser.add_argument(
            "--noise",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable K-day noisy forecast eval (DConfig.noise)",
        )
        parser.add_argument(
            "--forecast_noise",
            type=float,
            default=None,
            help="Gaussian scale on last-3 weather features (DConfig.forecast_noise)",
        )
        parser.add_argument(
            "--forecast_k",
            type=int,
            default=None,
            help="Horizon K (days) for noisy forecast rollouts (DConfig.forecast_k)",
        )
    return parser


def forecast_noise_pickle_suffix(
    noise=None,
    forecast_noise: float | None = None,
    forecast_k: int | None = None,
) -> str:
    """Filename fragment when noisy-forecast eval params are set, else ''."""
    parts: list[str] = []
    if noise is not None:
        parts.append(f"noise{int(bool(noise))}")
    if forecast_noise is not None:
        parts.append(f"fn{forecast_noise}")
    if forecast_k is not None:
        parts.append(f"k{forecast_k}")
    return ("_" + "_".join(parts)) if parts else ""


def apply_forecast_noise_config(config, args: argparse.Namespace) -> None:
    """Inject CLI forecast-noise flags into DConfig when provided."""
    if getattr(args, "noise", None) is not None:
        config.DConfig.noise = args.noise
    if getattr(args, "forecast_noise", None) is not None:
        config.DConfig.forecast_noise = args.forecast_noise
    if getattr(args, "forecast_k", None) is not None:
        config.DConfig.forecast_k = args.forecast_k


def iter_run_dirs(start_dir: str, prefix: str | None = None) -> list[str]:
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
    num_transfer_years: int | None = None,
    rnn_name: str | None = None,
):
    """Load config/data/model for args.config and load checkpoint."""
    config, data, fpath = utils.load_config_data_fpath(args)
    args.cultivar = config.DataConfig.cultivar
    config.DConfig.batch_size = batch_size
    if num_transfer_years is not None:
        config.DConfig.num_transfer_years = num_transfer_years
    apply_forecast_noise_config(config, args)
    calibrator = utils.load_model_from_config(config, data)
    calibrator.load_model(fpath, name=rnn_name or args.rnn_name)
    calibrator.eval()
    return config, data, fpath, calibrator


def alloc_result_arrays(dtype: str, num_runs: int, *, use_config_dtype: bool = True) -> dict[str, np.ndarray]:
    """Allocate aggregate and per-cultivar metric arrays for a dtype."""
    n_crops = len(CROP_NAMES[dtype])
    hier = (len(REGIONS), len(STATIONS), len(SITES), n_crops, 3, num_runs)
    arrays = {
        "all_avg_pheno": np.zeros((12, num_runs)),
        "all_avg_ch": np.zeros((3, num_runs)),
        "all_uq_ch": np.zeros((3, num_runs)),
        "all_avg_wf": np.zeros((3, num_runs)),
        "all_cultivar_avg_pheno": np.zeros((len(REGIONS), len(STATIONS), len(SITES), n_crops, 12, num_runs)),
        "all_cultivar_avg_ch": np.zeros(hier),
        "all_cultivar_uq_ch": np.zeros(hier),
        "all_cultivar_avg_wf": np.zeros(hier),
    }
    for m in EXTRA_METRICS:
        arrays[f"all_avg_{m}_ch"] = np.zeros((3, num_runs))
        arrays[f"all_cultivar_{m}_ch"] = np.zeros(hier)
        arrays[f"all_avg_{m}_wf"] = np.zeros((3, num_runs))
        arrays[f"all_cultivar_{m}_wf"] = np.zeros(hier)
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
        out.update(_split_extra_metrics(true_data, output_data))
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
        out.update(_split_extra_metrics(true_data, output_data))
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
                        t0 = true_cultivar_data[r][s][si][k][0]
                        t1 = true_cultivar_data[r][s][si][k][1]
                        t2 = true_cultivar_data[r][s][si][k][2]
                        o0 = output_cultivar_data[r][s][si][k][0]
                        o1 = output_cultivar_data[r][s][si][k][1]
                        o2 = output_cultivar_data[r][s][si][k][2]
                        tr, _ = compute_total_RMSE(t0, o0)
                        va, _ = compute_total_RMSE(t1, o1)
                        te, _ = compute_total_RMSE(t2, o2)
                        arrays["all_cultivar_avg_ch"][r, s, si, k, :, run_i] = np.array([tr, va, te])
                        extras = _split_extra_metrics([t0, t1, t2], [o0, o1, o2])
                        for m in EXTRA_METRICS:
                            arrays[f"all_cultivar_{m}_ch"][r, s, si, k, :, run_i] = extras[m]
                        if with_uq:
                            try:
                                uq_tr, _ = compute_total_uq(t0, o0)
                                uq_va, _ = compute_total_uq(t1, o1)
                                uq_te, _ = compute_total_uq(t2, o2)
                                arrays["all_cultivar_uq_ch"][r, s, si, k, :, run_i] = np.array([uq_tr, uq_va, uq_te])
                            except Exception:
                                pass
                    elif "wofost" in dtype:
                        t0 = true_cultivar_data[r][s][si][k][0]
                        t1 = true_cultivar_data[r][s][si][k][1]
                        t2 = true_cultivar_data[r][s][si][k][2]
                        o0 = output_cultivar_data[r][s][si][k][0]
                        o1 = output_cultivar_data[r][s][si][k][1]
                        o2 = output_cultivar_data[r][s][si][k][2]
                        tr, _ = compute_total_RMSE(t0, o0)
                        va, _ = compute_total_RMSE(t1, o1)
                        te, _ = compute_total_RMSE(t2, o2)
                        arrays["all_cultivar_avg_wf"][r, s, si, k, :, run_i] = np.array([tr, va, te])
                        extras = _split_extra_metrics([t0, t1, t2], [o0, o1, o2])
                        for m in EXTRA_METRICS:
                            arrays[f"all_cultivar_{m}_wf"][r, s, si, k, :, run_i] = extras[m]


def apply_split_scores(arrays: dict[str, np.ndarray], scores: dict[str, Any], run_i: int) -> None:
    kind = scores["kind"]
    if kind == "phenology":
        arrays["all_avg_pheno"][:, run_i] = scores["agg"]
    elif kind == "coldhardiness":
        arrays["all_avg_ch"][:, run_i] = scores["agg"]
        if "uq" in scores:
            arrays["all_uq_ch"][:, run_i] = scores["uq"]
        for m in EXTRA_METRICS:
            if m in scores:
                arrays[f"all_avg_{m}_ch"][:, run_i] = scores[m]
    elif kind == "wofost":
        arrays["all_avg_wf"][:, run_i] = scores["agg"]
        for m in EXTRA_METRICS:
            if m in scores:
                arrays[f"all_avg_{m}_wf"][:, run_i] = scores[m]


def save_result_pickles(
    start_dir: str,
    arrays: dict[str, np.ndarray],
    dtype: str,
    *,
    file_prefix: str = "",
    transfer_type: str | None = None,
    verbose_level: str | None = None,
    with_uq: bool = True,
    noise=None,
    forecast_noise: float | None = None,
    forecast_k: int | None = None,
) -> None:
    """
    Write agg/per pickles. Naming:
    - If transfer_type is set: results_*_{transfer_type}[_{verbose}][_noise…].pkl (compile_runs style)
    - Else: {file_prefix}results_*[_noise…].pkl (transfer strategy style)
    Also writes mae_/p90_/p95_/smax_ parallel files for CH and WOFOST.
    """
    os.makedirs(start_dir, exist_ok=True)
    noise_sfx = forecast_noise_pickle_suffix(noise, forecast_noise, forecast_k)

    def _dump(name: str, obj) -> None:
        with open(f"{start_dir}/{name}", "wb") as f:
            pkl.dump(obj, f)

    def _dump_extras(suffix: str, domain: str) -> None:
        """suffix e.g. '_direct_verb' or '_direct' or ''."""
        for m in EXTRA_METRICS:
            _dump(f"{file_prefix}{m}_agg_cultivars{suffix}.pkl", arrays[f"all_avg_{m}_{domain}"])
            _dump(f"{file_prefix}{m}_per_cultivars{suffix}.pkl", arrays[f"all_cultivar_{m}_{domain}"])

    if transfer_type is not None:
        if "phenology" in dtype:
            suffix = f"_{transfer_type}{noise_sfx}"
            _dump(f"{file_prefix}results_agg_cultivars{suffix}.pkl", arrays["all_avg_pheno"])
            _dump(f"{file_prefix}results_per_cultivars{suffix}.pkl", arrays["all_cultivar_avg_pheno"])
        elif "coldhardiness" in dtype:
            verb = verbose_level or "verb"
            suffix = f"_{transfer_type}_{verb}{noise_sfx}"
            _dump(f"{file_prefix}results_agg_cultivars{suffix}.pkl", arrays["all_avg_ch"])
            _dump(f"{file_prefix}results_per_cultivars{suffix}.pkl", arrays["all_cultivar_avg_ch"])
            _dump_extras(suffix, "ch")
            if with_uq:
                _dump(f"{file_prefix}uq_agg_cultivars{suffix}.pkl", arrays["all_uq_ch"])
                _dump(f"{file_prefix}uq_per_cultivars{suffix}.pkl", arrays["all_cultivar_uq_ch"])
        elif "wofost" in dtype:
            suffix = f"_{transfer_type}{noise_sfx}"
            _dump(f"{file_prefix}results_agg_cultivars{suffix}.pkl", arrays["all_avg_wf"])
            _dump(f"{file_prefix}results_per_cultivars{suffix}.pkl", arrays["all_cultivar_avg_wf"])
            _dump_extras(suffix, "wf")
    else:
        if "phenology" in dtype:
            _dump(f"{file_prefix}results_agg_cultivars{noise_sfx}.pkl", arrays["all_avg_pheno"])
            _dump(f"{file_prefix}results_per_cultivars{noise_sfx}.pkl", arrays["all_cultivar_avg_pheno"])
        elif "coldhardiness" in dtype:
            _dump(f"{file_prefix}results_agg_cultivars{noise_sfx}.pkl", arrays["all_avg_ch"])
            _dump(f"{file_prefix}results_per_cultivars{noise_sfx}.pkl", arrays["all_cultivar_avg_ch"])
            _dump_extras(noise_sfx, "ch")
        elif "wofost" in dtype:
            _dump(f"{file_prefix}results_agg_cultivars{noise_sfx}.pkl", arrays["all_avg_wf"])
            _dump(f"{file_prefix}results_per_cultivars{noise_sfx}.pkl", arrays["all_cultivar_avg_wf"])
            _dump_extras(noise_sfx, "wf")


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
