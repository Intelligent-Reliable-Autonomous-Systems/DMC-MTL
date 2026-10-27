"""
plot_utils.py

Contains data generation functions used in plotting interfaces

Written by Will Solow, 2025
"""

from argparse import Namespace

import gpytorch
import numpy as np
import torch
import torch.nn.functional as F
from model_engine.util import PHENOLOGY_INT
from omegaconf import DictConfig
from torch import nn

from plotters.plotting_functions import (
    plot_output_coldhardiness,
    plot_output_coldhardiness_error,
    plot_output_coldhardiness_transfer,
    plot_output_phenology,
    plot_output_wofost,
)


def compute_obs_RMSE(true: list[np.ndarray], model: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes RMSE of per observation
    """
    if isinstance(true, np.ndarray):
        true = [true]
        model = [model]

    max_len = 0
    for t in true:
        max_len = max(max_len, len(t))

    samples = np.zeros(shape=max_len)
    avgs = np.zeros(shape=max_len)

    for i in range(len(true)):
        for j in range(len(true[i])):
            avgs[j] += (true[i][j] - model[i][j]) ** 2
            samples[j] += 1

    samples = np.where(samples == 0, 1, samples)  # ignore divde by zero

    return np.sqrt(avgs / samples), None


def compute_total_uq(true: list[np.ndarray], model: list[np.ndarray], z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """Compute the calibration and sharpness for uncertainty"""
    if isinstance(true, np.ndarray):
        true = [true]
        model = [model]
    avgs = 0
    samples = 0
    for i in range(len(true)):
        if model[i].shape[-1] == 2 and model[i].ndim > 1:
            lower = model[i][:, 0] - z * model[i][:, 1]
            upper = model[i][:, 0] + z * model[i][:, 1]
            coverage = np.mean((true[i] >= lower) & (true[i] <= upper))
            avgs += coverage
        else:
            continue
        samples += 1
    if samples == 0:
        return 0, None
    else:
        return avgs / samples, None


def compute_total_RMSE(true: list[np.ndarray], model: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes total RMSE across entire dataset
    """

    avgs = 0
    samples = 0
    if isinstance(true, np.ndarray):
        true = [true]
        model = [model]

    for i in range(len(true)):
        if model[i].shape[-1] == 2:
            try:
                avgs += np.sum((true[i] - model[i][:, 0]) ** 2)
            except:
                avgs += np.sum((true[i] - model[i][0]) ** 2)
        else:
            avgs += np.sum((true[i] - model[i]) ** 2)
        samples += len(true[i])
    if samples == 0:
        return 0, None
    else:
        return np.sqrt(avgs / samples), None


def compute_day_RMSE(
    true: list[np.ndarray], model: list[np.ndarray], inds: list[np.ndarray], days: int = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes RMSE up to a given day
    """

    avgs = 0
    samples = 0
    if isinstance(true, np.ndarray):
        true = [true]
        model = [model]

    for i in range(len(true)):
        k = (inds[i] >= days) * (inds[i] < (days + 30))
        avgs += np.sum((true[i][k] - model[i][k]) ** 2)
        samples += np.sum(k)
    if samples == 0:
        return 0
    else:
        return np.sqrt(avgs / samples)


def gen_batch_data(
    calibrator: nn.Module,
    input_data: torch.Tensor,
    dates: np.ndarray,
    val_data: torch.Tensor,
    cultivars: torch.Tensor,
    regions: torch.Tensor,
    stations: torch.Tensor,
    sites: torch.Tensor,
    days: int = None,
    name: str = None,
    transfer_rate: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates model output from a single batch
    """
    with torch.no_grad():
        output, params, _ = calibrator.forward(
            input_data,
            dates,
            cultivars=cultivars,
            val_data=val_data,
            regions=regions,
            stations=stations,
            sites=sites,
            days=days,
            transfer_rate=transfer_rate,
            name=name,
        )

    true_data = val_data.cpu().squeeze().numpy()
    if isinstance(output, gpytorch.distributions.MultivariateNormal):
        output = output.mean
    if output.size(-1) == len(PHENOLOGY_INT):
        probs = F.softmax(output, dim=-1)
        output = torch.argmax(probs, dim=-1)
    output_data = output.detach().cpu().squeeze().numpy()

    params = params.cpu().numpy().squeeze() if params is not None else None

    return true_data, output_data, params


def gen_batch_transfer_data(
    calibrator: nn.Module,
    input_data: torch.Tensor,
    dates: np.ndarray,
    val_data: torch.Tensor,
    cultivars: torch.Tensor,
    regions: torch.Tensor,
    stations: torch.Tensor,
    sites: torch.Tensor,
    all_ids: torch.Tensor,
    days: int = None,
    name: str = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates model output from a single batch
    """
    with torch.no_grad():
        # Output in shape (B, L, 1), Attn Map in shape (B, num_transfers, L, 1), Orig pred in shape (B, num_transfers, L, 1)
        output, attn_map, orig_pred = calibrator.forward(
            input_data,
            dates,
            cultivars=cultivars,
            val_data=val_data,
            regions=regions,
            stations=stations,
            sites=sites,
            days=days,
            transfer_rate=1.0,
            all_ids=all_ids,
            name=name,
        )

    true_data = val_data.cpu().squeeze().numpy()
    output_data = output.detach().cpu().squeeze().numpy()
    attn_map = attn_map.detach().cpu().squeeze().numpy()
    orig_pred = orig_pred.detach().cpu().squeeze().numpy()

    return true_data, output_data, attn_map, orig_pred


def _batch_ids(calibrator: nn.Module, name: str, i: int):
    """Slice cultivar/region/station/site ID tensors for one batch."""
    cultivars = calibrator.cultivars[name][i : i + calibrator.batch_size] if calibrator.cultivars is not None else None
    regions = calibrator.regions[name][i : i + calibrator.batch_size] if calibrator.regions is not None else None
    stations = calibrator.stations[name][i : i + calibrator.batch_size] if calibrator.stations is not None else None
    sites = calibrator.sites[name][i : i + calibrator.batch_size] if calibrator.sites is not None else None
    return cultivars, regions, stations, sites


def append_masked_batch(
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
    n: int,
    all_inds=None,
    cult_inds=None,
) -> None:
    """Normalize shapes and append masked series into aggregate + cultivar trees."""
    if len(true.shape) == 1:
        true = true[np.newaxis, :]
        output = output[np.newaxis, :]
    if true.shape[-1] == 3:
        true = true[:, :, 0]
        output = output[:, :, 0]

    for k in range(len(true)):
        true_data[n].append(true[k][inds[k]])
        output_data[n].append(output[k][inds[k]])
        if all_inds is not None:
            all_inds[n].append(torch.argwhere(inds[k]).flatten().cpu().numpy())

    cm = calibrator.nn.cult_mapping if hasattr(calibrator.nn, "cult_mapping") else [0, 0]
    rm = calibrator.nn.reg_mapping if hasattr(calibrator.nn, "reg_mapping") else [0, 0]
    sm = calibrator.nn.stat_mapping if hasattr(calibrator.nn, "stat_mapping") else [0, 0]
    sim = calibrator.nn.site_mapping if hasattr(calibrator.nn, "site_mapping") else [0, 0]

    for k in range(len(true)):
        ck = int(cultivars[k].item()) if cultivars is not None else 0
        rk = int(regions[k].item()) if regions is not None else 0
        sk = int(stations[k].item()) if stations is not None else 0
        sik = int(sites[k].item()) if sites is not None else 0
        true_cultivar_data[rm[rk]][sm[sk]][sim[sik]][cm[ck]][n].append(true[k][inds[k]])
        output_cultivar_data[rm[rk]][sm[sk]][sim[sik]][cm[ck]][n].append(output[k][inds[k]])
        if cult_inds is not None:
            cult_inds[rm[rk]][sm[sk]][sim[sik]][cm[ck]][n].append(
                torch.argwhere(inds[k]).flatten().cpu().numpy()
            )


def gen_all_data_and_plot(
    config: DictConfig,
    fpath: str,
    args: Namespace,
    calibrator: nn.Module,
    true_data: list[np.ndarray],
    output_data: list[np.ndarray],
    true_cultivar_data: list[np.ndarray],
    output_cultivar_data: list[np.ndarray],
    name: str = "train",
    days: int = None,
    all_inds: list[np.ndarray] = None,
    cult_inds: list[np.ndarray] = None,
    transfer_rate: float = 0.0,
    mode: str = "default",
) -> None:
    """
    Unified train/val/test data generation + plotting.

    mode: "default" | "error" | "transfer"
    """
    n = {"train": 0, "val": 1}.get(name, 2)
    dtype = config.DataConfig.dtype

    for i in range(0, len(calibrator.data[name]), calibrator.batch_size):
        cultivars, regions, stations, sites = _batch_ids(calibrator, name, i)
        batch_idx = np.arange(start=i, stop=i + calibrator.batch_size)
        val_slice = calibrator.val[name][i : i + calibrator.batch_size]

        if mode == "error":
            p, o, t = [], [], []
            for j in np.arange(0, 270, 30):
                true, output, params = gen_batch_data(
                    calibrator,
                    calibrator.data[name][i : i + calibrator.batch_size],
                    calibrator.dates[name][i : i + calibrator.batch_size],
                    val_slice,
                    cultivars,
                    regions,
                    stations,
                    sites,
                    days=torch.tensor(np.tile(j, calibrator.batch_size)).unsqueeze(1).to(calibrator.device),
                )
                p.append(params)
                o.append(output)
                t.append(true)
            if "phenology" in dtype:
                plot_output_phenology(
                    config, fpath, batch_idx, output, params, val_slice, name=name, save=args.save
                )
            elif "coldhardiness" in dtype:
                plot_output_coldhardiness_error(
                    config, fpath, batch_idx, np.array(o), np.array(p), val_slice, name=name, save=args.save
                )
            elif "wofost" in dtype:
                plot_output_wofost(config, fpath, batch_idx, output, params, val_slice, name=name, save=args.save)
            else:
                raise Exception(f"Unknown dtype `{dtype}`, cannot save data")
            # error mode: plot-only (no buffer append)

        elif mode == "transfer":
            if "coldhardiness" not in dtype:
                raise Exception(f"Unknown dtype `{dtype}`, cannot save data")
            true, output, attn_map, orig_pred = gen_batch_transfer_data(
                calibrator,
                calibrator.data[name][i : i + calibrator.batch_size],
                calibrator.dates[name][i : i + calibrator.batch_size],
                val_slice,
                cultivars,
                regions,
                stations,
                sites,
                calibrator.all_ids[name],
                days=days,
                name=name,
            )
            inds = plot_output_coldhardiness_transfer(
                config,
                fpath,
                batch_idx,
                output,
                val_slice,
                attn_map,
                orig_pred,
                calibrator.all_ids[name].cpu().numpy(),
                regions.cpu().numpy(),
                stations.cpu().numpy(),
                sites.cpu().numpy(),
                cultivars.cpu().numpy(),
                name=name,
                save=args.save,
            )
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
                n,
                all_inds=all_inds,
                cult_inds=cult_inds,
            )

        else:  # default
            true, output, params = gen_batch_data(
                calibrator,
                calibrator.data[name][i : i + calibrator.batch_size],
                calibrator.dates[name][i : i + calibrator.batch_size],
                val_slice,
                cultivars,
                regions,
                stations,
                sites,
                days=days,
                name=name,
                transfer_rate=transfer_rate,
            )
            if "phenology" in dtype:
                inds = plot_output_phenology(
                    config, fpath, batch_idx, output, params, val_slice, name=name, save=args.save
                )
            elif "coldhardiness" in dtype:
                inds = plot_output_coldhardiness(
                    config, fpath, batch_idx, output, params, val_slice, name=name, save=args.save
                )
            elif "wofost" in dtype:
                inds = plot_output_wofost(
                    config, fpath, batch_idx, output, params, val_slice, name=name, save=args.save
                )
            else:
                raise Exception(f"Unknown dtype `{dtype}`, cannot save data")
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
                n,
                all_inds=all_inds,
                cult_inds=cult_inds,
            )

        if args.break_early:
            break


def compute_biological_realism(list_arr: list[np.ndarray]):
    """Compute the percentage of biologically realistic predictions."""
    biologically_unreal = 0
    total_count = 0
    for arr in list_arr:
        for j in range(len(arr)):
            total_count += len(arr[j])
            biologically_unreal += np.count_nonzero(np.diff(arr[j, 170:]) > 0.8)


def gen_all_data_and_plot_error(*args, **kwargs):
    """Thin wrapper: day-horizon error plotting mode."""
    kwargs.pop("mode", None)
    return gen_all_data_and_plot(*args, mode="error", **kwargs)


def gen_all_data_and_plot_transfer(*args, **kwargs):
    """Thin wrapper: transfer attention plotting mode."""
    kwargs.pop("mode", None)
    return gen_all_data_and_plot(*args, mode="transfer", **kwargs)
