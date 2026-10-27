"""
Shared helpers for embedding extraction and counterfactual batch construction.

Written by Will Solow, 2025
"""

from __future__ import annotations

from argparse import Namespace

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from plotters.plot_utils import gen_batch_data


def unique_tol(x: torch.Tensor, tol: float = 1e5, dim: int = 0) -> torch.Tensor:
    q = torch.round(x * tol) / tol
    return torch.unique(q, dim=dim)


def collect_hierarchy_indices(calibrator: nn.Module):
    """Concatenate train+test hierarchy IDs and return unique combos + per-level uniques."""
    cultivars = torch.cat((calibrator.cultivars["train"], calibrator.cultivars["test"]), dim=0)
    sites = torch.cat((calibrator.sites["train"], calibrator.sites["test"]), dim=0)
    stations = torch.cat((calibrator.stations["train"], calibrator.stations["test"]), dim=0)
    regions = torch.cat((calibrator.regions["train"], calibrator.regions["test"]), dim=0)

    unq_cult = torch.unique(cultivars).cpu().numpy().flatten().astype(np.int32)
    unq_site = torch.unique(sites).cpu().numpy().flatten().astype(np.int32)
    unq_stat = torch.unique(stations).cpu().numpy().flatten().astype(np.int32)
    unq_reg = torch.unique(regions).cpu().numpy().flatten().astype(np.int32)

    all_inds = torch.unique(torch.cat((regions, stations, sites, cultivars), dim=1), dim=0)
    return all_inds, unq_reg, unq_stat, unq_site, unq_cult


def extract_embeds(calibrator: nn.Module, all_inds: torch.Tensor, e_mode: str = "direct"):
    """Dispatch forward_embed_* (or plain forward_embed when e_mode is None/'default')."""
    text_emb = calibrator.text_model if hasattr(calibrator, "text_model") else None
    kwargs = dict(
        regions=all_inds[:, 0:1],
        stations=all_inds[:, 1:2],
        sites=all_inds[:, 2:3],
        cultivars=all_inds[:, 3:],
    )
    with torch.no_grad():
        if e_mode in (None, "default", "forward"):
            return calibrator.nn.forward_embed(**kwargs, text_embedder=text_emb)
        if e_mode == "direct":
            return calibrator.nn.forward_embed_direct(**kwargs, text_embedder=text_emb)
        if e_mode == "text":
            return calibrator.nn.forward_embed_text(**kwargs, text_embedder=text_emb)
        if e_mode == "adapt":
            return calibrator.nn.forward_embed_adapt(**kwargs, calibrator=calibrator)
        raise ValueError(f"Unknown e_mode `{e_mode}`")


def build_counterfactual_batch(calibrator: nn.Module, name: str, i: int):
    """Build cultivar/site counterfactual batch for sample ``i`` (former plot_embeddings / hybrid)."""
    all_cultivars = torch.unique(calibrator.cultivars[name]).unsqueeze(1)
    all_sites = torch.unique(calibrator.sites[name]).unsqueeze(1)

    data = calibrator.data[name][i, None]
    dates = calibrator.dates[name][i, None]
    val = calibrator.val[name][i, None]
    cultivars = calibrator.cultivars[name][i, None]
    regions = calibrator.regions[name][i, None]
    stations = calibrator.stations[name][i, None]
    sites = calibrator.sites[name][i, None]

    c_data = torch.cat((data, data.repeat(all_cultivars.shape[0], 1, 1)), dim=0)
    c_dates = np.concatenate((dates, np.repeat(dates, all_cultivars.shape[0], axis=0)), axis=0)
    c_val = torch.cat((val, val.repeat(all_cultivars.shape[0], 1, 1)), dim=0)
    c_cultivars = torch.cat((cultivars, all_cultivars), dim=0)
    c_regions = torch.cat((regions, regions.repeat(all_cultivars.shape[0], 1)), dim=0)
    c_stations = torch.cat((stations, stations.repeat(all_cultivars.shape[0], 1)), dim=0)
    c_sites = torch.cat((sites, sites.repeat(all_cultivars.shape[0], 1)), dim=0)
    j = c_data.shape[0]

    s_data = torch.cat((data, data.repeat(all_sites.shape[0], 1, 1)), dim=0)
    s_dates = np.concatenate((dates, np.repeat(dates, all_sites.shape[0], axis=0)), axis=0)
    s_val = torch.cat((val, val.repeat(all_sites.shape[0], 1, 1)), dim=0)
    s_cultivars = torch.cat((cultivars, cultivars.repeat(all_sites.shape[0], 1)), dim=0)
    s_regions = torch.cat((regions, regions.repeat(all_sites.shape[0], 1)), dim=0)
    s_stations = torch.cat((stations, stations.repeat(all_sites.shape[0], 1)), dim=0)
    s_sites = torch.cat((sites, all_sites), dim=0)
    k = s_data.shape[0]

    c_inds = torch.argwhere(calibrator.cultivars["test"].flatten() == cultivars.flatten())
    s_regions_c = calibrator.regions["test"].flatten()[c_inds]
    s_stations_c = calibrator.stations["test"].flatten()[c_inds]
    s_sites_c = calibrator.sites["test"].flatten()[c_inds]
    s_cultivars_c = calibrator.cultivars["test"].flatten()[c_inds]
    b_inds_c = torch.unique(
        torch.concatenate((s_regions_c, s_stations_c, s_sites_c, s_cultivars_c), dim=1).to(torch.int64),
        dim=0,
    )
    s_data_c = data.repeat(b_inds_c.shape[0], 1, 1)
    s_dates_c = np.repeat(dates.reshape(1, -1), b_inds_c.shape[0], axis=0)
    s_val_c = val.repeat(b_inds_c.shape[0], 1, 1)
    m = b_inds_c.shape[0]

    si_inds = torch.argwhere(calibrator.sites["test"].flatten() == sites.flatten())
    s_regions_s = calibrator.regions["test"].flatten()[si_inds]
    s_stations_s = calibrator.stations["test"].flatten()[si_inds]
    s_sites_s = calibrator.sites["test"].flatten()[si_inds]
    s_cultivars_s = calibrator.cultivars["test"].flatten()[si_inds]
    b_inds_s = torch.unique(
        torch.concatenate((s_regions_s, s_stations_s, s_sites_s, s_cultivars_s), dim=1).to(torch.int64),
        dim=0,
    )
    s_data_s = data.repeat(b_inds_s.shape[0], 1, 1)
    s_dates_s = np.repeat(dates.reshape(1, -1), b_inds_s.shape[0], axis=0)
    s_val_s = val.repeat(b_inds_s.shape[0], 1, 1)
    l = b_inds_s.shape[0]

    data_out = torch.cat((c_data, s_data, s_data_s, s_data_c), dim=0)
    dates_out = np.concatenate((c_dates, s_dates, s_dates_s, s_dates_c), axis=0)
    val_out = torch.cat((c_val, s_val, s_val_s, s_val_c), dim=0)
    cultivars_out = torch.cat((c_cultivars, s_cultivars, b_inds_s[:, 3, None], b_inds_c[:, 3, None]), dim=0)
    regions_out = torch.cat((c_regions, s_regions, b_inds_s[:, 0, None], b_inds_c[:, 0, None]), dim=0)
    stations_out = torch.cat((c_stations, s_stations, b_inds_s[:, 1, None], b_inds_c[:, 1, None]), dim=0)
    sites_out = torch.cat((c_sites, s_sites, b_inds_s[:, 2, None], b_inds_c[:, 2, None]), dim=0)

    true, output, _ = gen_batch_data(
        calibrator,
        data_out,
        dates_out,
        val_out,
        cultivars_out,
        regions_out,
        stations_out,
        sites_out,
    )
    return (
        true,
        output,
        regions_out.cpu().numpy().astype(np.int32),
        stations_out.cpu().numpy().astype(np.int32),
        sites_out.cpu().numpy().astype(np.int32),
        cultivars_out.cpu().numpy().astype(np.int32),
        (j, k, l, m),
    )


def load_run_calibrator(args: Namespace):
    """Load config + calibrator for an embedding plot CLI (same 5-line pattern as former scripts)."""
    import utils

    config, data, fpath = utils.load_config_data_fpath(args)
    config.DConfig.batch_size = 128
    calibrator = utils.load_model_from_config(config, data)
    calibrator.load_model(f"{fpath}", name=args.rnn_name)
    calibrator.eval()
    return config, fpath, calibrator
