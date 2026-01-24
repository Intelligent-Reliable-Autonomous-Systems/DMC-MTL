"""
plot_embeddings.py

Plots a variation over the embeddings for trained models

Written by Will Solow, 2025
"""

import os
import numpy as np
import argparse
import torch
from argparse import Namespace
from omegaconf import DictConfig
import torch.nn as nn
import matplotlib.pyplot as plt

import utils

from plotters.plot_utils import gen_batch_data
from model_engine.util import CROP_NAMES, REGIONS, STATIONS, SITES


def plot(
    config: DictConfig,
    fpath: str,
    calibrator: nn.Module,
    true: np.ndarray,
    output: np.ndarray,
    regions: np.ndarray,
    stations: np.ndarray,
    sites: np.ndarray,
    cultivars: np.ndarray,
    inds: tuple[int],
) -> None:
    """
    Plot the embeddings
    """

    fig, ax = plt.subplots(4, figsize=(18, 24))
    fig.subplots_adjust(top=0.98)
    fig.subplots_adjust(bottom=0.02)
    for k in range(inds[0]):
        cult = CROP_NAMES["grape_coldhardiness_reg"][cultivars[k]]
        reg = REGIONS[regions[k]]
        stat = STATIONS[stations[k]]
        site = SITES[sites[k]]

        if k == 0:
            ax[0].plot(output[k], label=f"PRED: {stat} {site} {cult}")
            ax[3].plot(output[k], label=f"PRED: {stat} {site} {cult}")
            ax[0].scatter(np.arange(len(true[k])), true[k], label=f"OBS: {stat} {site} {cult}")
            ax[3].scatter(np.arange(len(true[k])), true[k], label=f"OBS: {stat} {site} {cult}")
            orig_site = site
            orig_stat = stat
            orig_cult = cult
        else:
            if site == orig_site and orig_stat == stat and orig_cult == cult:
                continue
            ax[0].plot(output[k], alpha=0.5, label=f"EVAL: {stat} {site} {cult}")

    for k in range(sum(inds[:3]), sum(inds[:4])):
        cult = CROP_NAMES["grape_coldhardiness_reg"][cultivars[k]]
        reg = REGIONS[regions[k]]
        stat = STATIONS[stations[k]]
        site = SITES[sites[k]]
        if site == orig_site and orig_stat == stat and orig_cult == cult:
            continue
        else:
            ax[3].plot(output[k], alpha=0.5, label=f"EVAL: {stat} {site} {cult}")

    for k in range(inds[0], sum(inds[:2])):
        cult = CROP_NAMES["grape_coldhardiness_reg"][cultivars[k]]
        reg = REGIONS[regions[k]]
        stat = STATIONS[stations[k]]
        site = SITES[sites[k]]
        if k == inds[0]:
            ax[1].plot(output[k], label=f"PRED: {stat} {site} {cult}")
            ax[2].plot(output[k], label=f"PRED: {stat} {site} {cult}")
            ax[1].scatter(np.arange(len(true[k])), true[k], label=f"OBS: {stat} {site} {cult}")
            ax[2].scatter(np.arange(len(true[k])), true[k], label=f"OBS: {stat} {site} {cult}")
            orig_site = site
            orig_stat = stat
            orig_cult = cult
        else:
            if site == orig_site and orig_stat == stat and orig_cult == cult:
                continue
            ax[1].plot(output[k], alpha=0.5, label=f"EVAL: {stat} {site} {cult}")

    for k in range(sum(inds[:2]), sum(inds[:3])):
        cult = CROP_NAMES["grape_coldhardiness_reg"][cultivars[k]]
        reg = REGIONS[regions[k]]
        stat = STATIONS[stations[k]]
        site = SITES[sites[k]]

        if site == orig_site and orig_stat == stat and orig_cult == cult:
            continue
        else:
            ax[2].plot(output[k], alpha=0.5, label=f"EVAL: {stat} {site} {cult}")

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(
        f"{fpath}/Embed_{cultivars[0][0]}_{sites[0][0]}_{config.DataConfig.cultivar}.png",
        bbox_inches="tight",
    )
    plt.close()


def gen_embeddings_and_plot(
    config: DictConfig,
    fpath: str,
    args: Namespace,
    calibrator: nn.Module,
    name: str = "train",
) -> None:
    """
    Generates data for train and testing data and plots accordingly for RNNs
    """
    all_cultivars = torch.unique(calibrator.cultivars[name]).unsqueeze(1)
    all_regions = torch.unique(calibrator.regions[name]).unsqueeze(1)
    all_stations = torch.unique(calibrator.stations[name]).unsqueeze(1)
    all_sites = torch.unique(calibrator.sites[name]).unsqueeze(1)
    for i in range(0, len(calibrator.data[name])):

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

        data = torch.cat((c_data, s_data, s_data_s, s_data_c), dim=0)
        dates = np.concatenate((c_dates, s_dates, s_dates_s, s_dates_c), axis=0)
        val = torch.cat((c_val, s_val, s_val_s, s_val_c), dim=0)
        cultivars = torch.cat((c_cultivars, s_cultivars, b_inds_s[:, 3, None], b_inds_c[:, 3, None]), dim=0)
        regions = torch.cat((c_regions, s_regions, b_inds_s[:, 0, None], b_inds_c[:, 0, None]), dim=0)
        stations = torch.cat((c_stations, s_stations, b_inds_s[:, 1, None], b_inds_c[:, 1, None]), dim=0)
        sites = torch.cat((c_sites, s_sites, b_inds_s[:, 2, None], b_inds_c[:, 2, None]), dim=0)

        true, output, _ = gen_batch_data(
            calibrator,
            data,
            dates,
            val,
            cultivars,
            regions,
            stations,
            sites,
        )

        regions = regions.cpu().numpy().astype(np.int32)
        stations = stations.cpu().numpy().astype(np.int32)
        sites = sites.cpu().numpy().astype(np.int32)
        cultivars = cultivars.cpu().numpy().astype(np.int32)
        if "grape_coldhardiness" in config.DataConfig.dtype:
            plot(config, fpath, calibrator, true, output, regions, stations, sites, cultivars, (j, k, l, m))

        if args.break_early:
            break


def main():

    argparser = argparse.ArgumentParser(description="Plotting script for model")
    argparser.add_argument("--config", type=str, required=True, help="Path to the config file")
    argparser.add_argument("--break_early", action="store_false", help="If to break early when making data")
    argparser.add_argument("--save", action="store_false", help="If to save the plots")
    argparser.add_argument("--fname", type=str, default="data.txt", help="File to save data to")
    argparser.add_argument("--rnn_name", type=str, default="rnn_model_best.pt")
    argparser.add_argument(
        "--mode",
        default="default",
        choices=["default", "error"],
        help="Plot mode: default, rollout, error",
    )
    np.set_printoptions(precision=2)

    args = argparser.parse_args()
    fpath = f"{os.getcwd()}/{args.config}"

    config, data, fpath = utils.load_config_data_fpath(args)
    config.DConfig.batch_size = 128
    calibrator = utils.load_model_from_config(config, data)
    calibrator.load_model(f"{fpath}", name=args.rnn_name)
    calibrator.eval()

    # Generate all data
    gen_embeddings_and_plot(
        config,
        fpath,
        args,
        calibrator,
        name="test",
    )


if __name__ == "__main__":
    main()
