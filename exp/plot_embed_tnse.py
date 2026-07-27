"""
plot_embeddings.py

Plots a variation over the embeddings for trained models

Written by Will Solow, 2025
"""

import argparse
import os
from argparse import Namespace
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from matplotlib.patches import Patch
from omegaconf import DictConfig
from torch import nn

import utils
from assets import CROP_NAMES, REGIONS, SITES, STATIONS
from plotters.plot_utils import gen_batch_data

markers = ["o", "4", "^", "v", "+", "d", "3", "x", "*", "p", "h"]


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
    tsne: sklearn.manifold.TSNE,
    embed: np.ndarray,
    idx: np.ndarray,
    dtype: str,
    type: Literal["cult", "site", "reg", "stat"],
) -> None:
    """
    Plot the embeddings
    """

    fig, ax = plt.subplots(2, figsize=(6, 5))
    # fig.subplots_adjust(top=0.98)
    # fig.subplots_adjust(bottom=0.02)
    for k in range(inds[0]):
        cult = CROP_NAMES["grape_coldhardiness_reg"][cultivars[k]]
        reg = REGIONS[regions[k]]
        stat = STATIONS[stations[k]]
        site = SITES[sites[k]]

        if k == 0:
            ax[0].plot(output[k])
            orig_stat = stat
            orig_cult = cult
            orig_site = site
        else:
            if site == orig_site and orig_stat == stat and orig_cult == cult:
                continue
            ax[0].plot(output[k], alpha=0.5)

        tsne_embed = tsne.fit_transform(embed)

    n = 34
    cmap = plt.colormaps["gist_ncar"]
    cmap = [cmap(i / n) for i in range(n)]
    for i, xy in enumerate(tsne_embed):
        if type == "all":
            c = cmap[idx[i, 2]]
            m = markers[idx[i, 3] % 11]
            ax[1].scatter(
                tsne_embed[i, 0],
                tsne_embed[i, 1],
                color=c,
                marker=m,
                s=30,
                facecolors="none",
                linewidths=1.5,
                edgecolors=c,
            )

            labels = [f"{SITES[i]}" for i in range(len(SITES))]

            handles = [Patch(facecolor=cmap[i], label=label) for i, label in enumerate(labels)]
            # ax[1].legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax[0].set_xlabel("Days since Dormancy Season Start (September 7th)")
    ax[0].set_ylabel(r"LTE50 $^\circ$C")
    ax[1].set_xlabel("t-SNE Dimension 1")
    ax[1].set_ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    # plt.show()
    plt.savefig(
        f"{fpath}/Embed_{cultivars[0][0]}_{sites[0][0]}_{config.DataConfig.cultivar}.png",
        bbox_inches="tight",
    )
    plt.close()


def gen_embeddings(
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
    for i in range(len(calibrator.data[name])):
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
        return config, fpath, calibrator, true, output, regions, stations, sites, cultivars, (j, k, l, m)


def unique_tol(x: torch.Tensor, tol: float = 1e5, dim: int = 0):
    q = torch.round(x * tol) / tol
    return torch.unique(q, dim=dim)


def gen_tsne(
    config: DictConfig,
    fpath: str,
    args: Namespace,
    calibrator: nn.Module,
    name: str = "train",
) -> None:
    """
    Generates data for train and testing data and plots accordingly for RNNs
    """
    cultivars = torch.cat((calibrator.cultivars["train"], calibrator.cultivars["test"]), dim=0)
    sites = torch.cat((calibrator.sites["train"], calibrator.sites["test"]), dim=0)
    stations = torch.cat((calibrator.stations["train"], calibrator.stations["test"]), dim=0)
    regions = torch.cat((calibrator.regions["train"], calibrator.regions["test"]), dim=0)

    unq_cult = torch.unique(cultivars).cpu().numpy().flatten().astype(np.int32)
    unq_site = torch.unique(sites).cpu().numpy().flatten().astype(np.int32)
    unq_stat = torch.unique(stations).cpu().numpy().flatten().astype(np.int32)
    unq_reg = torch.unique(regions).cpu().numpy().flatten().astype(np.int32)

    all_inds = torch.cat((regions, stations, sites, cultivars), dim=1)
    all_inds = torch.unique(all_inds, dim=0)

    text_emb = calibrator.text_model if hasattr(calibrator, "text_model") else None
    with torch.no_grad():
        if args.e_mode == "direct":
            reg_embed, stat_embed, site_embed, cult_embed = calibrator.nn.forward_embed_direct(
                regions=all_inds[:, 0:1],
                stations=all_inds[:, 1:2],
                sites=all_inds[:, 2:3],
                cultivars=all_inds[:, 3:],
                text_embedder=text_emb,
            )
        elif args.e_mode == "text":
            reg_embed, stat_embed, site_embed, cult_embed = calibrator.nn.forward_embed_text(
                regions=all_inds[:, 0:1],
                stations=all_inds[:, 1:2],
                sites=all_inds[:, 2:3],
                cultivars=all_inds[:, 3:],
                text_embedder=text_emb,
            )
        elif args.e_mode == "adapt":
            reg_embed, stat_embed, site_embed, cult_embed = calibrator.nn.forward_embed_adapt(
                regions=all_inds[:, 0:1],
                stations=all_inds[:, 1:2],
                sites=all_inds[:, 2:3],
                cultivars=all_inds[:, 3:],
                calibrator=calibrator,
            )

    tsne = sklearn.manifold.TSNE(n_components=args.n_tsne, perplexity=5.0)
    all_emds = torch.cat((reg_embed, stat_embed, site_embed, cult_embed), dim=1).cpu().numpy()
    unq_inds = all_inds.cpu().numpy().astype(np.int32)
    unq_reg_embed = unique_tol(reg_embed).cpu().numpy()
    unq_stat_embed = unique_tol(stat_embed).cpu().numpy()
    unq_site_embed = unique_tol(site_embed).cpu().numpy()
    unq_cult_embed = unique_tol(cult_embed).cpu().numpy()

    return tsne, all_emds, unq_inds, config.DataConfig.dtype, f"{args.config}/all_embed.png", "all"


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
    )
    argparser.add_argument("--e_mode", type=str, default="direct", help="Type of embedding")
    argparser.add_argument("--n_tsne", type=int, default=2, help="Number of PCA components")

    np.set_printoptions(precision=2)

    args = argparser.parse_args()
    fpath = f"{os.getcwd()}/{args.config}"

    config, data, fpath = utils.load_config_data_fpath(args)
    config.DConfig.batch_size = 128
    calibrator = utils.load_model_from_config(config, data)
    calibrator.load_model(f"{fpath}", name=args.rnn_name)
    calibrator.eval()

    # Generate all data

    config, fpath, calibrator, true, output, regions, stations, sites, cultivars, all_inds = gen_embeddings(
        config,
        fpath,
        args,
        calibrator,
        name="test",
    )

    # generate tsne data
    tsne, all_emds, unq_inds, cdtype, dfpath, dtype = gen_tsne(
        config,
        fpath,
        args,
        calibrator,
        name="test",
    )

    for i in range(len(all_emds)):
        print(all_emds[i])

    plot(
        config,
        fpath,
        calibrator,
        true,
        output,
        regions,
        stations,
        sites,
        cultivars,
        all_inds,
        tsne,
        all_emds,
        unq_inds,
        cdtype,
        dtype,
    )


if __name__ == "__main__":
    main()
