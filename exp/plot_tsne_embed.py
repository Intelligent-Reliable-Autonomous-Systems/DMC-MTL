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
import torch.nn as nn
from matplotlib.patches import Patch
from omegaconf import DictConfig

import utils
from assets import CROP_NAMES, REGIONS, SITES, STATIONS

markers = ["o", "s", "^", "v", "D", "d", "P", "X", "*", "p", "h"]


def plot(
    tsne: sklearn.manifold.TSNE,
    embed: np.ndarray,
    idx: np.ndarray,
    dtype: str,
    fpath: str,
    type: Literal["cult", "site", "reg", "stat"],
):
    tsne_embed = tsne.fit_transform(embed)

    n = 30
    cmap = plt.colormaps["gist_ncar"]  # or 'turbo', 'rainbow', 'nipy_spectral'
    cmap = [cmap(i / n) for i in range(n)]
    plt.figure(figsize=(10, 10))
    for i, xy in enumerate(tsne_embed):
        if type == "cult":
            plt.annotate(
                CROP_NAMES[dtype][idx[i]], (xy[0], xy[1]), textcoords="offset points", xytext=(5, 5), ha="left"
            )
            plt.scatter(tsne_embed[i, 0], tsne_embed[i, 1])
        elif type == "site":
            plt.annotate(SITES[idx[i]], (xy[0], xy[1]), textcoords="offset points", xytext=(5, 5), ha="left")
            plt.scatter(tsne_embed[i, 0], tsne_embed[i, 1])
        elif type == "stat":
            plt.annotate(STATIONS[idx[i]], (xy[0], xy[1]), textcoords="offset points", xytext=(5, 5), ha="left")
            plt.scatter(tsne_embed[i, 0], tsne_embed[i, 1])
        elif type == "reg":
            plt.annotate(REGIONS[idx[i]], (xy[0], xy[1]), textcoords="offset points", xytext=(5, 5), ha="left")
            plt.scatter(tsne_embed[i, 0], tsne_embed[i, 1])

        elif type == "all":
            plt.tight_layout()
            # plt.annotate(
            #     f"{SITES[idx[i, 2]]}/{CROP_NAMES[dtype][idx[i, 3]]}",
            #     (xy[0], xy[1]),
            #     textcoords="offset points",
            #     xytext=(5, 5),
            #     ha="left",
            # )
            c = cmap[idx[i, 2]]
            m = markers[idx[i, 3] % 11]
            plt.scatter(tsne_embed[i, 0], tsne_embed[i, 1], color=c, marker=m)

            labels = [f"{SITES[i]}" for i in range(len(SITES))]

            handles = [Patch(facecolor=cmap[i], label=label) for i, label in enumerate(labels)]

            plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(f"T-SNE for embedding: {type}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig(fpath)
    plt.close()


def unique_tol(x: torch.Tensor, tol: float = 1e5, dim: int = 0):
    q = torch.round(x * tol) / tol
    return torch.unique(q, dim=dim)


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
        reg_embed, stat_embed, site_embed, cult_embed = calibrator.nn.forward_embed(
            regions=all_inds[:, 0:1],
            stations=all_inds[:, 1:2],
            sites=all_inds[:, 2:3],
            cultivars=all_inds[:, 3:],
            text_embedder=text_emb,
        )

    tsne = sklearn.manifold.TSNE(n_components=args.n_tsne, perplexity=5.0)
    all_emds = torch.cat((reg_embed, stat_embed, site_embed, cult_embed), dim=1).cpu().numpy()
    unq_inds = all_inds.cpu().numpy().astype(np.int32)
    unq_reg_embed = unique_tol(reg_embed).cpu().numpy()
    unq_stat_embed = unique_tol(stat_embed).cpu().numpy()
    unq_site_embed = unique_tol(site_embed).cpu().numpy()
    unq_cult_embed = unique_tol(cult_embed).cpu().numpy()

    plot(tsne, unq_cult_embed, unq_cult, config.DataConfig.dtype, f"{args.config}/cult_embed.png", type="cult")
    plot(tsne, unq_site_embed, unq_site, config.DataConfig.dtype, f"{args.config}/site_embed.png", type="site")
    plot(tsne, unq_stat_embed, unq_stat, config.DataConfig.dtype, f"{args.config}/stat_embed.png", type="stat")
    if unq_reg_embed.shape[0] > 1:
        plot(tsne, unq_reg_embed, unq_reg, config.DataConfig.dtype, f"{args.config}/reg_embed.png", type="reg")
    plot(tsne, all_emds, unq_inds, config.DataConfig.dtype, f"{args.config}/all_embed.png", type="all")


def main():

    argparser = argparse.ArgumentParser(description="Plotting script for model")
    argparser.add_argument("--config", type=str, required=True, help="Path to the config file")
    argparser.add_argument("--break_early", action="store_false", help="If to break early when making data")
    argparser.add_argument("--save", action="store_false", help="If to save the plots")
    argparser.add_argument("--fname", type=str, default="data.txt", help="File to save data to")
    argparser.add_argument("--rnn_name", type=str, default="rnn_model_best.pt")
    argparser.add_argument("--n_tsne", type=int, default=2, help="Number of PCA components")
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
