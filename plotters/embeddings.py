"""
Unified embedding / counterfactual plotting CLI.

Modes:
  --method {tsne,pca}   hierarchy embedding scatter (former plot_tsne_embed / plot_pca_embed)
  --mode counterfactual counterfactual trajectory panels (former plot_embeddings)
  --mode hybrid         counterfactual + t-SNE overlay (former plot_embed_tnse)

Written by Will Solow, 2025
"""

from __future__ import annotations

import argparse
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from matplotlib.patches import Patch
from omegaconf import DictConfig
from torch import nn

from assets import CROP_NAMES, REGIONS, SITES, STATIONS
from plotters.embed_utils import (
    build_counterfactual_batch,
    collect_hierarchy_indices,
    extract_embeds,
    load_run_calibrator,
    unique_tol,
)

MARKERS = ["o", "s", "^", "v", "D", "d", "P", "X", "*", "p", "h"]
HYBRID_MARKERS = ["o", "4", "^", "v", "+", "d", "3", "x", "*", "p", "h"]


def plot_reduction(
    reducer,
    embed: np.ndarray,
    idx: np.ndarray,
    dtype: str,
    fpath: str,
    type: Literal["cult", "site", "reg", "stat", "all"],
    title_prefix: str,
):
    """t-SNE or PCA 2D scatter with annotations (PCA) or site-colored markers (t-SNE all)."""
    projected = reducer.fit_transform(embed)
    is_tsne = title_prefix.upper().startswith("T-SNE") or "T-SNE" in title_prefix

    if is_tsne:
        n = 34
        cmap = [plt.colormaps["gist_ncar"](i / n) for i in range(n)]
        plt.figure(figsize=(10, 10))
        for i, _xy in enumerate(projected):
            if type == "all":
                plt.tight_layout()
                c = cmap[idx[i, 2]]
                m = MARKERS[idx[i, 3] % 11]
                plt.scatter(projected[i, 0], projected[i, 1], color=c, marker=m)
                labels = [f"{SITES[j]}" for j in range(len(SITES))]
                handles = [Patch(facecolor=cmap[j], label=label) for j, label in enumerate(labels)]
                plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        plt.scatter(projected[:, 0], projected[:, 1])
        for i, xy in enumerate(projected):
            if type == "cult":
                plt.annotate(
                    CROP_NAMES[dtype][idx[i]],
                    (xy[0], xy[1]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    ha="left",
                )
            elif type == "site":
                plt.annotate(SITES[idx[i]], (xy[0], xy[1]), textcoords="offset points", xytext=(5, 5), ha="left")
            elif type == "stat":
                plt.annotate(STATIONS[idx[i]], (xy[0], xy[1]), textcoords="offset points", xytext=(5, 5), ha="left")
            elif type == "reg":
                plt.annotate(REGIONS[idx[i]], (xy[0], xy[1]), textcoords="offset points", xytext=(5, 5), ha="left")
            elif type == "all":
                plt.annotate(
                    f"{SITES[idx[i, 2]]}/{CROP_NAMES[dtype][idx[i, 3]]}",
                    (xy[0], xy[1]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    ha="left",
                )

    plt.title(f"{title_prefix} for embedding: {type}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig(fpath)
    plt.close()


def plot_counterfactual(
    config: DictConfig,
    fpath: str,
    true: np.ndarray,
    output: np.ndarray,
    regions: np.ndarray,
    stations: np.ndarray,
    sites: np.ndarray,
    cultivars: np.ndarray,
    inds: tuple[int],
) -> None:
    fig, ax = plt.subplots(4, figsize=(18, 24))
    fig.subplots_adjust(top=0.98)
    fig.subplots_adjust(bottom=0.02)
    for k in range(inds[0]):
        cult = CROP_NAMES["grape_coldhardiness_reg"][cultivars[k]]
        stat = STATIONS[stations[k]]
        site = SITES[sites[k]]
        if k == 0:
            ax[0].plot(output[k], label=f"PRED: {stat} {site} {cult}")
            ax[3].plot(output[k], label=f"PRED: {stat} {site} {cult}")
            ax[0].scatter(np.arange(len(true[k])), true[k], label=f"OBS: {stat} {site} {cult}")
            ax[3].scatter(np.arange(len(true[k])), true[k], label=f"OBS: {stat} {site} {cult}")
            orig_site, orig_stat, orig_cult = site, stat, cult
        else:
            if site == orig_site and orig_stat == stat and orig_cult == cult:
                continue
            ax[0].plot(output[k], alpha=0.5, label=f"EVAL: {stat} {site} {cult}")

    for k in range(sum(inds[:3]), sum(inds[:4])):
        cult = CROP_NAMES["grape_coldhardiness_reg"][cultivars[k]]
        stat = STATIONS[stations[k]]
        site = SITES[sites[k]]
        if site == orig_site and orig_stat == stat and orig_cult == cult:
            continue
        ax[3].plot(output[k], alpha=0.5, label=f"EVAL: {stat} {site} {cult}")

    for k in range(inds[0], sum(inds[:2])):
        cult = CROP_NAMES["grape_coldhardiness_reg"][cultivars[k]]
        stat = STATIONS[stations[k]]
        site = SITES[sites[k]]
        if k == inds[0]:
            ax[1].plot(output[k], label=f"PRED: {stat} {site} {cult}")
            ax[2].plot(output[k], label=f"PRED: {stat} {site} {cult}")
            ax[1].scatter(np.arange(len(true[k])), true[k], label=f"OBS: {stat} {site} {cult}")
            ax[2].scatter(np.arange(len(true[k])), true[k], label=f"OBS: {stat} {site} {cult}")
            orig_site, orig_stat, orig_cult = site, stat, cult
        else:
            if site == orig_site and orig_stat == stat and orig_cult == cult:
                continue
            ax[1].plot(output[k], alpha=0.5, label=f"EVAL: {stat} {site} {cult}")

    for k in range(sum(inds[:2]), sum(inds[:3])):
        cult = CROP_NAMES["grape_coldhardiness_reg"][cultivars[k]]
        stat = STATIONS[stations[k]]
        site = SITES[sites[k]]
        if site == orig_site and orig_stat == stat and orig_cult == cult:
            continue
        ax[2].plot(output[k], alpha=0.5, label=f"EVAL: {stat} {site} {cult}")

    for a in ax:
        a.legend()
    plt.tight_layout()
    plt.savefig(
        f"{fpath}/Embed_{cultivars[0][0]}_{sites[0][0]}_{config.DataConfig.cultivar}.png",
        bbox_inches="tight",
    )
    plt.close()


def plot_hybrid(
    config: DictConfig,
    fpath: str,
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
    type: Literal["cult", "site", "reg", "stat", "all"] = "all",
) -> None:
    fig, ax = plt.subplots(2, figsize=(6, 5))
    for k in range(inds[0]):
        cult = CROP_NAMES["grape_coldhardiness_reg"][cultivars[k]]
        stat = STATIONS[stations[k]]
        site = SITES[sites[k]]
        if k == 0:
            ax[0].plot(output[k])
            orig_stat, orig_cult, orig_site = stat, cult, site
        else:
            if site == orig_site and orig_stat == stat and orig_cult == cult:
                continue
            ax[0].plot(output[k], alpha=0.5)
        tsne_embed = tsne.fit_transform(embed)

    n = 34
    cmap = [plt.colormaps["gist_ncar"](i / n) for i in range(n)]
    for i, _xy in enumerate(tsne_embed):
        if type == "all":
            c = cmap[idx[i, 2]]
            m = HYBRID_MARKERS[idx[i, 3] % 11]
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
    ax[0].set_xlabel("Days since Dormancy Season Start (September 7th)")
    ax[0].set_ylabel(r"LTE50 $^\circ$C")
    ax[1].set_xlabel("t-SNE Dimension 1")
    ax[1].set_ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig(
        f"{fpath}/Embed_{cultivars[0][0]}_{sites[0][0]}_{config.DataConfig.cultivar}.png",
        bbox_inches="tight",
    )
    plt.close()


def run_reduction_plots(config: DictConfig, args, calibrator: nn.Module, method: str) -> None:
    all_inds, unq_reg, unq_stat, unq_site, unq_cult = collect_hierarchy_indices(calibrator)
    e_mode = getattr(args, "e_mode", "direct")
    if method == "pca":
        e_mode = getattr(args, "e_mode", None) or "forward"
    reg_embed, stat_embed, site_embed, cult_embed = extract_embeds(calibrator, all_inds, e_mode=e_mode)

    if method == "tsne":
        reducer = sklearn.manifold.TSNE(n_components=args.n_components, perplexity=5.0)
        title = "T-SNE"
    else:
        reducer = sklearn.decomposition.PCA(n_components=args.n_components)
        title = "PCA"

    all_emds = torch.cat((reg_embed, stat_embed, site_embed, cult_embed), dim=1).cpu().numpy()
    unq_inds = all_inds.cpu().numpy().astype(np.int32)
    unq_reg_embed = unique_tol(reg_embed).cpu().numpy()
    unq_stat_embed = unique_tol(stat_embed).cpu().numpy()
    unq_site_embed = unique_tol(site_embed).cpu().numpy()
    unq_cult_embed = unique_tol(cult_embed).cpu().numpy()
    base = args.config

    plot_reduction(reducer, unq_cult_embed, unq_cult, config.DataConfig.dtype, f"{base}/cult_embed.png", "cult", title)
    plot_reduction(reducer, unq_site_embed, unq_site, config.DataConfig.dtype, f"{base}/site_embed.png", "site", title)
    plot_reduction(reducer, unq_stat_embed, unq_stat, config.DataConfig.dtype, f"{base}/stat_embed.png", "stat", title)
    if unq_reg_embed.shape[0] > 1:
        plot_reduction(reducer, unq_reg_embed, unq_reg, config.DataConfig.dtype, f"{base}/reg_embed.png", "reg", title)
    plot_reduction(reducer, all_emds, unq_inds, config.DataConfig.dtype, f"{base}/all_embed.png", "all", title)


def run_counterfactual(config: DictConfig, fpath: str, args, calibrator: nn.Module, name: str = "test") -> None:
    for i in range(0, len(calibrator.data[name])):
        true, output, regions, stations, sites, cultivars, inds = build_counterfactual_batch(calibrator, name, i)
        if "grape_coldhardiness" in config.DataConfig.dtype:
            plot_counterfactual(config, fpath, true, output, regions, stations, sites, cultivars, inds)
        if args.break_early:
            break


def run_hybrid(config: DictConfig, fpath: str, args, calibrator: nn.Module, name: str = "test") -> None:
    # First counterfactual sample (matches former script: gen_embeddings returns after first i)
    true, output, regions, stations, sites, cultivars, inds = build_counterfactual_batch(calibrator, name, 0)

    all_inds, *_ = collect_hierarchy_indices(calibrator)
    reg_embed, stat_embed, site_embed, cult_embed = extract_embeds(
        calibrator, all_inds, e_mode=getattr(args, "e_mode", "direct")
    )
    tsne = sklearn.manifold.TSNE(n_components=args.n_components, perplexity=5.0)
    all_emds = torch.cat((reg_embed, stat_embed, site_embed, cult_embed), dim=1).cpu().numpy()
    unq_inds = all_inds.cpu().numpy().astype(np.int32)

    plot_hybrid(
        config,
        fpath,
        true,
        output,
        regions,
        stations,
        sites,
        cultivars,
        inds,
        tsne,
        all_emds,
        unq_inds,
        type="all",
    )


def main(argv: list[str] | None = None, *, defaults: dict | None = None) -> None:
    defaults = defaults or {}
    parser = argparse.ArgumentParser(description="Unified embedding / counterfactual plotting")
    parser.add_argument("--config", type=str, required=True, help="Path to the run/config directory")
    # Legacy scripts used store_false for --break_early (default True = stop after first sample).
    parser.add_argument(
        "--break_early",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("break_early", True),
        help="Stop after first sample (counterfactual); default on for legacy parity",
    )
    parser.add_argument("--rnn_name", type=str, default="rnn_model_best.pt")
    parser.add_argument(
        "--method",
        type=str,
        default=defaults.get("method"),
        choices=["tsne", "pca"],
        help="Dimensionality reduction for hierarchy embeds",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=defaults.get("mode", "default"),
        choices=["default", "counterfactual", "hybrid"],
        help="default=reduction plots; counterfactual / hybrid as named",
    )
    parser.add_argument(
        "--e_mode",
        type=str,
        default=defaults.get("e_mode", "direct"),
        help="Embedding forward mode (direct/text/adapt/forward)",
    )
    parser.add_argument("--n_components", type=int, default=defaults.get("n_components", 2))
    # Back-compat aliases
    parser.add_argument("--n_tsne", type=int, default=None)
    parser.add_argument("--n_pca", type=int, default=None)

    args = parser.parse_args(argv)
    if args.n_tsne is not None:
        args.n_components = args.n_tsne
    if args.n_pca is not None:
        args.n_components = args.n_pca
    if args.method is None and args.mode == "default":
        args.method = defaults.get("method") or "tsne"

    np.set_printoptions(precision=2)
    config, fpath, calibrator = load_run_calibrator(args)

    if args.mode == "counterfactual":
        run_counterfactual(config, fpath, args, calibrator, name="test")
    elif args.mode == "hybrid":
        run_hybrid(config, fpath, args, calibrator, name="test")
    else:
        method = args.method or "tsne"
        if method == "pca" and defaults.get("e_mode") is None and args.e_mode == "direct":
            # Former PCA script used plain forward_embed
            args.e_mode = "forward"
        run_reduction_plots(config, args, calibrator, method=method)


if __name__ == "__main__":
    main()
