"""
integrated_gradients.py

Computes the integrated gradients of an input for a given model

Written by Will Solow, 2025
"""

import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import utils


def integrated_gradients(
    model: nn.Module,
    input: torch.Tensor,
    dates: np.ndarray,
    cultivars: torch.Tensor,
    regions: torch.Tensor,
    stations: torch.Tensor,
    sites: torch.Tensor,
    baseline: torch.Tensor = None,
    target_idx: int = 0,
    steps: int = 50,
    t: int = 0,
) -> torch.Tensor:
    """
    Computes Integrated Gradients for a specific output index in a multi-output regression model.
    """
    if isinstance(target_idx, int):
        target_idx = [target_idx]

    # Initialize
    input = input.requires_grad_()
    if baseline is None:
        baseline = 0 * input
    baseline = baseline.to(input.device)
    batch_size, dlen, n_features = input.shape

    # Scale/tile inputs to required shape
    scaled_inputs = torch.cat(
        [baseline + (float(i) / steps) * (input - baseline) for i in range(steps + 1)],
        dim=0,
    ).to(input.device)

    scaled_dates = np.repeat(dates, repeats=(steps + 1), axis=0)
    scaled_cultivars = torch.tile(cultivars, (steps + 1, 1)).to(input.device)
    scaled_regions = torch.tile(regions, (steps + 1, 1)).to(input.device)
    scaled_stations = torch.tile(stations, (steps + 1, 1)).to(input.device)
    scaled_sites = torch.tile(sites, (steps + 1, 1)).to(input.device)

    scaled_inputs.requires_grad_()
    outputs, params, _ = model.forward(
        scaled_inputs,
        scaled_dates,
        cultivars=scaled_cultivars,
        regions=scaled_regions,
        stations=scaled_stations,
        sites=scaled_sites,
    )
    attributions = torch.empty(batch_size, len(target_idx), dlen, n_features).to(input.device)

    for i, idx in enumerate(target_idx):
        grads = torch.autograd.grad(params[:, t, idx].sum(), scaled_inputs, create_graph=False)[0]
        grads = grads.view(steps + 1, batch_size, dlen, n_features)
        avg_grads = grads[:-1].mean(dim=0)

        ig = (input - baseline) * avg_grads

        attributions[:, i, :] = ig

    return attributions.detach().cpu().numpy()


def main():
    argparser = argparse.ArgumentParser(description="Plotting script for model")
    argparser.add_argument("--config", type=str, required=True, help="Path to the config file")
    argparser.add_argument("--rnn_name", type=str, default="rnn_model_best.pt")
    argparser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of Steps for integrated gradients",
    )
    argparser.add_argument("--param", type=str, default="TBASEM")
    argparser.add_argument(
        "--baseline",
        type=str,
        default="average",
        choices=["average", "zeros"],
        help="Baseline to use",
    )
    argparser.add_argument("--sample", type=int, default=0, help="year to sample")
    np.set_printoptions(precision=2)

    args = argparser.parse_args()
    fpath = f"{os.getcwd()}/{args.config}"

    config, data, fpath = utils.load_config_data_fpath(args)
    config.DConfig.batch_size = args.num_steps + 1  # Update the batch size to support passing all steps
    calibrator = utils.load_model_from_config(config, data)
    calibrator.load_model(f"{fpath}", name=args.rnn_name)
    os.makedirs(f"{fpath}/ig/", exist_ok=True)

    # Get parameters and adjust inputs
    params = config.params
    params_d = dict(zip(params, np.arange(len(params)).tolist()))
    input_vars = config.PConfig.input_vars
    input_vars = ["DAY0"] + input_vars
    input_vars[1] = "DAY1"
    param = args.param

    # Sample and time
    sample = args.sample
    t = args.num_steps
    dset = "test"

    # Set baseline for simualtion
    if args.baseline == "average":
        baseline = torch.mean(calibrator.data[dset], dim=0)
    elif args.baseline == "zeros":
        baseline = None

    if sample == -1:
        # Integrated Gradients
        attrs = np.empty(
            shape=(
                len(calibrator.data[dset]),
                calibrator.data[dset][0].shape[0],
                calibrator.data[dset][0].shape[1],
            )
        )
        for i in range(len(calibrator.data[dset])):
            # Data gathering
            data = calibrator.data[dset][i].unsqueeze(0)
            dates = calibrator.dates[dset][i][np.newaxis, :]
            cultivars = calibrator.cultivars[dset][i].unsqueeze(0)
            regions = calibrator.regions[dset][i].unsqueeze(0)
            stations = calibrator.stations[dset][i].unsqueeze(0)
            sites = calibrator.sites[dset][i].unsqueeze(0)
            attrs[i, :] = integrated_gradients(
                calibrator,
                data,
                dates,
                cultivars,
                regions,
                stations,
                sites,
                baseline=baseline,
                target_idx=params_d[param],
                t=t,
            ).squeeze()
        attrs = np.mean(attrs, axis=0)
        sample = "all"
    else:
        # Data gathering
        data = calibrator.data[dset][sample].unsqueeze(0)
        dates = calibrator.dates[dset][sample][np.newaxis, :]
        cultivars = calibrator.cultivars[dset][sample].unsqueeze(0)
        regions = calibrator.regions[dset][sample].unsqueeze(0)
        stations = calibrator.stations[dset][sample].unsqueeze(0)
        sites = calibrator.sites[dset][sample].unsqueeze(0)
        attrs = integrated_gradients(
            calibrator,
            data,
            dates,
            cultivars,
            regions,
            stations,
            sites,
            baseline=baseline,
            target_idx=params_d[param],
            t=t,
        ).squeeze()

    # Plotting
    fig, ax = plt.subplots(1, figsize=(10, 6))
    im = ax.imshow(
        attrs[:t],
        aspect="auto",
        cmap="RdBu",
        vmin=-np.max(np.abs(attrs)),
        vmax=np.max(np.abs(attrs)),
    )
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(input_vars)), labels=input_vars, rotation=45)
    ax.set_title(f"All Days Integrated Gradients: {param}")
    ax.set_ylabel("Days")
    plt.savefig(
        f"{fpath}/ig/{param}_d{t}_s{sample}_{args.baseline}_all.png",
        bbox_inches="tight",
    )
    plt.close()

    fig, ax = plt.subplots(1, figsize=(12, 1))
    im = ax.imshow(
        attrs[t][np.newaxis, :],
        aspect="auto",
        cmap="RdBu",
        vmin=-np.max(np.abs(attrs[t])),
        vmax=np.max(np.abs(attrs[t])),
    )
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(input_vars)), labels=input_vars, rotation=45)
    ax.set_title(f"Current Day Integrated Gradients: {param}")
    ax.set_ylabel(f"Day: {t}")
    ax.set_yticks(ticks=[], labels=[])

    plt.savefig(
        f"{fpath}/ig/{param}_d{t}_s{sample}_{args.baseline}_current.png",
        bbox_inches="tight",
    )
    plt.close()

    fig, ax = plt.subplots(1, figsize=(12, 1))
    im = ax.imshow(
        np.sum(np.abs(attrs[:t]), axis=0)[np.newaxis, :],
        aspect="auto",
        cmap="RdBu",
        vmin=-np.max(np.abs(np.sum(attrs[:t], axis=0))),
        vmax=np.max(np.abs(np.sum(attrs[:t], axis=0))),
    )
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(input_vars)), labels=input_vars, rotation=45)
    ax.set_title(f"Abs Sum Integrated Gradients: {param}")
    ax.set_ylabel(f"Day: {t}")
    ax.set_yticks(ticks=[], labels=[])

    plt.savefig(
        f"{fpath}/ig/{param}_d{t}_s{sample}_{args.baseline}_sum.png",
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    main()
