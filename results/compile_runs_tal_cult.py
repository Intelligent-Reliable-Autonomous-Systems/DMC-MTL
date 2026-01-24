"""
compile_runs.py

Compiles RNN data across multiple runs and takes the average RMSE,
saving data to file

Written by Will Solow, 2025
"""

import os
import numpy as np
import pickle as pkl
import argparse
import torch

import utils
from model_engine.util import CROP_NAMES, REGIONS, STATIONS, SITES

from plotters.plot_utils import compute_total_RMSE, gen_all_data_and_plot, compute_obs_RMSE
from plotters.plotting_functions import compute_rmse_plot, plot_output_coldhardiness
from plotters.plot_utils import gen_batch_data


def cartesian_product(*tensors):
    mesh = torch.meshgrid(*tensors, indexing="ij")
    return torch.stack(mesh, dim=-1).reshape(-1, len(tensors))


def main():

    argparser = argparse.ArgumentParser(description="Plotting script for model")
    argparser.add_argument("--start_dir", type=str, default="_runs/")
    argparser.add_argument("--break_early", action="store_true", help="If to break early when making data")
    argparser.add_argument("--save", action="store_true", help="If to save the plots")
    argparser.add_argument("--config", type=str, default="", help="To be filled in at runtime")
    argparser.add_argument("--prefix", type=str, default=None, help="Prefix of files to search for")
    argparser.add_argument("--num_runs", type=int, default=5)
    argparser.add_argument("--data_fname", type=str, default="rmse.txt")
    argparser.add_argument("--synth_test", type=str, default=None, help="Prefix of synthetic testing data")
    np.set_printoptions(precision=2)
    args = argparser.parse_args()

    config_dirs = utils.find_config_yaml_dirs(args.start_dir)

    # Setup total storage
    all_avg_pheno = np.zeros((12, args.num_runs))
    all_avg_ch = np.zeros((3, args.num_runs))
    all_avg_wf = np.zeros((3, args.num_runs))
    all_cultivar_avg_pheno = np.zeros(
        (len(REGIONS), len(STATIONS), len(SITES), len(CROP_NAMES["grape_phenology_"]), 12, args.num_runs)
    )
    all_cultivar_avg_ch = np.zeros(
        (len(REGIONS), len(STATIONS), len(SITES), len(CROP_NAMES["grape_coldhardiness_ferg"]), 3, args.num_runs)
    )
    all_cultivar_avg_wf = np.zeros((len(CROP_NAMES["wofost_"]), 3, args.num_runs))

    for i, config in enumerate(config_dirs):
        print(config)
        if args.prefix is not None:
            if args.prefix not in config:
                continue

        args.config = f"{args.start_dir}/{config}"
        fpath = f"{os.getcwd()}/{args.config}"

        config, data, fpath = utils.load_config_data_fpath(args)
        args.cultivar = config.DataConfig.cultivar
        config.DConfig.batch_size = 128
        calibrator = utils.load_model_from_config(config, data)

        calibrator.load_model(f"{fpath}", name="rnn_model_best.pt")
        calibrator.eval()

        # Setup per run storage
        true_data = [[], [], []]
        output_data = [[], [], []]
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
        wd_keys = list(config.DataConfig.withold.keys())
        wd = config.DataConfig.withold
        regions = wd["Region"] if "Region" in wd_keys else [None]
        stations = wd["Station"] if "Station" in wd_keys else [None]
        sites = wd["Site"] if "Site" in wd_keys else [None]
        cultivars = wd["Cultivar"] if "Cultivar" in wd_keys else [None]

        n = "test"
        for r in regions:
            for s in stations:

                for si in sites:
                    si = np.argwhere(SITES == si).flatten()
                    si_inds = torch.argwhere(calibrator.sites["train"].flatten() == si[0]) if len(si) != 0 else si
                    si_inds_test = torch.argwhere(calibrator.sites["test"].flatten() == si[0]) if len(si) != 0 else si
                    # Get all the possible cultivars at the testing site
                    s_test_regions = calibrator.regions["train"].flatten()[si_inds]
                    s_test_stations = calibrator.stations["train"].flatten()[si_inds]
                    s_test_sites = calibrator.sites["train"].flatten()[si_inds]
                    s_test_cults = calibrator.cultivars["train"].flatten()[si_inds]
                    s_test = torch.unique(
                        torch.concatenate((s_test_regions, s_test_stations, s_test_sites, s_test_cults), dim=1).to(
                            torch.int64
                        ),
                        dim=0,
                    )
                    for c in cultivars:
                        c = np.argwhere(CROP_NAMES[calibrator.config.DataConfig.dtype] == c).flatten()
                        c_inds = torch.argwhere(calibrator.cultivars["train"].flatten() == c[0]) if len(c) != 0 else c
                        c_inds_test = (
                            torch.argwhere(calibrator.cultivars["test"].flatten() == c[0]) if len(c) != 0 else c
                        )
                        # Get all the sites that have the target cultivar
                        c_test_regions = calibrator.regions["train"].flatten()[c_inds]
                        c_test_stations = calibrator.stations["train"].flatten()[c_inds]
                        c_test_sites = calibrator.sites["train"].flatten()[c_inds]
                        c_test_cults = calibrator.cultivars["train"].flatten()[c_inds]

                        test_regions = calibrator.regions["test"]
                        test_stations = calibrator.stations["test"]
                        test_sites = calibrator.sites["test"]
                        test_cults = calibrator.cultivars["test"]
                        all_info = torch.concatenate((test_regions, test_stations, test_sites, test_cults), dim=-1).to(
                            torch.int64
                        )

                        # Get all the sites and cultivars from which we can evaluate transfer from
                        s_transfer_inds = torch.argwhere(torch.isin(calibrator.sites["train"].flatten(), c_test_sites))
                        st_test_regions = calibrator.regions["train"].flatten()[s_transfer_inds]
                        st_test_stations = calibrator.stations["train"].flatten()[s_transfer_inds]
                        st_test_sites = calibrator.sites["train"].flatten()[s_transfer_inds]
                        st_test_cults = calibrator.cultivars["train"].flatten()[s_transfer_inds]
                        st_test_all = torch.unique(
                            torch.concatenate(
                                (st_test_regions, st_test_stations, st_test_sites, st_test_cults), dim=1
                            ).to(torch.int64),
                            dim=0,
                        )
                        st_test = st_test_all[torch.isin(st_test_all[:, -1], torch.unique(s_test[:, -1]))]
                        # Get all the cultivars at other sites that we can use for transfer
                        c_test = torch.unique(
                            torch.concatenate((c_test_regions, c_test_stations, c_test_sites, c_test_cults), dim=1).to(
                                torch.int64
                            ),
                            dim=0,
                        )
                        test_inds = torch.argwhere(torch.isin(all_info, s_test).all(dim=1))

                        c_repeat = (
                            all_info[test_inds].repeat_interleave(st_test.size(0), dim=1).squeeze().reshape(-1, 4)
                        )
                        a_repeat = test_inds.repeat_interleave(st_test.size(0))
                        b_repeat = st_test.repeat(test_inds.size(0), 1)
                        c_mask = c_repeat[:, -1] == b_repeat[:, -1]

                        a_repeat = a_repeat[c_mask]
                        b_repeat = b_repeat[c_mask]

                        cartesian = torch.concatenate((a_repeat.unsqueeze(1), b_repeat), dim=1)
                        transfer_true, transfer_output, transfer_params = gen_batch_data(
                            calibrator,
                            calibrator.data[n][cartesian[:, 0]],
                            calibrator.dates[n][cartesian[:, 0].cpu().numpy()],
                            calibrator.val[n][cartesian[:, 0]],
                            cartesian[:, 4],
                            cartesian[:, 1],
                            cartesian[:, 2],
                            cartesian[:, 3],
                        )

                        # Compute the average root mean squared error of each cultivar prediction
                        # at the site that we want to transfer to
                        rmse = torch.tensor(
                            np.sqrt(np.nanmean((transfer_true - transfer_output) ** 2, axis=1)), device=cartesian.device
                        )
                        uniq_keys, inv = torch.unique(cartesian[:, -2], dim=0, return_inverse=True)
                        group_sum = torch.zeros(len(uniq_keys), device=cartesian.device)
                        group_count = torch.zeros(len(uniq_keys), device=cartesian.device)
                        group_sum.scatter_add_(0, inv, rmse)
                        group_count.scatter_add_(0, inv, torch.ones_like(rmse))
                        avg_rmse = group_sum / group_count

                        soft_m = torch.nn.Softmax(dim=0)
                        weights = soft_m(avg_rmse).cpu().numpy()
                        c_test = c_test[torch.isin(c_test[:, -2], uniq_keys)]
                        a_inds = si_inds_test[torch.isin(si_inds_test, c_inds_test)]
                        a_repeat = a_inds.repeat_interleave(c_test.size(0))
                        b_repeat = c_test.repeat(a_inds.size(0), 1)
                        eval_cartesian = torch.concatenate((a_repeat.unsqueeze(1), b_repeat), dim=1)

                        true, output, params = gen_batch_data(
                            calibrator,
                            calibrator.data[n][eval_cartesian[:, 0]],
                            calibrator.dates[n][eval_cartesian[:, 0].cpu().numpy()],
                            calibrator.val[n][eval_cartesian[:, 0]],
                            eval_cartesian[:, 4],
                            eval_cartesian[:, 1],
                            eval_cartesian[:, 2],
                            eval_cartesian[:, 3],
                        )

                        true = true[np.arange(0, len(true), c_test.shape[0])]
                        output = np.stack(
                            [
                                np.average(
                                    output[i * c_test.shape[0] : (i + 1) * c_test.shape[0]], axis=0, weights=weights
                                )
                                for i in range(int(output.shape[0] / c_test.shape[0]))
                            ]
                        )

                        inds = plot_output_coldhardiness(
                            config,
                            fpath,
                            np.arange(start=i, stop=i + calibrator.batch_size),
                            output,
                            params,
                            calibrator.val[n][a_inds.to(torch.int32)],
                            name=n,
                            save=args.save,
                        )

                        if len(true.shape) == 1:
                            true = true[np.newaxis, :]
                            output = output[np.newaxis, :]
                        if true.shape[-1] == 3:
                            true = true[:, :, 0]
                            output = output[:, :, 0]
                        [true_data[2].append(true[k][inds[k]]) for k in range(len(true))]
                        [output_data[2].append(output[k][inds[k]]) for k in range(len(output))]

                        cm = calibrator.nn.cult_mapping if hasattr(calibrator.nn, "cult_mapping") else [0, 0]
                        rm = calibrator.nn.reg_mapping if hasattr(calibrator.nn, "reg_mapping") else [0, 0]
                        sm = calibrator.nn.stat_mapping if hasattr(calibrator.nn, "stat_mapping") else [0, 0]
                        sim = calibrator.nn.site_mapping if hasattr(calibrator.nn, "site_mapping") else [0, 0]
                        for k in range(len(true)):
                            ck = (
                                int(calibrator.cultivars[n][a_inds[k].to(torch.int32)].item())
                                if cultivars is not None
                                else 0
                            )
                            rk = (
                                int(calibrator.regions[n][a_inds[k].to(torch.int32)].item())
                                if calibrator.regions is not None
                                else 0
                            )
                            sk = (
                                int(calibrator.stations[n][a_inds[k].to(torch.int32)].item())
                                if calibrator.stations is not None
                                else 0
                            )
                            sik = (
                                int(calibrator.sites[n][a_inds[k].to(torch.int32)].item())
                                if calibrator.sites is not None
                                else 0
                            )

                            true_cultivar_data[rm[rk]][sm[sk]][sim[sik]][cm[ck]][2].append(true[k][inds[k]])
                            output_cultivar_data[rm[rk]][sm[sk]][sim[sik]][cm[ck]][2].append(output[k][inds[k]])

        # Store data for averaging
        if "grape_phenology" in config.DataConfig.dtype:
            train_avg, _ = compute_rmse_plot(config, true_data[0], output_data[0], fpath, save=False)
            test_avg, _ = compute_rmse_plot(config, true_data[2], output_data[2], fpath, name="test", save=False)
            val_avg, _ = compute_rmse_plot(config, true_data[1], output_data[1], fpath, name="val", save=False)
            all_avg_pheno[:, i] = np.concatenate(
                (
                    train_avg[1:-1],
                    [np.sum(train_avg[1:-1])],
                    val_avg[1:-1],
                    [np.sum(val_avg[1:-1])],
                    test_avg[1:-1],
                    [np.sum(test_avg[1:-1])],
                )
            )
        elif "grape_coldhardiness" in config.DataConfig.dtype:
            total_rmse, _ = compute_total_RMSE(true_data[0], output_data[0])
            val_total_rmse, _ = compute_total_RMSE(true_data[1], output_data[1])
            test_total_rmse, _ = compute_total_RMSE(true_data[2], output_data[2])
            all_avg_ch[:, i] = np.array([total_rmse, val_total_rmse, test_total_rmse])
        elif "wofost" in config.DataConfig.dtype:
            total_rmse, _ = compute_total_RMSE(true_data[0], output_data[0])
            val_total_rmse, _ = compute_total_RMSE(true_data[1], output_data[1])
            test_total_rmse, _ = compute_total_RMSE(true_data[2], output_data[2])
            all_avg_wf[:, i] = np.array([total_rmse, val_total_rmse, test_total_rmse])

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
                        if "grape_phenology" in config.DataConfig.dtype:
                            cultivar_train_avg_pheno, _ = compute_rmse_plot(
                                config,
                                true_cultivar_data[r][s][si][k][0],
                                output_cultivar_data[r][s][si][k][0],
                                fpath,
                                save=False,
                            )
                            cultivar_val_avg_pheno, _ = compute_rmse_plot(
                                config,
                                true_cultivar_data[r][s][si][k][1],
                                output_cultivar_data[r][s][si][k][1],
                                fpath,
                                name="val",
                                save=False,
                            )
                            cultivar_test_avg_pheno, _ = compute_rmse_plot(
                                config,
                                true_cultivar_data[r][s][si][k][2],
                                output_cultivar_data[r][s][si][k][2],
                                fpath,
                                name="test",
                                save=False,
                            )
                            all_cultivar_avg_pheno[r, s, si, k, :, i] = np.concatenate(
                                (
                                    cultivar_train_avg_pheno[1:-1],
                                    [np.sum(cultivar_train_avg_pheno[1:-1])],
                                    cultivar_val_avg_pheno[1:-1],
                                    [np.sum(cultivar_val_avg_pheno[1:-1])],
                                    cultivar_test_avg_pheno[1:-1],
                                    [np.sum(cultivar_test_avg_pheno[1:-1])],
                                )
                            )
                        elif "grape_coldhardiness" in config.DataConfig.dtype:
                            cultivar_train_rmse, _ = compute_total_RMSE(
                                true_cultivar_data[r][s][si][k][0], output_cultivar_data[r][s][si][k][0]
                            )
                            cultivar_val_rmse, _ = compute_total_RMSE(
                                true_cultivar_data[r][s][si][k][1], output_cultivar_data[r][s][si][k][1]
                            )
                            cultivar_test_rmse, _ = compute_total_RMSE(
                                true_cultivar_data[r][s][si][k][2], output_cultivar_data[r][s][si][k][2]
                            )
                            all_cultivar_avg_ch[r, s, si, k, :, i] = np.array(
                                [cultivar_train_rmse, cultivar_val_rmse, cultivar_test_rmse]
                            )
                        elif "wofost" in config.DataConfig.dtype:
                            cultivar_train_rmse, _ = compute_total_RMSE(
                                true_cultivar_data[r][s][si][k][0], output_cultivar_data[r][s][si][k][0]
                            )
                            cultivar_val_rmse, _ = compute_total_RMSE(
                                true_cultivar_data[r][s][si][k][1], output_cultivar_data[r][s][si][k][1]
                            )
                            cultivar_test_rmse, _ = compute_total_RMSE(
                                true_cultivar_data[r][s][si][k][2], output_cultivar_data[r][s][si][k][2]
                            )
                            all_cultivar_avg_wf[r, s, si, k, :, i] = np.array(
                                [cultivar_train_rmse, cultivar_val_rmse, cultivar_test_rmse]
                            )
    # Save Data
    prefix = "tal_cult_"
    if "grape_phenology" in config.DataConfig.dtype:
        with open(f"{args.start_dir}/{prefix}results_agg_cultivars.pkl", "wb") as f:
            pkl.dump(all_avg_pheno, f)
        f.close()
        with open(f"{args.start_dir}/{prefix}results_per_cultivars.pkl", "wb") as f:
            pkl.dump(all_cultivar_avg_pheno, f)
        f.close()
    elif "grape_coldhardiness" in config.DataConfig.dtype:
        with open(f"{args.start_dir}/{prefix}results_agg_cultivars.pkl", "wb") as f:
            pkl.dump(all_avg_ch, f)
        f.close()
        with open(f"{args.start_dir}/{prefix}results_per_cultivars.pkl", "wb") as f:
            pkl.dump(all_cultivar_avg_ch, f)
        f.close()
    elif "wofost" in config.DataConfig.dtype:
        with open(f"{args.start_dir}/{prefix}results_agg_cultivars.pkl", "wb") as f:
            pkl.dump(all_avg_wf, f)
        f.close()
        with open(f"{args.start_dir}/{prefix}results_per_cultivars.pkl", "wb") as f:
            pkl.dump(all_cultivar_avg_wf, f)
        f.close()


if __name__ == "__main__":
    main()
