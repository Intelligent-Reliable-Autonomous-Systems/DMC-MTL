"""
Transfer evaluation strategies.

Each predict_* fills true_data[2] / output_data[2] and cultivar trees for the test split.

Written by Will Solow, 2025
"""

import numpy as np
import torch

STRATEGY_PREFIX = {
    "mean_all": "all_mean_",
    "mean_cult": "cult_mean_",
    "mean_site": "site_mean_",
    "best_source": "best_source_",
    "tal_site": "tal_site_",
    "tal_cult": "tal_cult_",
    "cossim_true_site": "cossim_true_site_",
    "cossim_true_cult": "cossim_true_cult_",
}


def predict_mean_all(config, fpath, args, calibrator, true_data, output_data, true_cultivar_data, output_cultivar_data):
    """Transfer predictions for strategy `mean_all` (prefix all_mean_)."""
    from assets import CROP_NAMES, SITES
    from plotters.plot_utils import gen_batch_data
    from plotters.plotting_functions import plot_output_coldhardiness
    from results.eval_common import parse_withold

    regions, stations, sites, cultivars = parse_withold(config)
    n = "test"
    for r in regions:
        r_inds = torch.arange(calibrator.sites["train"].flatten().shape[0]).unsqueeze(1).reshape(-1, 1)
        r_inds_test = torch.arange(calibrator.sites["test"].flatten().shape[0]).unsqueeze(1).reshape(-1, 1)
        # Get all the possible cultivars at the testing site
        r_test_regions = calibrator.regions["train"].flatten()[r_inds]
        r_test_stations = calibrator.stations["train"].flatten()[r_inds]
        r_test_sites = calibrator.sites["train"].flatten()[r_inds]
        r_test_cults = calibrator.cultivars["train"].flatten()[r_inds]
        b_inds = torch.unique(
            torch.concatenate((r_test_regions, r_test_stations, r_test_sites, r_test_cults), dim=1).to(torch.int64),
            dim=0,
        )
        for s in stations:
            for si in sites:
                si = np.argwhere(SITES == si).flatten()
                si_inds_test = torch.argwhere(calibrator.sites["test"].flatten() == si[0]) if len(si) != 0 else si
                for c in cultivars:
                    c = np.argwhere(CROP_NAMES[calibrator.config.DataConfig.dtype] == c).flatten()
                    c_inds_test = torch.argwhere(calibrator.cultivars["test"].flatten() == c[0]) if len(c) != 0 else c

                    a_inds = si_inds_test[torch.isin(si_inds_test, c_inds_test)]

                    a_repeat = a_inds.repeat_interleave(b_inds.size(0))

                    b_repeat = b_inds.repeat(a_inds.size(0), 1)

                    cartesian = torch.concatenate((a_repeat.unsqueeze(1), b_repeat), dim=1)

                    true, output, params = gen_batch_data(
                        calibrator,
                        calibrator.data[n][cartesian[:, 0]],
                        calibrator.dates[n][cartesian[:, 0].cpu().numpy()],
                        calibrator.val[n][cartesian[:, 0]],
                        cartesian[:, 4],
                        cartesian[:, 1],
                        cartesian[:, 2],
                        cartesian[:, 3],
                    )
                    true = true[np.arange(0, len(true), b_inds.shape[0])]
                    output = np.stack(
                        [
                            np.mean(output[i * b_inds.shape[0] : (i + 1) * b_inds.shape[0]], axis=0)
                            for i in range(int(output.shape[0] / b_inds.shape[0]))
                        ]
                    )
                    inds = plot_output_coldhardiness(
                        config,
                        fpath,
                        np.arange(start=0, stop=calibrator.batch_size),
                        output,
                        params,
                        calibrator.val[n][a_inds.to(torch.int32)],
                        name=n,
                        save=args.save,
                    )
                    from results.eval_common import append_masked_predictions

                    append_masked_predictions(
                        true_data,
                        output_data,
                        true_cultivar_data,
                        output_cultivar_data,
                        calibrator,
                        true,
                        output,
                        inds,
                        a_inds,
                        split_idx=2,
                        name=n,
                    )


def predict_mean_cult(
    config, fpath, args, calibrator, true_data, output_data, true_cultivar_data, output_cultivar_data
):
    """Transfer predictions for strategy `mean_cult` (prefix cult_mean_)."""
    from assets import CROP_NAMES, SITES
    from plotters.plot_utils import gen_batch_data
    from plotters.plotting_functions import plot_output_coldhardiness
    from results.eval_common import parse_withold

    regions, stations, sites, cultivars = parse_withold(config)
    n = "test"
    for r in regions:
        for s in stations:
            for si in sites:
                si = np.argwhere(SITES == si).flatten()
                si_inds_test = torch.argwhere(calibrator.sites["test"].flatten() == si[0]) if len(si) != 0 else si
                for c in cultivars:
                    c = np.argwhere(CROP_NAMES[calibrator.config.DataConfig.dtype] == c).flatten()
                    c_inds = torch.argwhere(calibrator.cultivars["train"].flatten() == c[0]) if len(c) != 0 else c
                    c_inds_test = torch.argwhere(calibrator.cultivars["test"].flatten() == c[0]) if len(c) != 0 else c

                    test_regions = calibrator.regions["train"].flatten()[c_inds]
                    test_stations = calibrator.stations["train"].flatten()[c_inds]
                    test_sites = calibrator.sites["train"].flatten()[c_inds]
                    test_cults = calibrator.cultivars["train"].flatten()[c_inds]
                    a_inds = si_inds_test[torch.isin(si_inds_test, c_inds_test)]
                    b_inds = torch.unique(
                        torch.concatenate((test_regions, test_stations, test_sites, test_cults), dim=1).to(torch.int64),
                        dim=0,
                    )

                    a_repeat = a_inds.repeat_interleave(b_inds.size(0))

                    b_repeat = b_inds.repeat(a_inds.size(0), 1)

                    cartesian = torch.concatenate((a_repeat.unsqueeze(1), b_repeat), dim=1)

                    true, output, params = gen_batch_data(
                        calibrator,
                        calibrator.data[n][cartesian[:, 0]],
                        calibrator.dates[n][cartesian[:, 0].cpu().numpy()],
                        calibrator.val[n][cartesian[:, 0]],
                        cartesian[:, 4],
                        cartesian[:, 1],
                        cartesian[:, 2],
                        cartesian[:, 3],
                    )
                    true = true[np.arange(0, len(true), b_inds.shape[0])]
                    output = np.stack(
                        [
                            np.mean(output[i * b_inds.shape[0] : (i + 1) * b_inds.shape[0]], axis=0)
                            for i in range(int(output.shape[0] / b_inds.shape[0]))
                        ]
                    )
                    inds = plot_output_coldhardiness(
                        config,
                        fpath,
                        np.arange(start=0, stop=calibrator.batch_size),
                        output,
                        params,
                        calibrator.val[n][a_inds.to(torch.int32)],
                        name=n,
                        save=args.save,
                    )
                    from results.eval_common import append_masked_predictions

                    append_masked_predictions(
                        true_data,
                        output_data,
                        true_cultivar_data,
                        output_cultivar_data,
                        calibrator,
                        true,
                        output,
                        inds,
                        a_inds,
                        split_idx=2,
                        name=n,
                    )


def predict_mean_site(
    config, fpath, args, calibrator, true_data, output_data, true_cultivar_data, output_cultivar_data
):
    """Transfer predictions for strategy `mean_site` (prefix site_mean_)."""
    from assets import CROP_NAMES, SITES
    from plotters.plot_utils import gen_batch_data
    from plotters.plotting_functions import plot_output_coldhardiness
    from results.eval_common import parse_withold

    regions, stations, sites, cultivars = parse_withold(config)
    n = "test"
    for r in regions:
        for s in stations:
            for si in sites:
                si = np.argwhere(SITES == si).flatten()
                si_inds = torch.argwhere(calibrator.sites["train"].flatten() == si[0]) if len(si) != 0 else si
                si_inds_test = torch.argwhere(calibrator.sites["test"].flatten() == si[0]) if len(si) != 0 else si

                test_regions = calibrator.regions["train"].flatten()[si_inds]
                test_stations = calibrator.stations["train"].flatten()[si_inds]
                test_sites = calibrator.sites["train"].flatten()[si_inds]
                test_cults = calibrator.cultivars["train"].flatten()[si_inds]

                for c in cultivars:
                    c = np.argwhere(CROP_NAMES[calibrator.config.DataConfig.dtype] == c).flatten()
                    c_inds_test = torch.argwhere(calibrator.cultivars["test"].flatten() == c[0]) if len(c) != 0 else c
                    a_inds = si_inds_test[torch.isin(si_inds_test, c_inds_test)]

                    b_inds = torch.unique(
                        torch.concatenate((test_regions, test_stations, test_sites, test_cults), dim=1).to(torch.int64),
                        dim=0,
                    )

                    a_repeat = a_inds.repeat_interleave(b_inds.size(0))

                    b_repeat = b_inds.repeat(a_inds.size(0), 1)

                    cartesian = torch.concatenate((a_repeat.unsqueeze(1), b_repeat), dim=1)
                    true, output, params = gen_batch_data(
                        calibrator,
                        calibrator.data[n][cartesian[:, 0]],
                        calibrator.dates[n][cartesian[:, 0].cpu().numpy()],
                        calibrator.val[n][cartesian[:, 0]],
                        cartesian[:, 4],
                        cartesian[:, 1],
                        cartesian[:, 2],
                        cartesian[:, 3],
                    )
                    true = true[np.arange(0, len(true), b_inds.shape[0])]

                    output = np.stack(
                        [
                            np.mean(output[i * b_inds.shape[0] : (i + 1) * b_inds.shape[0]], axis=0)
                            for i in range(int(output.shape[0] / b_inds.shape[0]))
                        ]
                    )
                    inds = plot_output_coldhardiness(
                        config,
                        fpath,
                        np.arange(start=0, stop=calibrator.batch_size),
                        output,
                        params,
                        calibrator.val[n][a_inds.to(torch.int32)],
                        name=n,
                        save=args.save,
                    )

                    from results.eval_common import append_masked_predictions

                    append_masked_predictions(
                        true_data,
                        output_data,
                        true_cultivar_data,
                        output_cultivar_data,
                        calibrator,
                        true,
                        output,
                        inds,
                        a_inds,
                        split_idx=2,
                        name=n,
                    )


def predict_best_source(
    config, fpath, args, calibrator, true_data, output_data, true_cultivar_data, output_cultivar_data
):
    """Transfer predictions for strategy `best_source` (prefix best_source_)."""
    from assets import CROP_NAMES, SITES
    from plotters.plot_utils import gen_batch_data
    from plotters.plotting_functions import plot_output_coldhardiness
    from results.eval_common import parse_withold

    regions, stations, sites, cultivars = parse_withold(config)
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
                    c_inds_test = torch.argwhere(calibrator.cultivars["test"].flatten() == c[0]) if len(c) != 0 else c
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
                        torch.concatenate((st_test_regions, st_test_stations, st_test_sites, st_test_cults), dim=1).to(
                            torch.int64
                        ),
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

                    c_repeat = all_info[test_inds].repeat_interleave(st_test.size(0), dim=1).squeeze().reshape(-1, 4)
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
                    uniq_keys, inv = torch.unique(cartesian[:, 1:5], dim=0, return_inverse=True)
                    group_sum = torch.zeros(len(uniq_keys), device=cartesian.device)
                    group_count = torch.zeros(len(uniq_keys), device=cartesian.device)
                    group_sum.scatter_add_(0, inv, rmse)
                    group_count.scatter_add_(0, inv, torch.ones_like(rmse))
                    avg_rmse = group_sum / group_count
                    best_transfer = uniq_keys[torch.argmin(avg_rmse)]
                    transfer_ind = torch.argwhere((best_transfer[1:3] == c_test[:, 1:3]).all(dim=1)).flatten()

                    c_transfer = c_test[transfer_ind]
                    a_inds = si_inds_test[torch.isin(si_inds_test, c_inds_test)]
                    a_repeat = a_inds.repeat_interleave(c_transfer.size(0))
                    b_repeat = c_transfer.repeat(a_inds.size(0), 1)

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

                    inds = plot_output_coldhardiness(
                        config,
                        fpath,
                        np.arange(start=0, stop=calibrator.batch_size),
                        output,
                        params,
                        calibrator.val[n][a_inds.to(torch.int32)],
                        name=n,
                        save=args.save,
                    )

                    from results.eval_common import append_masked_predictions

                    append_masked_predictions(
                        true_data,
                        output_data,
                        true_cultivar_data,
                        output_cultivar_data,
                        calibrator,
                        true,
                        output,
                        inds,
                        a_inds,
                        split_idx=2,
                        name=n,
                    )


def predict_tal_site(config, fpath, args, calibrator, true_data, output_data, true_cultivar_data, output_cultivar_data):
    """Transfer predictions for strategy `tal_site` (prefix tal_site_)."""
    from assets import CROP_NAMES, SITES
    from plotters.plot_utils import gen_batch_data
    from plotters.plotting_functions import plot_output_coldhardiness
    from results.eval_common import parse_withold

    regions, stations, sites, cultivars = parse_withold(config)
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
                    c_inds_test = torch.argwhere(calibrator.cultivars["test"].flatten() == c[0]) if len(c) != 0 else c
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
                        torch.concatenate((st_test_regions, st_test_stations, st_test_sites, st_test_cults), dim=1).to(
                            torch.int64
                        ),
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

                    c_repeat = all_info[test_inds].repeat_interleave(st_test.size(0), dim=1).squeeze().reshape(-1, 4)
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
                    uniq_keys, inv = torch.unique(cartesian[:, -1], dim=0, return_inverse=True)
                    group_sum = torch.zeros(len(uniq_keys), device=cartesian.device)
                    group_count = torch.zeros(len(uniq_keys), device=cartesian.device)
                    group_sum.scatter_add_(0, inv, rmse)
                    group_count.scatter_add_(0, inv, torch.ones_like(rmse))
                    avg_rmse = group_sum / group_count

                    soft_m = torch.nn.Softmax(dim=0)
                    weights = soft_m(avg_rmse).cpu().numpy()
                    a_inds = si_inds_test[torch.isin(si_inds_test, c_inds_test)]
                    s_test = s_test[torch.isin(s_test[:, -1], uniq_keys)]
                    a_repeat = a_inds.repeat_interleave(s_test.size(0))
                    b_repeat = s_test.repeat(a_inds.size(0), 1)
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

                    true = true[np.arange(0, len(true), s_test.shape[0])]
                    output = np.stack(
                        [
                            np.average(output[i * s_test.shape[0] : (i + 1) * s_test.shape[0]], axis=0, weights=weights)
                            for i in range(int(output.shape[0] / s_test.shape[0]))
                        ]
                    )

                    inds = plot_output_coldhardiness(
                        config,
                        fpath,
                        np.arange(start=0, stop=calibrator.batch_size),
                        output,
                        params,
                        calibrator.val[n][a_inds.to(torch.int32)],
                        name=n,
                        save=args.save,
                    )

                    from results.eval_common import append_masked_predictions

                    append_masked_predictions(
                        true_data,
                        output_data,
                        true_cultivar_data,
                        output_cultivar_data,
                        calibrator,
                        true,
                        output,
                        inds,
                        a_inds,
                        split_idx=2,
                        name=n,
                    )


def predict_tal_cult(config, fpath, args, calibrator, true_data, output_data, true_cultivar_data, output_cultivar_data):
    """Transfer predictions for strategy `tal_cult` (prefix tal_cult_)."""
    from assets import CROP_NAMES, SITES
    from plotters.plot_utils import gen_batch_data
    from plotters.plotting_functions import plot_output_coldhardiness
    from results.eval_common import parse_withold

    regions, stations, sites, cultivars = parse_withold(config)
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
                    c_inds_test = torch.argwhere(calibrator.cultivars["test"].flatten() == c[0]) if len(c) != 0 else c
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
                        torch.concatenate((st_test_regions, st_test_stations, st_test_sites, st_test_cults), dim=1).to(
                            torch.int64
                        ),
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

                    c_repeat = all_info[test_inds].repeat_interleave(st_test.size(0), dim=1).squeeze().reshape(-1, 4)
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
                            np.average(output[i * c_test.shape[0] : (i + 1) * c_test.shape[0]], axis=0, weights=weights)
                            for i in range(int(output.shape[0] / c_test.shape[0]))
                        ]
                    )

                    inds = plot_output_coldhardiness(
                        config,
                        fpath,
                        np.arange(start=0, stop=calibrator.batch_size),
                        output,
                        params,
                        calibrator.val[n][a_inds.to(torch.int32)],
                        name=n,
                        save=args.save,
                    )

                    from results.eval_common import append_masked_predictions

                    append_masked_predictions(
                        true_data,
                        output_data,
                        true_cultivar_data,
                        output_cultivar_data,
                        calibrator,
                        true,
                        output,
                        inds,
                        a_inds,
                        split_idx=2,
                        name=n,
                    )


def predict_cossim_true_site(
    config, fpath, args, calibrator, true_data, output_data, true_cultivar_data, output_cultivar_data
):
    """Transfer predictions for strategy `cossim_true_site` (prefix cossim_true_site_)."""
    from assets import CROP_NAMES, SITES
    from plotters.plot_utils import gen_batch_data
    from plotters.plotting_functions import plot_output_coldhardiness
    from results.eval_common import parse_withold

    regions, stations, sites, cultivars = parse_withold(config)
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
                    c_inds_test = torch.argwhere(calibrator.cultivars["test"].flatten() == c[0]) if len(c) != 0 else c
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
                        torch.concatenate((st_test_regions, st_test_stations, st_test_sites, st_test_cults), dim=1).to(
                            torch.int64
                        ),
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

                    c_repeat = all_info[test_inds].repeat_interleave(st_test.size(0), dim=1).squeeze().reshape(-1, 4)
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
                    transfer_true = (transfer_true - np.nanmean(transfer_true, axis=1)[:, np.newaxis]) / np.nanstd(
                        transfer_true, axis=1
                    )[:, np.newaxis]
                    transfer_output = (
                        transfer_output - np.nanmean(transfer_output, axis=1)[:, np.newaxis]
                    ) / np.nanstd(transfer_output, axis=1)[:, np.newaxis]

                    # Cosine similarity
                    rmse = torch.tensor(
                        np.nansum(transfer_true * transfer_output, axis=1)
                        / (
                            np.sqrt(np.nansum(transfer_true**2, axis=1))
                            * np.sqrt(np.nansum(transfer_output**2, axis=1))
                        ),
                        device=cartesian.device,
                    )
                    uniq_keys, inv = torch.unique(cartesian[:, -1], dim=0, return_inverse=True)
                    group_sum = torch.zeros(len(uniq_keys), device=cartesian.device)
                    group_count = torch.zeros(len(uniq_keys), device=cartesian.device)
                    group_sum.scatter_add_(0, inv, rmse)
                    group_count.scatter_add_(0, inv, torch.ones_like(rmse))
                    avg_rmse = group_sum / group_count

                    soft_m = torch.nn.Softmax(dim=0)
                    weights = soft_m(avg_rmse).cpu().numpy()
                    a_inds = si_inds_test[torch.isin(si_inds_test, c_inds_test)]
                    s_test = s_test[torch.isin(s_test[:, -1], uniq_keys)]
                    a_repeat = a_inds.repeat_interleave(s_test.size(0))
                    b_repeat = s_test.repeat(a_inds.size(0), 1)
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

                    true = true[np.arange(0, len(true), s_test.shape[0])]
                    output = np.stack(
                        [
                            np.average(output[i * s_test.shape[0] : (i + 1) * s_test.shape[0]], axis=0, weights=weights)
                            for i in range(int(output.shape[0] / s_test.shape[0]))
                        ]
                    )

                    inds = plot_output_coldhardiness(
                        config,
                        fpath,
                        np.arange(start=0, stop=calibrator.batch_size),
                        output,
                        params,
                        calibrator.val[n][a_inds.to(torch.int32)],
                        name=n,
                        save=args.save,
                    )

                    from results.eval_common import append_masked_predictions

                    append_masked_predictions(
                        true_data,
                        output_data,
                        true_cultivar_data,
                        output_cultivar_data,
                        calibrator,
                        true,
                        output,
                        inds,
                        a_inds,
                        split_idx=2,
                        name=n,
                    )


def predict_cossim_true_cult(
    config, fpath, args, calibrator, true_data, output_data, true_cultivar_data, output_cultivar_data
):
    """Transfer predictions for strategy `cossim_true_cult` (prefix cossim_true_cult_)."""
    from assets import CROP_NAMES, SITES
    from plotters.plot_utils import gen_batch_data
    from plotters.plotting_functions import plot_output_coldhardiness
    from results.eval_common import parse_withold

    regions, stations, sites, cultivars = parse_withold(config)
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
                    c_inds_test = torch.argwhere(calibrator.cultivars["test"].flatten() == c[0]) if len(c) != 0 else c
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
                        torch.concatenate((st_test_regions, st_test_stations, st_test_sites, st_test_cults), dim=1).to(
                            torch.int64
                        ),
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

                    c_repeat = all_info[test_inds].repeat_interleave(st_test.size(0), dim=1).squeeze().reshape(-1, 4)
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
                    transfer_true = (transfer_true - np.nanmean(transfer_true, axis=1)[:, np.newaxis]) / np.nanstd(
                        transfer_true, axis=1
                    )[:, np.newaxis]
                    transfer_output = (
                        transfer_output - np.nanmean(transfer_output, axis=1)[:, np.newaxis]
                    ) / np.nanstd(transfer_output, axis=1)[:, np.newaxis]

                    # Cosine similarity
                    rmse = torch.tensor(
                        np.nansum(transfer_true * transfer_output, axis=1)
                        / (
                            np.sqrt(np.nansum(transfer_true**2, axis=1))
                            * np.sqrt(np.nansum(transfer_output**2, axis=1))
                        ),
                        device=cartesian.device,
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
                            np.average(output[i * c_test.shape[0] : (i + 1) * c_test.shape[0]], axis=0, weights=weights)
                            for i in range(int(output.shape[0] / c_test.shape[0]))
                        ]
                    )

                    inds = plot_output_coldhardiness(
                        config,
                        fpath,
                        np.arange(start=0, stop=calibrator.batch_size),
                        output,
                        params,
                        calibrator.val[n][a_inds.to(torch.int32)],
                        name=n,
                        save=args.save,
                    )

                    from results.eval_common import append_masked_predictions

                    append_masked_predictions(
                        true_data,
                        output_data,
                        true_cultivar_data,
                        output_cultivar_data,
                        calibrator,
                        true,
                        output,
                        inds,
                        a_inds,
                        split_idx=2,
                        name=n,
                    )


STRATEGIES = {
    "mean_all": predict_mean_all,
    "mean_cult": predict_mean_cult,
    "mean_site": predict_mean_site,
    "best_source": predict_best_source,
    "tal_site": predict_tal_site,
    "tal_cult": predict_tal_cult,
    "cossim_true_site": predict_cossim_true_site,
    "cossim_true_cult": predict_cossim_true_cult,
}


def run_strategy(
    name, config, fpath, args, calibrator, true_data, output_data, true_cultivar_data, output_cultivar_data
):
    if name not in STRATEGIES:
        raise KeyError(f"Unknown transfer strategy `{name}`. Choose from {list(STRATEGIES)}")
    return STRATEGIES[name](
        config, fpath, args, calibrator, true_data, output_data, true_cultivar_data, output_cultivar_data
    )
