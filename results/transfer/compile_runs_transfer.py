"""
compile_runs_transfer.py

Shared multi-run transfer evaluation entry. Select strategy via --strategy.

Written by Will Solow, 2025
"""

import argparse

import numpy as np

from results.eval_common import (
    add_compile_args,
    alloc_result_arrays,
    apply_split_scores,
    empty_prediction_buffers,
    iter_run_dirs,
    load_run_calibrator,
    save_result_pickles,
    score_per_cultivar,
    score_splits,
)
from results.transfer.strategies import STRATEGY_PREFIX, STRATEGIES, run_strategy


def main(strategy: str | None = None):
    parser = argparse.ArgumentParser(description="Compile transfer RMSE across runs")
    add_compile_args(parser, transfer=True)
    parser.add_argument(
        "--strategy",
        type=str,
        default=strategy,
        choices=list(STRATEGIES.keys()),
        help="Transfer aggregation strategy",
    )
    parser.add_argument(
        "--result_prefix",
        type=str,
        default=None,
        help="Override pickle filename prefix (defaults from strategy)",
    )
    np.set_printoptions(precision=2)
    args = parser.parse_args()
    if args.strategy is None:
        raise SystemExit("--strategy is required")

    import utils

    config_dirs = iter_run_dirs(args.start_dir, args.prefix)
    if not config_dirs:
        raise SystemExit(f"No run dirs found under {args.start_dir}")

    args.config = f"{args.start_dir}/{config_dirs[0]}"
    config0, _, _ = utils.load_config_data_fpath(args)
    arrays = alloc_result_arrays(config0.DataConfig.dtype, args.num_runs)
    last_dtype = config0.DataConfig.dtype

    run_i = 0
    for config_rel in config_dirs:
        print(config_rel)
        args.config = f"{args.start_dir}/{config_rel}"
        config, data, fpath, calibrator = load_run_calibrator(args, batch_size=128)
        last_dtype = config.DataConfig.dtype

        true_data, output_data, true_cultivar_data, output_cultivar_data = empty_prediction_buffers(calibrator)
        run_strategy(
            args.strategy,
            config,
            fpath,
            args,
            calibrator,
            true_data,
            output_data,
            true_cultivar_data,
            output_cultivar_data,
        )

        scores = score_splits(config, true_data, output_data, fpath)
        apply_split_scores(arrays, scores, run_i)
        score_per_cultivar(
            config,
            calibrator,
            true_cultivar_data,
            output_cultivar_data,
            fpath,
            arrays,
            run_i,
            with_uq=False,
        )
        run_i += 1

    file_prefix = args.result_prefix or STRATEGY_PREFIX[args.strategy]
    if args.synth_test is not None:
        file_prefix = args.synth_test + "_" + file_prefix
    save_result_pickles(
        args.start_dir,
        arrays,
        last_dtype,
        file_prefix=file_prefix,
        transfer_type=None,
        with_uq=False,
    )


if __name__ == "__main__":
    main()
