"""
compile_runs.py

Compiles RNN metrics across multiple runs and saves pickles.

Written by Will Solow, 2025
"""

import argparse

import numpy as np

from plotters.plot_utils import gen_all_data_and_plot
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


def main():
    parser = argparse.ArgumentParser(description="Compile RMSE across runs")
    add_compile_args(parser)
    np.set_printoptions(precision=2)
    args = parser.parse_args()

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
        config, data, fpath, calibrator = load_run_calibrator(
            args, batch_size=72, num_transfer_years=args.num_transfer_years
        )
        last_dtype = config.DataConfig.dtype

        true_data, output_data, true_cultivar_data, output_cultivar_data = empty_prediction_buffers(calibrator)

        for n in ["train", "test"]:
            gen_all_data_and_plot(
                config,
                fpath,
                args,
                calibrator,
                true_data,
                output_data,
                true_cultivar_data,
                output_cultivar_data,
                name=n,
                transfer_rate=args.transfer_rate,
            )
        if config.DataConfig.val_set:
            gen_all_data_and_plot(
                config,
                fpath,
                args,
                calibrator,
                true_data,
                output_data,
                true_cultivar_data,
                output_cultivar_data,
                name="val",
                transfer_rate=args.transfer_rate,
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
            with_uq=True,
        )
        run_i += 1

    file_prefix = args.synth_test + "_" if args.synth_test is not None else ""
    save_result_pickles(
        args.start_dir,
        arrays,
        last_dtype,
        file_prefix=file_prefix,
        transfer_type=args.transfer_type,
        verbose_level=args.verbose_level,
        with_uq=True,
    )


if __name__ == "__main__":
    main()
