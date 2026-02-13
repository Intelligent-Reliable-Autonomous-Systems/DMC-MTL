import argparse
import os

import utils


def main():
    argparser = argparse.ArgumentParser(description="Plotting script for model")
    argparser.add_argument("--config", type=str, required=True, help="Path to the config file")
    argparser.add_argument("--break_early", action="store_false", help="If to break early when making data")
    argparser.add_argument("--save", action="store_false", help="If to save the plots")
    argparser.add_argument("--fname", type=str, default="data.txt", help="File to save data to")
    argparser.add_argument("--rnn_name", type=str, default="rnn_model_best.pt")

    args = argparser.parse_args()
    fpath = f"{os.getcwd()}/{args.config}"

    config, data, fpath = utils.load_config_data_fpath(args)
    config.DConfig.batch_size = 12
    # Plot tensorboard outputs

    # config.DConfig.batch_size = 4
    calibrator = utils.load_model_from_config(config, data)
    calibrator.load_model(f"{fpath}", name="rnn_model_best.pt")

    m1 = calibrator.nn

    calibrator2 = utils.load_model_from_config(config, data)
    calibrator2.load_model(f"{fpath}", name="rnn_model_current.pt")

    m2 = calibrator2.nn

    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert n1 == n2
        diff = (p1 - p2).abs().max().item()
        if diff > 1e-6:
            print(f"{n1:40s} max diff = {diff}")


if __name__ == "__main__":
    main()
