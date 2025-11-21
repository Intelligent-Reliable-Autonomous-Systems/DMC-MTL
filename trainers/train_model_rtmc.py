"""
train_model.py

Entry interface for RNN training.

Written by Will Solow, 2025
"""

import argparse
import utils
from train_algs import FineTuner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, type=str, help="Path to Config")
    parser.add_argument("--seed", default=0, type=int, help="Seed of Experiment")
    parser.add_argument("--region", default=None, type=str, help="Region")
    parser.add_argument("--station", default=None, type=str, help="Station")
    parser.add_argument("--site", default=None, type=str, help="Site")
    parser.add_argument("--cultivar", default=None, type=str, help="Cultivar Type")
    parser.add_argument("--rnn_fpath", default=None, type=str, help="Path to RNN config and agent")

    args = parser.parse_args()

    config, data = utils.load_config_data(args)

    calibrator = FineTuner.FineTuner(config, data, rnn_fpath=args.rnn_fpath)

    calibrator.optimize()


if __name__ == "__main__":
    main()
