"""
data_real.py

Interface for loading and saving phenology or cold hardiness data
from file

Written by Will Solow, 2025
"""

import argparse
import _data.process_data_real as ld
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cultivar", default="Aligote", type=str, help="Path to Config")
    parser.add_argument("--name", default="grape_phenology", type=str)
    args = parser.parse_args()

    if args.name == "grape_phenology":

        data, _ = ld.load_and_process_data_phenology(args.cultivar)
        with open(f"_data/processed_data/{args.name}_{args.cultivar}.pkl", "wb") as f:
            pickle.dump(data, f)
        p_str = f"{args.cultivar}, {len(data)}"

    elif args.name == "grape_coldhardiness":
        data, _ = ld.load_and_process_data_coldhardiness(args.cultivar)
        with open(f"_data/processed_data/{args.name}_{args.cultivar}.pkl", "wb") as f:
            pickle.dump(data, f)
        ch = 0
        for d in data:
            ch += d.loc[:, "LTE50"].count().sum()
        p_str = f"{args.cultivar}, {len(data)}, {ch}"
    else:
        raise Exception(f"Unrecognized data name `{args.name}`")

    print(p_str)


if __name__ == "__main__":
    main()
