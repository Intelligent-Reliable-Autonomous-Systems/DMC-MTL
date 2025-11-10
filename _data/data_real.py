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
    parser.add_argument("--cultivar", default="Chardonnay", type=str, help="Path to Config")
    parser.add_argument("--name", default="grape_coldhardiness", type=str)
    parser.add_argument("--region", default="WA", type=str)
    parser.add_argument("--site", default="Prosser", type=str)
    parser.add_argument("--station", default="Roza2", type=str)
    args = parser.parse_args()

    if "grape_phenology" in args.name:

        data, _ = ld.load_and_process_data_phenology(args.cultivar, args.site)

        with open(
            f"_data/processed_data/{args.name}/{args.region}/{args.station}/{args.site}/{args.region}_{args.site}_{args.name}_{args.cultivar.replace(' ', '_')}.pkl",
            "wb",
        ) as f:
            pickle.dump(data, f)

        p_str = f"{args.cultivar}, {len(data)}"

    elif "grape_coldhardiness" in args.name:
        if args.region in ["WA"]:
            data, _ = ld.load_and_process_data_coldhardiness(args.cultivar, args.site)
        else:
            data, _ = ld.load_and_process_ca_data_coldhardiness(args.region, args.site, args.cultivar, args.station)
        with open(
            f"_data/processed_data/{args.name}/{args.region}/{args.station}/{args.site}/{args.region}_{args.site}_{args.name}_{args.cultivar.replace(' ', '_')}.pkl",
            "wb",
        ) as f:
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
