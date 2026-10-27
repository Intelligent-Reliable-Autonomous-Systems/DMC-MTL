"""
compile_data_counts.py

Compiles the data for phenology and cold hardiness and
nicely prints to string

Written by Will Solow, 2025
"""

import argparse

import _data.process_data_real as ld
from _data.data_utils import count_lte50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cultivar", default="Aligote", type=str, help="Cultivar name")
    parser.add_argument("--location", default="Prosser", type=str, help="Site / location")
    args = parser.parse_args()

    print_str = f"{args.cultivar} & "
    data, _ = ld.load_and_process_data_phenology(args.cultivar, args.location)
    print_str += f"{data[0].loc[0,'DATE'][:4]}-{data[-1].loc[0,'DATE'][:4]} & "

    print_str += f"{len(data)} & "
    data, _ = ld.load_and_process_data_coldhardiness(args.cultivar, args.location)

    print_str += f"{len(data)} & "
    print_str += f"{count_lte50(data)} \\\\ "
    print(print_str)


if __name__ == "__main__":
    main()
