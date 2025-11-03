"""
compile_data_counts.py

Compiles the data for phenology and cold hardiness and
nicely prints to string

Written by Will Solow, 2025
"""

import argparse
import _data.process_data_real as ld


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cultivar", default="Aligote", type=str, help="Path to Config")
    args = parser.parse_args()

    print_str = f"{args.cultivar} & "
    data, _ = ld.load_and_process_data_phenology(args.cultivar)
    print_str += f"{data[0].loc[0,'DATE'][:4]}-{data[-1].loc[0,'DATE'][:4]} & "

    print_str += f"{len(data)} & "
    data, _ = ld.load_and_process_data_coldhardiness(args.cultivar)

    print_str += f"{len(data)} & "
    lte_num = 0
    for d in data:
        lte_num += d.loc[:, "LTE50"].notna().sum()
    print_str += f"{lte_num} \\\\ "


if __name__ == "__main__":
    main()
