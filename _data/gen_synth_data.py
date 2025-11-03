"""
gen_synth_data.py

Generates synthetic data from a given model for a given
number of years

Written by Will Solow, 2025
"""

import pickle as pkl
import numpy as np
import argparse
from omegaconf import OmegaConf
import utils
import datetime

from model_engine.engine import BatchModelEngine
from model_engine.inputs.weather_util import daylength


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="wofost_synth_config", type=str, help="Path to Config")
    parser.add_argument("--crop_variety", default="Winter_Wheat_101", type=str, help="Variety to run")
    parser.add_argument("--model", choices=["wofost", "ch", "pheno"], type=str, help="model type")
    args = parser.parse_args()

    # Process configuration
    config = OmegaConf.load(f"_data/configs/{args.config}.yaml")
    config = OmegaConf.merge(utils.Args, config)

    model_name, model_num = config.PConfig.model_parameters.split(":")
    config.PConfig.model_parameters = f"{model_name}:{args.crop_variety}"

    # Create Model
    eng = BatchModelEngine(config=config.PConfig, device="cuda")
    if args.model == "wofost":
        end_date = datetime.date(1990, 9, 1)  # WOFOST
    elif args.model == "ch":
        end_date = datetime.date(1990, 5, 15)  # COLD HARDINESS
    elif args.model == "pheno":
        end_date = datetime.date(1990, 9, 7)  # PHENOLOGY

    same_yr = False if args.model == "ch" else True  # False for CH, True for Phenology and WOFOST

    path = f"_data/processed_data/synth5_{model_name}_{args.crop_variety}.pkl"

    yrs = np.random.choice(np.arange(1991, 2023), size=9, replace=False)
    # yrs = [1995, 2000, 2005, 2010, 2015, 2020]
    eng.reset()
    df = eng.run_all(end_date=end_date, same_yr=same_yr)

    dfs = [df]

    for yr in yrs:
        print(yr)
        eng.reset(year=yr)
        df = eng.run_all(end_date, same_yr=same_yr)
        dfs.append(df)

    LAT = config.PConfig.latitude
    prob = 0.13
    for df in dfs:
        days = df["DAY"].to_numpy().astype(np.datetime64)
        df["DAYL"] = daylength(days, np.tile(LAT, len(days)))
        df[df.columns.difference(["DAY"])] = df[df.columns.difference(["DAY"])].astype(np.float32)
        df["LAT"] = LAT
        df["LON"] = config.PConfig.longitude
        # mask = np.random.rand(len(df)) > prob
        mask = ~(np.arange(len(df)) % 7 == 0)
        if args.model == "ch":
            df.loc[mask, "LTE50"] = np.nan  # Uncomment for cold-hardiness
        print(df)

    with open(path, "wb") as f:
        pkl.dump(dfs, f)


if __name__ == "__main__":
    main()
