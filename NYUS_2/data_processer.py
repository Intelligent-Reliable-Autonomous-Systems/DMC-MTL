import os
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

root_dir = Path("_data/processed_data/grape_coldhardiness")

pkl_files = list(root_dir.rglob("*.pkl"))


def convert_tmin_tmax_to_fahrenheit(df):
    cols = ["tmin", "tmax"]
    df[cols] = df[cols] * 9.0 / 5.0 + 32
    return df


RENAME_COLUMN_MAP = {
    "DATE": "date",
    "TMIN": "tmin",
    "TMAX": "tmax",
}


for p in pkl_files:
    if "synth" in str(p) or "nyus" in str(p):
        continue
    with open(p, "rb") as f:
        df = pickle.load(f)
    fname = str(p).strip(".pkl")
    for i, d in enumerate(df):
        d = d[["DATE", "TMIN", "TMAX", "LAT", "LTE50"]].rename(columns=RENAME_COLUMN_MAP, inplace=False)
        d["date"] = pd.to_datetime(d["date"], format="%Y-%m-%d").dt.strftime("%m/%d/%Y")
        d = convert_tmin_tmax_to_fahrenheit(d)
        lat = float(d["LAT"][0])
        d.drop(columns=["LAT"], inplace=True)
        cultivar = str(p).split("grape_coldhardiness_")[1].strip(".pkl")
        d.to_csv(f"{fname}_{i}.csv", index=False)
        subprocess.run(
            ["Rscript", "NYUS_2/data/Feature_extraction.R", os.path.abspath(f"{fname}_{i}.csv"), str(lat), cultivar],
            check=True,
        )
    dfs = []
    for i, d in enumerate(df):
        rd = pd.read_csv(f"{fname}_{i}.csv")
        rd["LTE50"] = d["LTE50"]
        dfs.append(rd)
    dfs = np.asarray(dfs, dtype=object)
    with open(f"{fname}_nyus.pkl", "wb") as f:
        pickle.dump(dfs, f)
