import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
from autogluon.tabular import TabularPredictor

parser = argparse.ArgumentParser()
parser.add_argument(
    "--fname",
    type=str,
    default="_data/processed_data/grape_coldhardiness/BCOV/Kelowna_UBC/OKFallsWest/BCOV_OKFallsWest_grape_coldhardiness_Gewurztraminer_nyus.pkl",
)
args = parser.parse_args()
with open(args.fname, "rb") as f:
    df = pickle.load(f)
predictor_LT50 = TabularPredictor.load("NYUS_2/NYUS2_2", require_version_match=False)

for d in df:
    d_x = d.drop(columns=["Date", "LTE50"])
    d_y = predictor_LT50.predict(d_x)
    print(d_y)
    plt.plot(d_y.to_numpy())
    plt.scatter(np.arange(len(d["LTE50"])), d["LTE50"])
    plt.savefig("plot.png", dpi=300, bbox_inches="tight")
