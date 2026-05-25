import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--region", type=str, default="PA")
args = parser.parse_args()

df = pd.read_csv("NYUS_2/data/all_data.csv")

reg_data = df[df["Location"] == args.region]
cultivars = []
for i in range(len(reg_data)):
    x = reg_data.columns[df.loc[i].eq(1)].tolist()
    cultivars.append(x[0].removeprefix("Cultivar.").replace("_", " ").title())
reg_data["Cultivars"] = cultivars

reg_data[["Location", "Date", "LT50", "Cultivars"]].to_csv(f"NYUS_2/data/new_regions/{args.region}.csv", index=False)
