import pickle

with open(
    "_data/processed_data/grape_coldhardiness/ONNP/Grimbsy_Mtn/BeamsvilleBench/ONNP_BeamsvilleBench_grape_coldhardiness_Cabernet_Franc_nyus.pkl",
    "rb",
) as f:
    out = pickle.load(f)

print(out[0].columns)
