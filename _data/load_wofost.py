"""
load_wofost_pickles.py

This file loads all wofost data and masks the observations, resaving files

Written by Will Solow, 2025
"""

import os
import pickle
import pandas as pd


def load_wofost_pickles(folder="_data/processed_data", prefix="wofost"):
    """Loads all .pkl files in a folder starting with the given prefix."""
    data = {}
    for filename in os.listdir(folder):
        if filename.startswith(prefix) and filename.endswith(".pkl"):
            path = os.path.join(folder, filename)
            with open(path, "rb") as f:
                data[filename] = pickle.load(f)
    return data


def sparsify_wso_column(dataframes):
    """Set all values in the 'WSO' column to NaN except every 7th."""
    sparse_dfs = []
    for df in dataframes:
        df_sparse = df.copy()
        if "WSO" in df_sparse.columns:
            mask = (df_sparse.index % 7) != 0
            df_sparse.loc[mask, "WSO"] = pd.NA
        sparse_dfs.append(df_sparse)
    return sparse_dfs


def process_and_save_sparse_wofost(data_dict, folder="_data/processed_data"):
    """Processes the loaded wofost files and saves the sparsified versions."""
    for filename, df_list in data_dict.items():
        sparse_df_list = sparsify_wso_column(df_list)
        sparse_filename = filename.replace("wofost", "wofost_sparse")
        sparse_path = os.path.join(folder, sparse_filename)
        with open(sparse_path, "wb") as f:
            pickle.dump(sparse_df_list, f)


if __name__ == "__main__":
    loaded_data = load_wofost_pickles()

    process_and_save_sparse_wofost(loaded_data)
