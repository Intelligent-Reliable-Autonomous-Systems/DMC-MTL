"""
Shared helpers for _data sparsify / count utilities.

Written by Will Solow, 2025
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sparsify_column(dfs: list, col: str, every_n: int = 7) -> list:
    """Set ``col`` to NaN except every ``every_n``-th row (index-based).

    Works for list of DataFrames. Mutates copies; original frames unchanged.
    """
    sparse_dfs = []
    for df in dfs:
        df_sparse = df.copy()
        if col in df_sparse.columns:
            mask = (np.arange(len(df_sparse)) % every_n) != 0
            df_sparse.loc[mask, col] = np.nan
        sparse_dfs.append(df_sparse)
    return sparse_dfs


def sparsify_inplace(df: pd.DataFrame, col: str, every_n: int = 7) -> None:
    """In-place every-n sparsify used during synth generation."""
    if col not in df.columns:
        return
    mask = (np.arange(len(df)) % every_n) != 0
    df.loc[mask, col] = np.nan


def count_lte50(data) -> int:
    """Sum non-null LTE50 observations across a list/array of season DataFrames."""
    total = 0
    for d in data:
        total += int(d.loc[:, "LTE50"].notna().sum())
    return total
