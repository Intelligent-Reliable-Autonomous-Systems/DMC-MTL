"""
process_data_real.py

Processes phenology and cold hardiness data based on the
.csv files we've received from AgAid/Markus Keller.
Assumes that linear interpolation has been performed first to handle
missing weather data.

Written by Will Solow, 2025
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from model_engine.inputs.weather_util import daylength

# Absolute path of .csv files relative to this directory
DATASET_DIRECTORY_GRAPE = f"{Path(os.getcwd()).parent.absolute()}/Frost_Mitigation_Datasets/ColdHardiness/Grapes/Processed/WashingtonState/Prosser/Python/New/"
DATASET_DIRECTORY_CHERRY = f"{Path(os.getcwd()).parent.absolute()}/Frost_Mitigation_Datasets/ColdHardiness/Cherries/Processed/WashingtonState/Prosser/"

# Specific Names
PHENO_STAGES = [
    "Ecodorm",
    "Budburst/Budbreak",
    "Full Bloom",
    "Veraison 50%",
    "Harvest",
    "Endodorm",
]
PHENO_INT = {
    "Ecodorm": 0,
    "Budburst/Budbreak": 1,
    "Full Bloom": 2,
    "Veraison 50%": 3,
    "Harvest": 4,
    "Endodorm": 5,
}
RENAME_COLUMN_MAP = {
    "SR_MJM2": "IRRAD",
    "P_INCHES": "RAIN",
    "MEAN_AT": "TEMP",
    "MAX_AT": "TMAX",
    "MIN_AT": "TMIN",
}

COL_ORDERING = [
    "DATE",
    "PHENOLOGY",
    "TMIN",
    "TMAX",
    "TEMP",
    "RAIN",
    "IRRAD",
    "LAT",
    "LON",
]


# Conversions
MPH_TO_MS = 0.44704
IN_TO_CM = 2.54
MJ_TO_J = 1000000
LAT = 46
LON = -120


def load_and_process_data_phenology(cultivar: str, location: str) -> np.ndarray:
    """
    Load and process AgAid GitHub repository data
    Do not assume dormancy data is available
    Include reduced years by not ommitting certain weather variables
    """

    df = pd.read_csv(f"{DATASET_DIRECTORY_GRAPE}" + f"ColdHardiness_Grape_{location}_{cultivar}.csv")
    # Remove all grape stages we are not interested in predicting
    df.loc[~df["PHENOLOGY"].isin(PHENO_STAGES), "PHENOLOGY"] = np.nan

    # Drop all columns we don't care about
    df.drop(
        columns=[
            "AWN_STATION",
            "SEASON",
            "SEASON_JDAY",
            "LTE10",
            "LTE50",
            "LTE90",
            "PREDICTED_LTE10",
            "PREDICTED_LTE50",
            "PREDICTED_LTE90",
            "PREDICTED_BUDBREAK",
            "MIN_ST2",
            "ST2",
            "MAX_ST2",
            "MIN_ST8",
            "ST8",
            "MAX_ST8",
            "SM8_PCNT",
            "SWP8_KPA",
            "MSLP_HPA",
            "LW_UNITY",
            "ETR",
        ],
        inplace=True,
    )

    # Unit conversions
    # Convert MJ to J
    df.loc[:, "SR_MJM2"] *= MJ_TO_J

    # Convert inches of rainfall to cm
    df.loc[:, "P_INCHES"] *= IN_TO_CM

    # Convet mph wind speed to m/s
    df.loc[:, ["WS_MPH", "MAX_WS_MPH"]] *= MPH_TO_MS

    # Rename columns for compatibility
    df.rename(columns=RENAME_COLUMN_MAP, inplace=True)

    # Add latitute and longitude columns
    df["LAT"] = LAT
    df["LON"] = LON

    days = df["DATE"].to_numpy().astype(np.datetime64)
    df["DAYL"] = daylength(days, np.tile(LAT, len(days)))
    df.drop(columns=["AVG_AT", "WD_DEGREE"], inplace=True)

    df_list = []
    stages_list = []

    ys = np.argwhere(df["YEAR_JDAY"] == 1).flatten()
    for i in range(len(ys)):
        # Get the slices of each individual year
        if i == len(ys) - 1:
            year_df = df[ys[i] :].copy().reset_index(drop=True)
        else:
            year_df = df[ys[i] : ys[i + 1]].copy().reset_index(drop=True)

            # Get the switch to the dormancy season so we can add endodorm
        for ds in np.argwhere(np.diff(df["DORMANT_SEASON"], prepend=[0]) == 1):
            df.loc[ds, "PHENOLOGY"] = PHENO_INT["Endodorm"]

        # Check if any real values are present, if not fill with fake ones
        eco = np.argwhere(year_df["PHENOLOGY"] == "Ecodorm").flatten()
        if len(eco) == 0:
            year_df.loc[0, "PHENOLOGY"] = PHENO_INT["Ecodorm"]
            if len(year_df) >= 334:
                year_df.loc[334, "PHENOLOGY"] = PHENO_INT["Ecodorm"]
        elif len(eco) == 1:
            if eco < 150 and len(year_df) >= 334:
                year_df.loc[334, "PHENOLOGY"] = PHENO_INT["Ecodorm"]
            else:
                year_df.loc[0, "PHENOLOGY"] = PHENO_INT["Ecodorm"]

        endo = np.argwhere(year_df["PHENOLOGY"] == "Endodorm").flatten()
        if len(endo) == 0:
            if year_df.loc[0, "DORMANT_SEASON"] == 0:
                dorm = np.argwhere(np.diff(year_df["DORMANT_SEASON"], prepend=[0]) == 1).flatten()
            else:
                dorm = np.argwhere(np.diff(year_df["DORMANT_SEASON"], prepend=[1]) == 1).flatten()
            year_df.loc[dorm, "PHENOLOGY"] = PHENO_INT["Endodorm"]

        # Covert phenology to int values
        for i in range(len(year_df["PHENOLOGY"])):
            if isinstance(year_df.loc[i, "PHENOLOGY"], str):
                year_df.loc[i, "PHENOLOGY"] = PHENO_INT[year_df["PHENOLOGY"].iloc[i]]
        # Change Phenology dtype
        year_df["PHENOLOGY"] = year_df["PHENOLOGY"].astype("float64")

        # Handle case where a state occurs out of order by removing it
        pheno_changes = np.argwhere(~np.isnan(year_df["PHENOLOGY"])).flatten()
        pheno_change_vals = year_df.loc[pheno_changes, "PHENOLOGY"].to_numpy().astype("int64")
        for j in range(len(pheno_change_vals) - 1):
            if pheno_change_vals[j] > pheno_change_vals[j + 1] and pheno_change_vals[j] != PHENO_INT["Endodorm"]:
                year_df.loc[pheno_changes[j], "PHENOLOGY"] = np.nan

        # Forward fill with non-na values
        year_df["PHENOLOGY"] = year_df["PHENOLOGY"].ffill().astype("int64")

        pheno_states = np.unique(year_df["PHENOLOGY"])

        if PHENO_INT["Budburst/Budbreak"] not in pheno_states:
            continue

        if (
            PHENO_INT["Budburst/Budbreak"] in pheno_states
            and PHENO_INT["Endodorm"] in pheno_states
            and PHENO_INT["Full Bloom"] not in pheno_states
        ):
            continue

        if (
            PHENO_INT["Full Bloom"] in pheno_states
            and PHENO_INT["Endodorm"] in pheno_states
            and PHENO_INT["Veraison 50%"] not in pheno_states
        ):
            continue

        # Interpolate passed variables
        year_df, percent_missing = interpolate_data(
            year_df,
            df,
            cols=[
                "TEMP",
                "TMAX",
                "TMIN",
                "RAIN",
                "MIN_RH",
                "MAX_RH",
                "AVG_RH",
                "MIN_DEWPT",
                "AVG_DEWPT",
                "MAX_DEWPT",
                "WS_MPH",
                "MAX_WS_MPH",
                "ETO",
                "IRRAD",
            ],
            max_days=float("inf"),
        )

        # If too much missing weather data, continue
        if percent_missing >= 0.1:
            continue

        # If there are any nan values in the weather throw out the entire year
        if year_df.isnull().any().any():
            continue
        if (year_df.loc[:, ["IRRAD"]] == 0).all().all():
            continue

        year_stages = []
        if PHENO_INT["Ecodorm"] in pheno_states:
            year_stages.append(PHENO_INT["Ecodorm"])
        if PHENO_INT["Endodorm"] in pheno_states:
            year_stages.append(PHENO_INT["Endodorm"])
        if PHENO_INT["Budburst/Budbreak"] in pheno_states:
            year_stages.append(PHENO_INT["Budburst/Budbreak"])
        if PHENO_INT["Full Bloom"] in pheno_states:
            year_stages.append(PHENO_INT["Full Bloom"])
        if PHENO_INT["Veraison 50%"] in pheno_states:
            year_stages.append(PHENO_INT["Veraison 50%"])
        if PHENO_INT["Harvest"] in pheno_states:
            year_stages.append(PHENO_INT["Harvest"])

        # Only go through the onset of dormancy
        if len(dorm) != 0:
            year_df = year_df[: int(dorm)]

        # If passes all tests then append
        df_list.append(year_df)
        stages_list.append(year_stages)

    for yr_df in df_list:
        yr_df.drop(columns=["DORMANT_SEASON", "YEAR_JDAY"], inplace=True)

    if len(df_list) != 1:
        return np.array(df_list, dtype=object), np.array(stages_list, dtype=object)
    else:
        np_arr_df = np.empty(len(df_list), dtype=object)
        np_arr_stages = np.empty(len(stages_list), dtype=object)
        for i, df in enumerate(df_list):
            np_arr_df[i] = df
        for i, s in enumerate(stages_list):
            np_arr_stages[i] = s
        return np_arr_df, np_arr_stages


def split_dormant_seasons(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Split a full CH dataframe into dormant-season year slices."""
    ys = np.argwhere(np.diff(df["DORMANT_SEASON"], prepend=[0]) == 1).flatten()
    ye = np.argwhere(np.diff(df["DORMANT_SEASON"], prepend=[0]) == -1).flatten()
    years = []
    for i in range(len(ys)):
        if i >= len(ye):
            year_df = df[ys[i] :].copy().reset_index(drop=True)
        else:
            year_df = df[ys[i] : ye[i]].copy().reset_index(drop=True)
        years.append(year_df)
    return years


def package_year_dfs(df_list: list, stages_list: list | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Package year DataFrames into object arrays (handles len==1 edge case)."""
    if stages_list is None:
        stages_list = []
    if len(df_list) != 1:
        return np.array(df_list, dtype=object), np.array(stages_list, dtype=object)
    np_arr_df = np.empty(len(df_list), dtype=object)
    np_arr_stages = np.empty(len(stages_list), dtype=object)
    for i, d in enumerate(df_list):
        np_arr_df[i] = d
    for i, s in enumerate(stages_list):
        np_arr_stages[i] = s
    return np_arr_df, np_arr_stages


def _print_weather_nan_debug(year_df: pd.DataFrame, lte_cols: list[str]) -> None:
    tmp = year_df.drop(columns=lte_cols, inplace=False)
    nan_mask = tmp.isnull()
    if not nan_mask.any().any():
        return
    print("breaking from weather nan")
    cols_with_nan = nan_mask.any()
    print("Columns with NaNs:")
    print(cols_with_nan[cols_with_nan].index.tolist())
    rows_with_nan = nan_mask.any(axis=1)
    print("Rows with NaNs:")
    print(tmp.index[rows_with_nan])
    print("Exact NaN locations:")
    for col in tmp.columns:
        nan_rows = tmp.index[tmp[col].isna()]
        if len(nan_rows) > 0:
            print(f"{col}: {list(nan_rows)}")


def filter_ch_year(
    year_df: pd.DataFrame,
    full_df: pd.DataFrame,
    *,
    lte_cols: list[str],
    weather_cols: list[str],
    missing_thresh: float,
    verbose: bool | str = False,
    lte_first: bool = True,
    reject_zero_irrad: bool = False,
) -> pd.DataFrame | None:
    """Interpolate and apply CH year QC filters. Returns year_df or None if rejected."""
    year_df, percent_missing = interpolate_data(
        year_df, full_df, cols=weather_cols, max_days=float("inf")
    )

    def _reject_lte() -> bool:
        if year_df.loc[:, lte_cols].isnull().all().all():
            if verbose and verbose != "ca":
                print("Breaking from no LTE")
            return True
        return False

    def _reject_missing() -> bool:
        if verbose and verbose != "ca":
            print(percent_missing)
        if percent_missing >= missing_thresh:
            if verbose == "ca":
                print("MSSING:", percent_missing)
            elif verbose:
                print("BREAKING FROM MISSING")
            return True
        return False

    def _reject_weather_nan() -> bool:
        if year_df.drop(columns=lte_cols, inplace=False).isnull().any().any():
            if verbose and verbose != "ca":
                print("breaking from weather nan")
                _print_weather_nan_debug(year_df, lte_cols)
            return True
        return False

    checks = (_reject_lte, _reject_missing, _reject_weather_nan) if lte_first else (
        _reject_missing,
        _reject_weather_nan,
        _reject_lte,
    )
    for check in checks:
        if check():
            return None
    if reject_zero_irrad and (year_df.loc[:, ["IRRAD"]] == 0).all().all():
        return None
    return year_df


def _grape_datasets_root() -> str:
    return f"{Path(os.getcwd()).parent.absolute()}/grape-datasets"


def _lte_vals_ca(region: str) -> list[str]:
    if region in ["NS", "BCOV"]:
        return ["LTE50"]
    if region == "ONNP":
        return ["LTE50", "LTE10", "LTE90"]
    raise Exception(f"Unexpected region `{region}`")


def _lte_vals_mi(region: str) -> list[str]:
    if region == "MI":
        return ["LTE50"]
    raise Exception(f"Unexpected region `{region}`")


def _lte_vals_ny(region: str) -> list[str]:
    if region == "NY":
        return ["LTE50"]
    raise Exception(f"Unexpected region `{region}`")


def _lte_vals_noaa(region: str) -> list[str]:
    if region in ["WI", "PA", "QC", "TX"]:
        return ["LTE50"]
    raise Exception(f"Unexpected region `{region}`")


REGIONAL_CH = {
    "ca": {
        "subdir": "ch_weather_data_ca",
        "missing_thresh": 0.1,
        "weather_cols": ["TEMP", "TMAX", "TMIN"],
        "drop_before": ["RAIN"],
        "drop_after": ["DORMANT_SEASON", "Unnamed: 0"],
        "lte_vals": _lte_vals_ca,
        "verbose": "ca",
    },
    "mi": {
        "subdir": "ch_weather_data_mi",
        "missing_thresh": 0.2,
        "weather_cols": [
            "TEMP",
            "TMAX",
            "TMIN",
            "RAIN",
            "MAX_DEWPT",
            "MIN_DEWPT",
            "MAX_RH",
            "MIN_RH",
            "ETO",
            "IRRAD",
            "MAX_WS_MPH",
        ],
        "drop_before": [],
        "drop_after": ["DORMANT_SEASON", "Unnamed: 0"],
        "lte_vals": _lte_vals_mi,
        "verbose": True,
    },
    "ny": {
        "subdir": "ch_weather_data_ny",
        "missing_thresh": 0.2,
        "weather_cols": ["TEMP", "TMAX", "TMIN", "RAIN", "AVG_RH", "IRRAD", "WS_MPH"],
        "drop_before": [],
        "drop_after": ["DORMANT_SEASON", "Unnamed: 0"],
        "lte_vals": _lte_vals_ny,
        "verbose": True,
    },
    "noaa": {
        "subdir": "ch_weather_data_noaa",
        "missing_thresh": 0.2,
        "weather_cols": ["TEMP", "TMAX", "TMIN"],
        "drop_before": [],
        "drop_after": ["DORMANT_SEASON", "Unnamed: 0"],
        "lte_vals": _lte_vals_noaa,
        "verbose": True,
    },
}


def _load_regional_ch(
    region: str,
    site: str,
    cultivar: str,
    station: str,
    *,
    preset: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared regional cold-hardiness loader (CA / MI / NY / NOAA)."""
    cfg = REGIONAL_CH[preset]
    lte_vals = cfg["lte_vals"](region)
    dataset_dir = f"{_grape_datasets_root()}/{cfg['subdir']}/"
    df = pd.read_csv(f"{dataset_dir}ColdHardiness_Grape_{region}{site}_{cultivar}.csv")
    for col in cfg["drop_before"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    df_list = []
    for year_df in split_dormant_seasons(df):
        kept = filter_ch_year(
            year_df,
            df,
            lte_cols=lte_vals,
            weather_cols=cfg["weather_cols"],
            missing_thresh=cfg["missing_thresh"],
            verbose=cfg["verbose"],
            lte_first=True,
        )
        if kept is not None:
            df_list.append(kept)

    for yr_df in df_list:
        yr_df.drop(columns=cfg["drop_after"], inplace=True, errors="ignore")
    return package_year_dfs(df_list)


def load_and_process_data_coldhardiness(cultivar: str, location: str, fruit: str = "Grape") -> np.ndarray:
    """
    Load and process AgAid GitHub repository data for cold hardiness
    Do not assume dormancy data is available
    Includes relevant feature variables
    """
    if fruit == "Grape":
        directory = DATASET_DIRECTORY_GRAPE
    elif fruit == "Cherry":
        directory = DATASET_DIRECTORY_CHERRY
    else:
        raise Exception(f"Unknown Fruit Type `{fruit}`")
    df = pd.read_csv(f"{directory}" + f"ColdHardiness_{fruit}_{location}_{cultivar}.csv")

    df.drop(
        columns=[
            "AWN_STATION",
            "SEASON",
            "SEASON_JDAY",
            "PREDICTED_LTE10",
            "PHENOLOGY",
            "PREDICTED_LTE50",
            "PREDICTED_LTE90",
            "PREDICTED_BUDBREAK",
            "MIN_ST2",
            "ST2",
            "MAX_ST2",
            "MIN_ST8",
            "ST8",
            "MAX_ST8",
            "SM8_PCNT",
            "SWP8_KPA",
            "MSLP_HPA",
            "LW_UNITY",
            "ETR",
        ],
        inplace=True,
    )

    df.loc[:, "SR_MJM2"] *= MJ_TO_J
    df.loc[:, "P_INCHES"] *= IN_TO_CM
    df.loc[:, ["WS_MPH", "MAX_WS_MPH"]] *= MPH_TO_MS
    df.rename(columns=RENAME_COLUMN_MAP, inplace=True)
    df["LAT"] = LAT
    df["LON"] = LON
    days = df["DATE"].to_numpy().astype(np.datetime64)
    df["DAYL"] = daylength(days, np.tile(LAT, len(days)))
    df.drop(columns=["AVG_AT", "WD_DEGREE"], inplace=True)

    lte_cols = ["LTE50", "LTE10", "LTE90"]
    weather_cols = [
        "TEMP",
        "TMAX",
        "TMIN",
        "RAIN",
        "MIN_RH",
        "MAX_RH",
        "AVG_RH",
        "MIN_DEWPT",
        "AVG_DEWPT",
        "MAX_DEWPT",
        "WS_MPH",
        "MAX_WS_MPH",
        "ETO",
        "IRRAD",
    ]
    df_list = []
    for year_df in split_dormant_seasons(df):
        kept = filter_ch_year(
            year_df,
            df,
            lte_cols=lte_cols,
            weather_cols=weather_cols,
            missing_thresh=0.1,
            lte_first=False,
            reject_zero_irrad=True,
        )
        if kept is not None:
            df_list.append(kept)
    for yr_df in df_list:
        yr_df.drop(columns=["DORMANT_SEASON", "YEAR_JDAY"], inplace=True)
    return package_year_dfs(df_list)


def load_and_process_data_cherry_coldhardiness(cultivar: str, location: str, fruit: str = "Cherry") -> np.ndarray:
    """
    Load and process AgAid GitHub repository data for cold hardiness
    Do not assume dormancy data is available
    Includes relevant feature variables
    """
    if fruit == "Grape":
        directory = DATASET_DIRECTORY_GRAPE
    elif fruit == "Cherry":
        directory = DATASET_DIRECTORY_CHERRY
    else:
        raise Exception(f"Unknown Fruit Type `{fruit}`")
    df = pd.read_csv(f"{directory}" + f"ColdHardiness_{fruit}_{cultivar}.csv")

    df.drop(
        columns=[
            "AWN_STATION",
            "SEASON",
            "SEASON_JDAY",
            "PHENOLOGY",
            "MIN_ST2",
            "ST2",
            "MAX_ST2",
            "MIN_ST8",
            "ST8",
            "MAX_ST8",
            "SM8_PCNT",
            "SWP8_KPA",
            "MSLP_HPA",
            "LW_UNITY",
            "ETR",
            "LTER_ACC",
            "LTER_DEACC",
            "P_INCHES",
            "MIN_REL_HUMIDITY",
            "MAX_REL_HUMIDITY",
            "AVG_REL_HUMIDITY",
            "MIN_DEWPT",
            "AVG_DEWPT",
            "MAX_DEWPT",
            "WS_MPH",
            "MAX_WS_MPH",
            "ETO",
            "SR_MJM2",
        ],
        inplace=True,
        errors="ignore",
    )

    df.rename(columns=RENAME_COLUMN_MAP, inplace=True)
    df["LAT"] = LAT
    df["LON"] = LON
    days = df["DATE"].to_numpy().astype(np.datetime64)
    df["DAYL"] = daylength(days, np.tile(LAT, len(days)))
    df.drop(columns=["AVG_AT", "WD_DEGREE"], inplace=True)

    lte_cols = ["LTE50", "LTE10"]
    weather_cols = ["TEMP", "TMAX", "TMIN"]
    df_list = []
    for year_df in split_dormant_seasons(df):
        kept = filter_ch_year(
            year_df,
            df,
            lte_cols=lte_cols,
            weather_cols=weather_cols,
            missing_thresh=0.1,
            lte_first=False,
        )
        if kept is not None:
            df_list.append(kept)
    for yr_df in df_list:
        yr_df.drop(columns=["DORMANT_SEASON", "YEAR_JDAY"], inplace=True)
    return package_year_dfs(df_list)


def load_and_process_ca_data_coldhardiness(region: str, site: str, cultivar: str, station: str):
    return _load_regional_ch(region, site, cultivar, station, preset="ca")


def load_and_process_mi_data_coldhardiness(region: str, site: str, cultivar: str, station: str):
    return _load_regional_ch(region, site, cultivar, station, preset="mi")


def load_and_process_ny_data_coldhardiness(region: str, site: str, cultivar: str, station: str):
    return _load_regional_ch(region, site, cultivar, station, preset="ny")


def load_and_process_noaa_data_coldhardiness(region: str, site: str, cultivar: str, station: str):
    return _load_regional_ch(region, site, cultivar, station, preset="noaa")


def chunk_consecutive(arr: np.ndarray) -> list[list[int]]:
    """
    Chunks data into consecutive missing value arrays
    """
    arr = np.sort(arr)
    chunks = [[int(arr[0])]]
    chunk_ind = 0
    for i in range(1, len(arr)):
        if chunks[chunk_ind][-1] + 1 == arr[i]:
            chunks[chunk_ind].append(int(arr[i]))
        else:
            chunks.append([int(arr[i])])
            chunk_ind += 1

    return chunks


def interpolate(df: pd.DataFrame, full_df: pd.DataFrame, col: str, inds: np.ndarray) -> None:
    """
    Linearly interpolate data from the missing column within a dataframe slice
    Uses lookup in the entire data frame if nan's appear at the beginning or end
    """

    start_inds = inds + np.argwhere(df.loc[0, "DATE"] == full_df.loc[:, "DATE"]).flatten()[0]

    # At beginning of array, can't interpolate data
    if start_inds[0] == 0:
        return
    if start_inds[-1] + 1 == len(full_df):
        return
    # Get the starting and ending values
    start = full_df.loc[start_inds[0] - 1, col]
    end = full_df.loc[start_inds[-1] + 1, col]

    # Finds the next non nan values before and after window
    # Exits if exceeds full array length
    i = start_inds[0] - 1
    while start == np.nan or start == -100:
        start = full_df.loc[i, col]
        i -= 1
        if i < 0:
            return

    j = start_inds[-1] + 1
    while end == np.nan or end == -100:
        end = full_df.loc[j, col]
        j += 1
        if j >= len(full_df):
            return

    # Perform linear interpolation only within slice of dataframe
    rng = (j - i) + 1
    for k in range(len(inds)):
        df.loc[inds[k], col] = np.round(start + ((k + start_inds[0] - i) / rng) * (end - start), decimals=2)


def interpolate_data(
    df: pd.DataFrame,
    full_df: pd.DataFrame,
    cols: list[str] = [
        "TEMP",
        "TMAX",
        "TMIN",
        "RAIN",
        "MIN_RH",
        "MAX_RH",
        "AVG_RH",
        "MIN_DEWPT",
        "AVG_DEWPT",
        "MAX_DEWPT",
        "WS_MPH",
        "MAX_WS_MPH",
    ],
    max_days=float("inf"),
) -> tuple[pd.DataFrame, float]:
    """
    Interpolate columns of DF
    """
    max_percent_missing = 0

    for c in cols:
        # Get -100 and null values
        false_vals = np.argwhere(df[c] == -100).flatten()
        df.loc[false_vals, c] = np.nan
        nulls = df[c][df[c].isnull()].index.tolist()
        smr = []
        if c == "IRRAD":
            smr = np.argwhere(df[c] == 0).flatten()
        percent_missing = np.concatenate((false_vals, nulls, smr), axis=0).shape[0] / len(df)
        if max_percent_missing < percent_missing:
            max_percent_missing = percent_missing

        # Chunk into consecutive lists of missing values
        missings = np.concatenate((false_vals, nulls, smr))
        if missings.shape[0] != 0:
            ms = chunk_consecutive(missings)
            for m in ms:
                if len(m) < max_days:
                    interpolate(df, full_df, c, m)
    return df, percent_missing
