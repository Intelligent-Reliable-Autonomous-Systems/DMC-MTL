"""
utils.py

Declaration of arguments dataclasses and helper
functions for loading configurations

Written by Will Solow, 2025
"""

import os
from typing import Optional
from dataclasses import dataclass
from argparse import Namespace

import numpy as np
import pandas as pd
from torch import nn
from omegaconf import OmegaConf, DictConfig
from model_engine import util
from model_engine.util import CROP_NAMES

from train_algs import DMC


@dataclass
class PConfig:
    """Path to configuration files"""

    config_fpath: Optional[str] = None
    """Model to use"""
    model: Optional[str] = None
    """Model parameters to use"""
    model_parameters: Optional[str] = None
    """Latitude of location"""
    latitude: Optional[int] = None
    """Longitude of location"""
    longitude: Optional[int] = None
    """Start date"""
    start_date: Optional[str] = None
    """Output Variables"""
    output_vars: Optional[list] = None
    """Input variables"""
    input_vars: Optional[list] = None


@dataclass
class DConfig:
    """Model Type"""

    type: Optional[str] = None
    """Hidden dim of Model"""
    hidden_dim: Optional[int] = None
    """Model architecture"""
    arch: Optional[str] = None
    """Loss function"""
    loss_func: Optional[str] = None
    """Learning rate for training"""
    learning_rate: Optional[float] = None
    """Learning rate reduction factor"""
    lr_factor: Optional[float] = None
    """Batch size"""
    batch_size: Optional[int] = None
    """Epochs for training"""
    epochs: Optional[int] = None
    """Hyperparameter for PINNs, weighting model output"""
    pinn_p: Optional[float] = 0.5
    """Embedding operation"""
    embed_op: Optional[str] = None


@dataclass
class Args:
    """Model configuration"""

    PConfig: object = PConfig
    """Deep Model Args"""
    DConfig: object = DConfig
    """Cultivar"""
    cultivar: Optional[str] = None
    """Cultivar(s) to withhold from training set"""
    withold_cultivars: Optional[dict] = None
    """Path to real data"""
    data_fpath: Optional[str] = True
    """Data type (cold-hardiness, phenology, wheat)"""
    dtype: Optional[str] = None
    """Region of data"""
    region: Optional[str] = None
    """Weather station for data"""
    station: Optional[str] = None
    """Site of data"""
    site: Optional[str] = None
    """If using synthetic data"""
    synth_data: Optional[str] = None
    """If using a validation set or not"""
    val_set: Optional[bool] = None
    """Parameters to predict"""
    params: Optional[list] = None
    """Ranges for each parameter to predict"""
    params_range: Optional[list] = None
    """Path to save model"""
    log_path: Optional[str] = None
    """Run name for experiment"""
    run_name: Optional[str] = None
    """Seed"""
    seed: Optional[int] = None
    """Model fpath"""
    model_fpath: Optional[str] = None


def load_config_data(args: Namespace) -> tuple[DictConfig, list[pd.DataFrame]]:
    """
    Load the configuration and data and assumes
    that the configuration file is in _train_configs
    """

    config = OmegaConf.load(f"_train_configs/{args.config}.yaml")
    config = OmegaConf.merge(Args, config)
    config.seed = int(args.seed)

    if hasattr(args, "cultivar"):
        if args.cultivar is not None:
            config.cultivar = args.cultivar

    return config, load_data_from_config(config)


def load_config_data_fpath(
    args: Namespace,
) -> tuple[DictConfig, list[pd.DataFrame], str]:
    """
    Loads configuration file and data and returns
    filepath to load model
    """

    fpath = f"{os.getcwd()}/{args.config}"

    config = OmegaConf.load(f"{fpath}/config.yaml")
    config = OmegaConf.merge(Args, config)

    if hasattr(args, "synth_test"):
        if args.synth_test is not None:
            config.synth_data = args.synth_test

    return config, load_data_from_config(config), fpath


def load_data_from_config(config: DictConfig) -> list[pd.DataFrame]:
    """
    Loads data from a OmegaConf configuration file
    """

    try:
        model_name, _ = config.PConfig.model_parameters.split(":")
    except:
        raise Exception(f"Incorrectly specified model_parameters file `{config.PConfig.model_parameters}`")
    if config.synth_data is not None:
        path = f"{config.data_fpath}{config.dtype}/synth/{config.synth_data}_{model_name}_"
    else:
        path = f"{config.data_fpath}{config.dtype}/{config.region}/{config.station}/{config.site}/{config.region}_{config.site}_{model_name}_"

    if config.cultivar == "Multi":
        data = util.load_data_multi(path, CROP_NAMES[model_name])
    else:
        data = util.load_data(f"{path}{config.cultivar}.pkl")

    for d in data:
        d.rename(columns={"DATE": "DAY"}, inplace=True)
    return data


def load_model_from_config(config: DictConfig, data: list[pd.DataFrame]) -> nn.Module:
    """
    Load the model to train using the configuration and pass
    it args and data
    """
    if config.cultivar == "Multi":
        assert not (
            config.DConfig.arch == "FCGRU"
            or config.DConfig.arch == "BaseGRU"
            or config.DConfig.arch == "FCLSTM"
            or config.DConfig.arch == "FCGRU"
        ), "Must use a Multi Architecture"
    if config.DConfig.type == "Param":
        calibrator = DMC.ParamRNN(config, data)
    elif config.DConfig.type == "NoObsParam":
        calibrator = DMC.NoObsParamRNN(config, data)
    elif config.DConfig.type == "TransformerParam":
        calibrator = DMC.TransformerParam(config, data)
    elif config.DConfig.type == "Deep":
        calibrator = DMC.DeepRNN(config, data)
    elif config.DConfig.type == "PINN":
        calibrator = DMC.PINNRNN(config, data)
    elif config.DConfig.type == "Stationary":
        calibrator = DMC.StationaryModel(config, data)
    elif config.DConfig.type == "Hybrid":
        calibrator = DMC.HybridModel(config, data)
    elif config.DConfig.type == "Residual":
        calibrator = DMC.ResidualRNN(config, data)
    else:
        raise NotImplementedError(f"Unrecognized RNN model type `{config.DConfig.type}`")

    return calibrator


def load_dfs(path: str) -> list[pd.DataFrame]:
    """
    Load dataframes from a filepath
    """
    df = pd.read_csv(path)

    df["DAY"] = pd.to_datetime(df["DAY"])

    inds = np.argwhere((df.DAY.dt.month == 1) & (df.DAY.dt.day == 1)).flatten()
    inds = np.concatenate((inds, [-1]))
    df_list = []
    for i in range(1, len(inds)):
        if inds[i] == -1:
            df_list.append(df.loc[inds[i - 1] :])
        else:
            df_list.append(df.loc[inds[i - 1] : inds[i] - 1])
    return df_list
