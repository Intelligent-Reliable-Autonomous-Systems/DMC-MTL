"""Base class for NYUS_2_2

Written by Will Solow, 2025"""

import os

import numpy as np
import pandas as pd
import torch
from autogluon.tabular import TabularPredictor
from omegaconf import DictConfig

from train_algs.base.process_data import assign_id_tensors, split_train_test_indices

_NYUS_MODEL_DIR = "NYUS_2/NYUS2_2"


class NYUS2:
    def __init__(self, config: DictConfig, data: list[pd.DataFrame]):

        predictor_path = os.path.join(_NYUS_MODEL_DIR, "predictor.pkl")
        if not os.path.isfile(predictor_path):
            raise FileNotFoundError(
                f"NYUS AutoGluon artifact missing: `{predictor_path}`. "
                f"Restore the full TabularPredictor save dir under `{_NYUS_MODEL_DIR}` "
                "(predictor.pkl + learner/model pickles); only partial weights are present."
            )
        self.nn = TabularPredictor.load(_NYUS_MODEL_DIR, require_version_match=False)
        self.data = data
        self.config = config
        self.device = "cpu"
        self.batch_size = config.DConfig.batch_size

        self._process_data(data)

        cult_orig = torch.unique(torch.concatenate(list(self.cultivars.values()), axis=0)).to(torch.int)
        self.cult_mapping = torch.zeros((int(cult_orig.max()) + 1,)).to(torch.int).to(self.device)
        self.cult_mapping[cult_orig] = torch.arange(len(cult_orig)).to(torch.int).to(self.device)

        reg_orig = torch.unique(torch.concatenate(list(self.regions.values()), axis=0)).to(torch.int)
        self.reg_mapping = torch.zeros((int(reg_orig.max()) + 1,)).to(torch.int).to(self.device)
        self.reg_mapping[reg_orig] = torch.arange(len(reg_orig)).to(torch.int).to(self.device)

        stat_orig = torch.unique(torch.concatenate(list(self.stations.values()), axis=0)).to(torch.int)
        self.stat_mapping = torch.zeros((int(stat_orig.max()) + 1,)).to(torch.int).to(self.device)
        self.stat_mapping[stat_orig] = torch.arange(len(stat_orig)).to(torch.int).to(self.device)

        site_orig = torch.unique(torch.concatenate(list(self.sites.values()), axis=0)).to(torch.int)
        self.site_mapping = torch.zeros((int(site_orig.max()) + 1,)).to(torch.int).to(self.device)
        self.site_mapping[site_orig] = torch.arange(len(site_orig)).to(torch.int).to(self.device)

    def forward(
        self,
        data: list[pd.DataFrame],
        **kwargs,
    ) -> tuple[list[np.ndarray]]:
        """
        Forward pass call for integrated gradients attribution sequence batch data
        """
        out = []
        for d in data:
            out.append(self.nn.predict(d, as_pandas=False).astype(np.float32))
        return out

    def _process_data(self, data: list[pd.DataFrame]) -> None:
        """Process data for AutoGluon: keep raw DataFrames, share ID split with DMC."""

        if len(data) < 2:
            raise Exception(f"Data size: {len(data)}. Insufficient data for building training set.")

        real_columns = data[0].columns
        for d in data:
            d.columns = real_columns
        input_data = [
            d.drop(columns=["Date", "LTE50", "CULTIVAR", "REGION", "STATION", "SITE"], errors="ignore") for d in data
        ]
        output_data = [d.loc[:, "LTE50"].to_numpy().astype(np.float32) for d in data]

        split = split_train_test_indices(self, data)
        train_inds = split["train_inds"]
        test_inds = split["test_inds"]

        self.data = {
            "train": [input_data[i] for i in train_inds],
            "test": [input_data[i] for i in test_inds],
        }
        self.val = {
            "train": [output_data[i] for i in train_inds],
            "test": [output_data[i] for i in test_inds],
        }

        assign_id_tensors(self, split)

        if len(self.data["test"]) < 1:
            raise Exception("Insuffient per-cultivar data to build test set")
