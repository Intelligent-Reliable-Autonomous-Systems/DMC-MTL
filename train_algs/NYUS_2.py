"""Base class for NYUS_2_2

Written by Will Solow, 2025"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from autogluon.tabular import TabularPredictor
from omegaconf import DictConfig

from assets import CROP_NAMES, REGIONS, SITES, STATIONS
from train_algs.base.base import BaseModel
from train_algs.base.process_data import withold_checker


class BaseRNN(BaseModel):
    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:

        super().__init__(config, data)

        self.nn = TabularPredictor.load("NYUS_2/NYUS2_2", require_version_match=False)

    def forward(
        self,
        data: torch.Tensor,
        dates: np.ndarray,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass call for integrated gradients attribution sequence batch data
        """

        return output_tens, None, None


class NYUS2:
    def __init__(self, config: DictConfig, data: list[pd.DataFrame]):

        self.nn = TabularPredictor.load("NYUS_2/NYUS2_2", require_version_match=False)
        self.data = data
        self.config = config
        self.device = "cpu"
        self.batch_size = config.DConfig.batch_size

        NYUS2.process_data_novalset(self, data)

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

    @staticmethod
    def process_data_novalset(model: nn.Module, data: list[pd.DataFrame]) -> None:
        """Process all of the initial data"""

        if len(data) < 2:
            raise Exception(f"Data size: {len(data)}. Insufficient data for building training set.")

        # Get input data for use with model
        real_columns = data[0].columns
        for d in data:
            d.columns = real_columns
        model.input_data = [  # 0, 130
            d.drop(columns=["Date", "LTE50", "CULTIVAR", "REGION", "STATION", "SITE"], errors="ignore") for d in data
        ]

        # Get output data, d.columns.intersection([130, "LTE50"])
        model.output_data = [d.loc[:, "LTE50"].to_numpy().astype(np.float32) for d in data]

        x = model.config.DataConfig.test_set_size  # Number of years for testing set

        # Shuffle to get train and test splits for data
        train_inds = np.empty(shape=(0,))
        test_inds = np.empty(shape=(0,))
        cultivar_data = np.array([d.loc[0, "CULTIVAR"] for d in data]) if "CULTIVAR" in data[0].columns else None
        region_data = np.array([d.loc[0, "REGION"] for d in data]) if "REGION" in data[0].columns else None
        station_data = np.array([d.loc[0, "STATION"] for d in data]) if "STATION" in data[0].columns else None
        site_data = np.array([d.loc[0, "SITE"] for d in data]) if "SITE" in data[0].columns else None

        c_counts = []
        for c in range(len(CROP_NAMES[model.config.DataConfig.dtype])):
            c_counts.append(len(np.argwhere(c == cultivar_data).flatten()))

        c_inds = (
            np.argsort(c_counts)[::-1][: model.config.DataConfig.prim_cultivars :]
            if model.config.DataConfig.prim_cultivars
            else np.array([])
        )
        if model.config.DataConfig.synth_data is None:
            for r in range(len(REGIONS)):
                for s in range(len(STATIONS)):
                    for si in range(len(SITES)):
                        for c in range(len(CROP_NAMES[model.config.DataConfig.dtype])):
                            cultivar_inds = np.argwhere(
                                (c == cultivar_data) & (r == region_data) & (s == station_data) & (si == site_data)
                            ).flatten()
                            if len(cultivar_inds) < 2:
                                continue
                            if model.config.DataConfig.withold is not None:
                                if withold_checker(model, model.config.DataConfig.withold, c, si, s, r):
                                    print(
                                        f"Witholding: {REGIONS[r]}, {STATIONS[s]}, {SITES[si]}, {CROP_NAMES[model.config.DataConfig.dtype][c]}"
                                    )
                                    continue

                            np.random.shuffle(cultivar_inds)
                            test_inds = np.concatenate((test_inds, cultivar_inds[:x])).astype(np.int32)

                            if model.config.DataConfig.data_cap is None or c in c_inds:
                                train_inds = np.concatenate((train_inds, cultivar_inds[x:][:])).astype(np.int32)
                            elif len(cultivar_inds[x:]) <= model.config.DataConfig.data_cap:
                                train_inds = np.concatenate((train_inds, cultivar_inds[x:][:])).astype(np.int32)
                            else:
                                train_inds = np.concatenate(
                                    (train_inds, cultivar_inds[x:][: model.config.DataConfig.data_cap])
                                ).astype(np.int32)

        np.random.shuffle(train_inds)
        np.random.shuffle(test_inds)
        model.data = {
            "train": [model.input_data[i] for i in train_inds],
            "test": [model.input_data[i] for i in test_inds],
        }
        model.val = {
            "train": [model.output_data[i] for i in train_inds],
            "test": [model.output_data[i] for i in test_inds],
        }

        cultivar_data = (
            torch.tensor(cultivar_data).to(torch.float32).to(model.device).unsqueeze(1)
            if cultivar_data is not None
            else None
        )
        region_data = (
            torch.tensor(region_data).to(torch.float32).to(model.device).unsqueeze(1)
            if region_data is not None
            else None
        )
        station_data = (
            torch.tensor(station_data).to(torch.float32).to(model.device).unsqueeze(1)
            if station_data is not None
            else None
        )
        site_data = (
            torch.tensor(site_data).to(torch.float32).to(model.device).unsqueeze(1) if site_data is not None else None
        )

        model.num_cultivars = len(torch.unique(cultivar_data)) if cultivar_data is not None else 1
        model.num_regions = len(torch.unique(region_data)) if region_data is not None else 1
        model.num_stations = len(torch.unique(station_data)) if station_data is not None else 1
        model.num_sites = len(torch.unique(site_data)) if site_data is not None else 1

        model.cultivars = {
            "train": torch.stack([cultivar_data[i] for i in train_inds]).to(torch.float32),
            "test": torch.stack([cultivar_data[i] for i in test_inds]).to(torch.float32),
        }
        model.regions = (
            {
                "train": torch.stack([region_data[i] for i in train_inds]).to(torch.float32),
                "test": torch.stack([region_data[i] for i in test_inds]).to(torch.float32),
            }
            if region_data is not None
            else None
        )
        model.stations = (
            {
                "train": torch.stack([station_data[i] for i in train_inds]).to(torch.float32),
                "test": torch.stack([station_data[i] for i in test_inds]).to(torch.float32),
            }
            if station_data is not None
            else None
        )
        model.sites = (
            {
                "train": torch.stack([site_data[i] for i in train_inds]).to(torch.float32),
                "test": torch.stack([site_data[i] for i in test_inds]).to(torch.float32),
            }
            if site_data is not None
            else None
        )

        model.all_inds = {
            "train": torch.cat(
                (model.regions["train"], model.stations["train"], model.sites["train"], model.cultivars["train"]), dim=1
            ),
            "test": torch.cat(
                (model.regions["test"], model.stations["test"], model.sites["test"], model.cultivars["test"]), dim=1
            ),
        }

        model.all_ids = {
            "train": torch.unique(
                torch.cat(
                    (model.regions["train"], model.stations["train"], model.sites["train"], model.cultivars["train"]),
                    dim=1,
                ),
                dim=0,
            ),
            "test": torch.unique(
                torch.cat(
                    (model.regions["test"], model.stations["test"], model.sites["test"], model.cultivars["test"]), dim=1
                ),
                dim=0,
            ),
        }

        if len(model.data["test"]) < 1:
            raise Exception("Insuffient per-cultivar data to build test set")
