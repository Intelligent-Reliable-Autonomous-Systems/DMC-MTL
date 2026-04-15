"""Base class for NYUS_2_2

Written by Will Solow, 2025"""

import numpy as np
import pandas as pd
import torch
from autogluon.tabular import TabularPredictor
from omegaconf import DictConfig

from train_algs.base.base import BaseModel


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
