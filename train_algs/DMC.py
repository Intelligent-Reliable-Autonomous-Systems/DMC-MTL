"""
DMC.py

Coupling classes for DMC calibrators (Param/Deep/PINN/…). Shared
training lives in train_algs.base.rnn.BaseRNN.

Written by Will Solow, 2025
"""

import numpy as np
import pandas as pd
import torch
from model_engine.engine import get_engine
from model_engine.util import per_task_param_loader
from omegaconf import DictConfig

from assets import CROP_NAMES
from train_algs.base.rnn import BaseRNN

# Re-export for callers that still import BaseRNN from DMC
__all__ = [
    "BaseRNN",
    "ParamRNN",
    "TransformerParam",
    "NoObsParamRNN",
    "WindowParamRNN",
    "DeepRNN",
    "PINNRNN",
    "ResidualRNN",
    "StationaryModel",
    "HybridModel",
    "DeepTextRNN",
]


class ParamRNN(BaseRNN):
    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:

        super().__init__(config, data)

        self.model = get_engine(self.config)(
            num_models=self.batch_size,
            config=config.PConfig,
            inputprovider=self.input_data,
            device=self.device,
        )

        self.task_params = per_task_param_loader(CROP_NAMES, config, self.params).to(self.device)

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
        self.nn.zero_grad()

        data, dates, cultivars, b_size, dlen = self.handle_data(data, dates, cultivars)

        output_tens, param_tens, _ = self.setup_storage(b_size, dlen)

        # Hidden state
        hn_cn = self.nn.get_init_state(batch_size=data.shape[0]) if hasattr(self.nn, "get_init_state") else None
        batch_params = (
            self.task_params[cultivars.to(torch.long).squeeze()] if cultivars is not None else self.task_params[0]
        )
        self.model.set_model_params(batch_params, self.params)
        output = self.model.reset(b_size)

        # Run through entirety of time series predicting params
        for i in range(dlen):
            params_predict, hn_cn = self.nn(
                torch.cat((output.view(output.shape[0], -1).detach(), data[:, i]), dim=-1),
                hn=hn_cn,
                cultivars=cultivars,
                regions=regions,
                stations=stations,
                sites=sites,
            )
            params_predict = self.param_cast(params_predict, prev_params=batch_params)
            batch_params = params_predict
            self.model.set_model_params(params_predict, self.params)
            output = self.model.run(dates=dates[:, i], cultivars=cultivars)

            output_tens[:, i] = output
            param_tens[:, i] = params_predict

        return output_tens, param_tens, None


class TransformerParam(BaseRNN):
    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:

        super().__init__(config, data)

        self.model = get_engine(self.config)(
            num_models=self.batch_size,
            config=config.PConfig,
            inputprovider=self.input_data,
            device=self.device,
        )

    def forward(
        self,
        data: torch.Tensor = None,
        dates: np.ndarray = None,
        cultivars: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass call for integrated gradients attribution sequence batch data
        """

        data, dates, cultivars, b_size, dlen = self.handle_data(data, dates, cultivars)

        output_tens, param_tens, _ = self.setup_storage(b_size, dlen)

        self.nn.zero_grad()
        output = self.model.reset(b_size)

        # Run through entirety of time series predicting parameters for physical model at each step
        params_predict, hn_cn = self.nn(input=data, cultivars=cultivars)
        params_predict = self.param_cast(params_predict)
        param_tens = params_predict

        for j in range(dlen):
            self.model.set_model_params(param_tens[:, j], self.params)
            output = self.model.run(dates=dates[:, j], cultivars=cultivars)

            output_tens[:, j] = output

        return output_tens, param_tens, None


class NoObsParamRNN(BaseRNN):
    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:

        super().__init__(config, data)

        self.model = get_engine(self.config)(
            num_models=self.batch_size,
            config=config.PConfig,
            inputprovider=self.input_data,
            device=self.device,
        )

    def forward(
        self,
        data: torch.Tensor = None,
        dates: np.ndarray = None,
        cultivars: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass call for integrated gradients attribution sequence batch data
        """
        data, dates, cultivars, b_size, dlen = self.handle_data(data, dates, cultivars)

        output_tens, param_tens, _ = self.setup_storage(b_size, dlen)

        self.nn.zero_grad()
        hn_cn = self.nn.get_init_state(batch_size=b_size)
        output = self.model.reset(b_size)
        # Run through entirety of time series predicting parameters for physical model at each step
        for i in range(dlen):
            params_predict, hn_cn = self.nn(data[:, i], hn_cn, cultivars)
            params_predict = self.param_cast(params_predict)
            self.model.set_model_params(params_predict, self.params)

            output = self.model.run(dates=dates[:, i], cultivars=cultivars)

            output_tens[:, i] = output
            param_tens[:, i] = params_predict

        return output_tens, param_tens, None


class WindowParamRNN(BaseRNN):
    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:

        super().__init__(config, data)

        self.model = get_engine(self.config)(
            num_models=self.batch_size,
            config=config.PConfig,
            inputprovider=self.input_data,
            device=self.device,
        )

    def forward(
        self,
        data: torch.Tensor = None,
        dates: np.ndarray = None,
        cultivars: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass call for integrated gradients attribution sequence batch data
        """
        data, dates, cultivars, b_size, dlen = self.handle_data(data, dates, cultivars)

        output_tens, param_tens, _ = self.setup_storage(b_size, dlen)

        self.nn.zero_grad()
        hn_cn = None
        output = self.model.reset(b_size)
        # Run through entirety of time series predicting parameters for physical model at each step
        for i in range(dlen):
            k = max(i - self.config.DConfig.window_size, 0)

            params_predict, _ = self.nn(data[:, k : i + 1], hn_cn, cultivars)
            params_predict = self.param_cast(params_predict)
            self.model.set_model_params(params_predict, self.params)

            output = self.model.run(dates=dates[:, i], cultivars=cultivars)

            output_tens[:, i] = output
            param_tens[:, i] = params_predict

        return output_tens, param_tens, None


class DeepRNN(BaseRNN):
    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:
        super().__init__(config, data)

    def forward(
        self,
        data: torch.Tensor = None,
        dates: np.ndarray = None,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        data, dates, cultivars, b_size, dlen = self.handle_data(data, dates, cultivars)

        if self.config.DConfig.noise:
            return self.forward_with_noise(
                data,
                dates,
                cultivars,
                regions,
                stations,
                sites,
                b_size=b_size,
                dlen=dlen,
                **kwargs,
            )

        output_tens, _, _ = self.setup_storage(b_size, dlen)

        self.nn.zero_grad()
        hn_cn = self.nn.get_init_state(batch_size=b_size)

        # Run through entirety of time series predicting output for each step
        output_predict, hn_cn = self.nn(
            data,
            hn_cn,
            cultivars=cultivars,
            regions=regions,
            stations=stations,
            sites=sites,
        )
        output_tens = output_predict

        return output_tens, None, None


class PINNRNN(BaseRNN):
    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:
        super().__init__(config, data)
        self.model = get_engine(self.config)(
            num_models=self.batch_size,
            config=config.PConfig,
            inputprovider=self.input_data,
            device=self.device,
        )

        self.task_params = per_task_param_loader(CROP_NAMES, config, self.params).to(self.device)

    def forward(
        self,
        data: torch.Tensor = None,
        dates: np.ndarray = None,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        data, dates, cultivars, b_size, dlen = self.handle_data(data, dates, cultivars)

        output_tens, _, model_output_tens = self.setup_storage(b_size, dlen)

        self.nn.zero_grad()
        hn_cn = self.nn.get_init_state(batch_size=b_size)

        batch_params = (
            self.task_params[cultivars.to(torch.long).squeeze()] if cultivars is not None else self.task_params[0]
        )
        self.model.set_model_params(batch_params, self.params)
        model_output = self.model.reset(b_size)

        # Run through entirety of time series predicting output for each step
        for i in range(dlen):
            output_predict, hn_cn = self.nn(
                torch.cat(
                    (model_output.view(model_output.shape[0], -1).detach(), data[:, i]),
                    dim=-1,
                ),
                hn_cn,
                cultivars=cultivars,
                regions=regions,
                stations=stations,
                sites=sites,
            )
            model_output = self.model.run(dates=dates[:, i], cultivars=cultivars)

            output_tens[:, i] = output_predict
            model_output_tens[:, i] = model_output.detach()

        return output_tens, None, model_output_tens


class ResidualRNN(BaseRNN):
    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:
        super().__init__(config, data)
        self.model = get_engine(self.config)(
            num_models=self.batch_size,
            config=config.PConfig,
            inputprovider=self.input_data,
            device=self.device,
        )
        self.task_params = per_task_param_loader(CROP_NAMES, config, self.params).to(self.device)

    def forward(
        self,
        data: torch.Tensor,
        dates: np.ndarray,
        cultivars: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        data, dates, cultivars, b_size, dlen = self.handle_data(data, dates, cultivars)

        output_tens, _, _ = self.setup_storage(b_size, dlen)

        self.nn.zero_grad()
        hn_cn = self.nn.get_init_state(batch_size=b_size)
        model_output = self.model.reset(b_size)

        # Set parameters for model given the batch
        batch_params = self.task_params[cultivars.to(torch.long).squeeze()]
        self.model.set_model_params(batch_params, self.params)

        # Run through entirety of time series predicting residual
        for i in range(dlen):
            residual_predict, hn_cn = self.nn(data[:, i], hn_cn, cultivars)
            model_output = self.model.run(dates=dates[:, i], cultivars=cultivars)
            output_tens[:, i] = model_output + residual_predict

        return output_tens, None, None


class StationaryModel(BaseRNN):
    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:

        super().__init__(config, data)

        self.model = get_engine(self.config)(
            num_models=self.batch_size,
            config=config.PConfig,
            inputprovider=self.input_data,
            device=self.device,
        )

        self.task_params = per_task_param_loader(CROP_NAMES, config, self.params).to(self.device)

    def _noise_weather_attrs(self) -> list[str]:
        provider = self.model.inputdataprovider
        return [a for a in ("TEMP", "TMIN", "TMAX") if a in getattr(provider, "attrs", [])]

    def forward_with_noise(
        self,
        data: torch.Tensor,
        dates: np.ndarray = None,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        b_size: int = None,
        dlen: int = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """K-day physical-model forecast under Gaussian noise on TEMP/TMIN/TMAX."""
        assert self.config.DConfig.forecast_noise is not None, "DConfig.forecast_noise must be set when noise=True"
        assert self.config.DConfig.forecast_k is not None, "DConfig.forecast_k must be set when noise=True"

        K = int(self.config.DConfig.forecast_k)
        noise_scale = float(self.config.DConfig.forecast_noise)
        if b_size is None or dlen is None:
            data, dates, cultivars, b_size, dlen = self.handle_data(data, dates, cultivars)
        output_tens, _, _ = self.setup_storage(b_size, dlen)

        batch_params = (
            self.task_params[cultivars.to(torch.long).squeeze()] if cultivars is not None else self.task_params[0]
        )
        self.model.set_model_params(batch_params, self.params)
        self.model.set_model_params(self.param_cast(self.nn.model_params), self.params)

        provider = self.model.inputdataprovider
        clean_values = provider.values.clone()
        noisy_values = clean_values.clone()
        for attr in self._noise_weather_attrs():
            idx = provider.attrs.index(attr)
            noisy_values[:, idx] = noisy_values[:, idx] + noise_scale * torch.randn_like(noisy_values[:, idx])

        try:
            for T in range(K, dlen):
                start = T - K
                self.model.set_model_params(batch_params, self.params)
                self.model.set_model_params(self.param_cast(self.nn.model_params), self.params)
                self.model.reset(b_size)
                provider.values = clean_values
                for i in range(start):
                    self.model.run(dates=dates[:, i], cultivars=cultivars)
                provider.values = noisy_values
                pred = None
                for s in range(K):
                    pred = self.model.run(dates=dates[:, start + s], cultivars=cultivars)
                output_tens[:, T] = pred
        finally:
            provider.values = clean_values

        return output_tens, None, None

    def forward(
        self,
        data: torch.Tensor,
        dates: np.ndarray,
        cultivars: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        data, dates, cultivars, b_size, dlen = self.handle_data(data, dates, cultivars)

        if self.config.DConfig.noise:
            return self.forward_with_noise(
                data,
                dates,
                cultivars,
                b_size=b_size,
                dlen=dlen,
                **kwargs,
            )

        output_tens, _, _ = self.setup_storage(b_size, dlen)
        batch_params = (
            self.task_params[cultivars.to(torch.long).squeeze()] if cultivars is not None else self.task_params[0]
        )

        self.model.set_model_params(batch_params, self.params)
        self.model.set_model_params(self.param_cast(self.nn.model_params), self.params)
        model_output = self.model.reset(b_size)

        for i in range(dlen):
            output = self.model.run(dates=dates[:, i], cultivars=cultivars)
            output_tens[:, i] = output

        return output_tens, None, None


class HybridModel(BaseRNN):
    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:

        super().__init__(config, data)

        self.model = get_engine(self.config)(
            num_models=self.batch_size,
            config=config.PConfig,
            inputprovider=self.input_data,
            device=self.device,
        )

        self.task_params = per_task_param_loader(CROP_NAMES, config, self.params).to(self.device)

    def forward(
        self,
        data: torch.Tensor,
        dates: np.ndarray,
        cultivars: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        data, dates, cultivars, b_size, dlen = self.handle_data(data, dates, cultivars)

        output_tens, _, _ = self.setup_storage(b_size, dlen)
        batch_params = (
            self.task_params[cultivars.to(torch.long).squeeze()] if cultivars is not None else self.task_params[0]
        )

        self.model.set_model_params(batch_params, self.params)
        self.model.set_model_params(self.param_cast(self.nn.model_params), self.params)
        output = self.model.reset(b_size)

        for i in range(dlen):
            temp_response, _ = self.nn(data[:, i])  # Will return None if not using hybrid model

            output = self.model.run(dates=dates[:, i], cultivars=cultivars, TRESP=temp_response)

            output_tens[:, i] = output

        return output_tens, None, None


class DeepTextRNN(BaseRNN):
    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:
        super().__init__(config, data)

        from sentence_transformers import SentenceTransformer

        self.text_model = SentenceTransformer("intfloat/e5-large-v2", device=self.device)

    def forward(
        self,
        data: torch.Tensor = None,
        dates: np.ndarray = None,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        data, dates, cultivars, b_size, dlen = self.handle_data(data, dates, cultivars)

        output_tens, _, _ = self.setup_storage(b_size, dlen)

        self.nn.zero_grad()
        hn_cn = self.nn.get_init_state(batch_size=b_size)

        # Run through entirety of time series predicting output for each step

        output_predict, hn_cn = self.nn(
            data,
            hn_cn,
            cultivars=cultivars,
            regions=regions,
            stations=stations,
            sites=sites,
            text_embedder=self.text_model,
        )
        output_tens = output_predict

        return output_tens, None, None
