"""
ParamRNN.py

Contains classes for running Param RNN models. We assume that
the model predicts parameters for a physical model.

Written by Will Solow, 2025
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from omegaconf import OmegaConf, DictConfig

from model_engine.engine import get_engine

from train_algs.base.DMC_Base import (
    FCGRU,
    EmbeddingFCGRU,
    GHCNDeepEmbeddingGRU,
    OneHotEmbeddingFCGRU,
    FFTempResponse,
    ParamModel,
    EmbeddingFCFF,
    DeepEmbeddingGRU
)
from train_algs.base.base import BaseModel
from model_engine.util import per_task_param_loader

from train_algs.base.util import (
    setup_logging,
    cumulative_error,
    get_grad_norm,
    log_training,
)


class BaseRNN(BaseModel):

    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:

        super().__init__(config, data)

        self.nn = BaseRNN.make_rnn(self, config).to(self.device)
        (
            self.nn.rnn.flatten_parameters()
            if hasattr(self.nn, "rnn") and hasattr(self.nn.rnn, "flatten_parameters")
            else None
        )
        self.make_optimizer(self.nn)

    @staticmethod
    def make_rnn(model: nn.Module, config: DictConfig) -> nn.Module:
        """Make the RNN"""

        # Create RNN Model
        if config.DConfig.arch == "FCGRU" or config.DConfig.arch == "MultiHeadFCGRU":
            nn = FCGRU(config, model)
        elif config.DConfig.arch == "EmbedFCGRU":
            nn = EmbeddingFCGRU(config, model)
        elif config.DConfig.arch == "DeepEmbedGRU":
            nn = DeepEmbeddingGRU(config, model)
        elif config.DConfig.arch == "GHCNDeepEmbedFCGRU":
            nn = GHCNDeepEmbeddingGRU(config, model)
        elif config.DConfig.arch == "OneHotEmbedFCGRU":
            nn = OneHotEmbeddingFCGRU(config, model)
        elif config.DConfig.arch == "EmbedFCFF":
            nn = EmbeddingFCFF(config, model)
        elif config.DConfig.arch == "FFTempResponse":
            nn = FFTempResponse(config, model)
        elif config.DConfig.arch == "ParamModel":
            nn = ParamModel(config, model)
        else:
            raise Exception(f"Unrecognized Model Architecture `{config.DConfig.arch}`")

        return nn

    def optimize(self) -> None:

        writer, run_name, log_path = setup_logging(self.config)
        os.makedirs(log_path, exist_ok=True)

        with open(f"{log_path}/config.yaml", "w", encoding="utf-8") as fp:
            OmegaConf.save(config=self.config, f=fp.name)
        fp.close()

        self.best_cum_rmse = float("inf")
        self.best_eval_loss = float("inf")
        self.best_rmse = torch.zeros(size=(4,)).to(self.device)

        for param in self.nn.parameters():
            param.requires_grad = True
        self.nn.train()

        train_name = "train"
        test_name = "val" if self.config.val_set else "test"
        for epoch in range(self.epochs):

            train_loss = 0
            grad = 0
            inds = np.arange(len(self.data[train_name]))
            np.random.shuffle(inds)
            data_shuffled = self.data[train_name][inds]
            val_shuffled = self.val[train_name][inds]
            dates_shuffled = self.dates[train_name][inds]
            cultivars_shuffled = self.cultivars[train_name][inds] if self.cultivars is not None else None
            regions_shuffled = self.regions[train_name][inds] if self.regions is not None else None
            stations_shuffled = self.stations[train_name][inds] if self.stations is not None else None
            sites_shuffled = self.sites[train_name][inds] if self.sites is not None else None
            train_avg = torch.zeros(size=(4,)).to(self.device)

            # Training
            for i in range(0, len(self.data[train_name]), self.batch_size):
                self.optimizer.zero_grad()

                batch_data = data_shuffled[i : i + self.batch_size]
                batch_dates = dates_shuffled[i : i + self.batch_size]
                batch_cultivars = (
                    cultivars_shuffled[i : i + self.batch_size] if cultivars_shuffled is not None else None
                )
                batch_regions = regions_shuffled[i : i + self.batch_size] if regions_shuffled is not None else None
                batch_stations = stations_shuffled[i : i + self.batch_size] if stations_shuffled is not None else None
                batch_sites = sites_shuffled[i : i + self.batch_size] if sites_shuffled is not None else None
                target = val_shuffled[i : i + self.batch_size]
                output, _, model_output = self.forward(
                    batch_data,
                    batch_dates,
                    cultivars=batch_cultivars,
                    regions=batch_regions,
                    stations=batch_stations,
                    sites=batch_sites,
                )

                # Compute PINN Loss
                if self.config.DConfig.type == "PINN":
                    if "CrossEntropyLoss" in self.config.DConfig.loss_func:
                        output_loss = self.loss_func(
                            output.view(-1, output.size(-1)),
                            target.nan_to_num(nan=0.0).squeeze().view(-1).to(torch.long),
                        )
                        output_loss = output_loss.reshape(output.size(0), -1).unsqueeze(-1)

                        model_loss = self.loss_func_mse(model_output, target.nan_to_num(nan=0.0))
                    else:
                        output_loss = self.loss_func(output, target.nan_to_num(nan=0.0))
                        model_loss = self.loss_func(model_output, target.nan_to_num(nan=0.0))

                    mask = ~torch.isnan(target)
                    output_loss = (output_loss * mask).sum() / mask.sum()
                    model_loss = (model_loss * mask).sum() / mask.sum()

                    loss = (1 - self.p) * output_loss + self.p * model_loss
                else:
                    # Handle Cross Entropy Loss by flattening batch
                    if "CrossEntropyLoss" in self.config.DConfig.loss_func:
                        loss = self.loss_func(
                            output.view(-1, output.size(-1)),
                            target.nan_to_num(nan=0.0).squeeze().view(-1).to(torch.long),
                        )
                        loss = loss.reshape(output.size(0), -1).unsqueeze(-1)
                    else:
                        loss = self.loss_func(output, target.nan_to_num(nan=0.0))

                    mask = ~torch.isnan(target)
                    loss = (loss * mask).sum() / mask.sum()
                    loss.backward()

                self.optimizer.step()
                train_loss += loss.item()
                grad += get_grad_norm(self.nn)

                avg_ = cumulative_error(target, output, mask, device=self.device)
                train_avg[:3] += avg_[1:-1]
                train_avg[-1] += torch.sum(avg_[1:-1]).to(self.device)

            # Evaluation
            eval_loss = 0
            eval_avg = torch.zeros(size=(4,)).to(self.device)

            for j in range(0, len(self.data[test_name]), self.batch_size):
                self.optimizer.zero_grad()
                batch_data = self.data[test_name][j : j + self.batch_size]
                batch_dates = self.dates[test_name][j : j + self.batch_size]
                batch_cultivars = (
                    self.cultivars[test_name][j : j + self.batch_size] if self.cultivars is not None else None
                )
                batch_regions = self.regions[test_name][j : j + self.batch_size] if self.regions is not None else None
                batch_stations = (
                    self.stations[test_name][j : j + self.batch_size] if self.stations is not None else None
                )
                batch_sites = self.sites[test_name][j : j + self.batch_size] if self.sites is not None else None

                eval_target = self.val[test_name][j : j + self.batch_size]
                eval_output, _, eval_model_output = self.forward(
                    batch_data,
                    batch_dates,
                    cultivars=batch_cultivars,
                    regions=batch_regions,
                    stations=batch_stations,
                    sites=batch_sites,
                )

                # Compute PINN Loss
                if self.config.DConfig.type == "PINN":
                    if "CrossEntropyLoss" in self.config.DConfig.loss_func:
                        eval_output_loss = self.loss_func(
                            eval_output.view(-1, output.size(-1)),
                            eval_target.nan_to_num(nan=0.0).squeeze().view(-1).to(torch.long),
                        )
                        eval_output_loss = eval_output_loss.reshape(output.size(0), -1).unsqueeze(-1)

                        eval_model_loss = self.loss_func_mse(eval_model_output, eval_target.nan_to_num(nan=0.0))
                    else:
                        eval_output_loss = self.loss_func(eval_output, eval_target.nan_to_num(nan=0.0))
                        eval_model_loss = self.loss_func(eval_model_output, eval_target.nan_to_num(nan=0.0))

                    eval_mask = ~torch.isnan(eval_target)
                    eval_output_loss = (eval_output_loss * eval_mask).sum() / eval_mask.sum()
                    eval_model_loss = (eval_model_loss * eval_mask).sum() / eval_mask.sum()

                    eval_loss = (1 - self.p) * eval_output_loss + self.p * eval_model_loss
                else:
                    # Handle Cross Entropy Loss by flattening series
                    if "CrossEntropyLoss" in self.config.DConfig.loss_func:
                        eval_loss = self.loss_func(
                            eval_output.view(-1, eval_output.size(-1)),
                            eval_target.nan_to_num(nan=0.0).squeeze().view(-1).to(torch.long),
                        )
                        eval_loss = eval_loss.reshape(eval_output.size(0), -1).unsqueeze(-1)
                    else:
                        eval_loss = self.loss_func(eval_output, eval_target.nan_to_num(nan=0.0))
                    eval_mask = ~torch.isnan(eval_target)
                    eval_loss = (eval_loss * eval_mask).sum() / eval_mask.sum()
                    eval_loss += eval_loss.item()

                avg_ = cumulative_error(eval_target, eval_output, eval_mask, device=self.device)
                eval_avg[:3] += avg_[1:-1]
                eval_avg[-1] += torch.sum(avg_[1:-1]).to(self.device)

            log_training(
                self,
                writer,
                log_path,
                epoch,
                train_loss,
                eval_loss,
                train_avg,
                eval_avg,
                grad,
            )

        self.scheduler.step(float(eval_loss if self.config.val_set else train_loss))


class ParamRNN(BaseRNN):

    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:

        super(ParamRNN, self).__init__(config, data)

        self.model = get_engine(self.config)(
            num_models=self.batch_size,
            config=config.PConfig,
            inputprovider=self.input_data,
            device=self.device,
        )

        self.task_params = per_task_param_loader(config, self.params).to(self.device)

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

        super(TransformerParam, self).__init__(config, data)

        self.model = get_engine(self.config)(
            num_models=self.batch_size,
            config=config.PConfig,
            inputprovider=self.input_data,
            device=self.device,
        )

    def forward(
        self, data: torch.Tensor = None, dates: np.ndarray = None, cultivars: torch.Tensor = None, **kwargs
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

        super(NoObsParamRNN, self).__init__(config, data)

        self.model = get_engine(self.config)(
            num_models=self.batch_size,
            config=config.PConfig,
            inputprovider=self.input_data,
            device=self.device,
        )

    def forward(
        self, data: torch.Tensor = None, dates: np.ndarray = None, cultivars: torch.Tensor = None, **kwargs
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

        super(WindowParamRNN, self).__init__(config, data)

        self.model = get_engine(self.config)(
            num_models=self.batch_size,
            config=config.PConfig,
            inputprovider=self.input_data,
            device=self.device,
        )

    def forward(
        self, data: torch.Tensor = None, dates: np.ndarray = None, cultivars: torch.Tensor = None, **kwargs
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
            k = 0 if i - self.config.DConfig.window_size < 0 else i - self.config.DConfig.window_size

            params_predict, _ = self.nn(data[:, k : i + 1], hn_cn, cultivars)
            params_predict = self.param_cast(params_predict)
            self.model.set_model_params(params_predict, self.params)

            output = self.model.run(dates=dates[:, i], cultivars=cultivars)

            output_tens[:, i] = output
            param_tens[:, i] = params_predict

        return output_tens, param_tens, None


class DeepRNN(BaseRNN):

    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:
        super(DeepRNN, self).__init__(config, data)

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
        for i in range(dlen):
            output_predict, hn_cn = self.nn(
                data[:, i], hn_cn, cultivars=cultivars, regions=regions, stations=stations, sites=sites
            )
            output_tens[:, i] = output_predict

        return output_tens, None, None


class PINNRNN(BaseRNN):

    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:
        super(PINNRNN, self).__init__(config, data)
        self.model = get_engine(self.config)(
            num_models=self.batch_size,
            config=config.PConfig,
            inputprovider=self.input_data,
            device=self.device,
        )

        self.task_params = per_task_param_loader(config, self.params).to(self.device)

    def forward(
        self, data: torch.Tensor = None, dates: np.ndarray = None, cultivars: torch.Tensor = None, **kwargs
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
                torch.cat((model_output.view(model_output.shape[0], -1).detach(), data[:, i]), dim=-1),
                hn_cn,
                cultivars=cultivars,
            )
            model_output = self.model.run(dates=dates[:, i], cultivars=cultivars)

            output_tens[:, i] = output_predict
            model_output_tens[:, i] = model_output.detach()

        return output_tens, None, model_output_tens


class ResidualRNN(BaseRNN):

    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:
        super(ResidualRNN, self).__init__(config, data)
        self.model = get_engine(self.config)(
            num_models=self.batch_size,
            config=config.PConfig,
            inputprovider=self.input_data,
            device=self.device,
        )
        self.task_params = per_task_param_loader(config, self.params).to(self.device)

    def forward(
        self, data: torch.Tensor, dates: np.ndarray, cultivars: torch.Tensor = None, **kwargs
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

        super(StationaryModel, self).__init__(config, data)

        self.model = get_engine(self.config)(
            num_models=self.batch_size,
            config=config.PConfig,
            inputprovider=self.input_data,
            device=self.device,
        )

        self.task_params = per_task_param_loader(config, self.params).to(self.device)

    def forward(
        self, data: torch.Tensor, dates: np.ndarray, cultivars: torch.Tensor = None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        data, dates, cultivars, b_size, dlen = self.handle_data(data, dates, cultivars)

        output_tens, _, _ = self.setup_storage(b_size, dlen)
        batch_params = (
            self.task_params[cultivars.to(torch.long).squeeze()] if cultivars is not None else self.task_params[0]
        )

        self.model.set_model_params(batch_params, self.params)
        model_output = self.model.reset(b_size)

        for i in range(dlen):
            output = self.model.run(dates=dates[:, i], cultivars=cultivars)
            output_tens[:, i] = output

        return output_tens, None, None


class HybridModel(BaseRNN):

    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:

        super(HybridModel, self).__init__(config, data)

        self.model = get_engine(self.config)(
            num_models=self.batch_size,
            config=config.PConfig,
            inputprovider=self.input_data,
            device=self.device,
        )

        self.task_params = per_task_param_loader(config, self.params).to(self.device)

    def forward(
        self, data: torch.Tensor, dates: np.ndarray, cultivars: torch.Tensor = None, **kwargs
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
