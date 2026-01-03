"""
FineTuner.py

Contains classes for running Finetuning models. We assume that
the model predicts parameters for a physical model and is then
finetuned by another model.

Written by Will Solow, 2025
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from omegaconf import OmegaConf, DictConfig
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from train_algs.base.util import ScaledMSELoss


from model_engine.engine import get_engine
from train_algs.base.util import (
    setup_logging,
    cumulative_error,
    get_grad_norm,
    log_training,
)

from model_engine.util import per_task_param_loader
from train_algs.DMC import BaseRNN
from train_algs.base.base import BaseModel
from train_algs.base.FineTuner_Base import (
    ErrorFCTuner,
    EmbedErrorFCTuner,
)
from train_algs.base.util import load_config, load_model_fpath


class BaseFineTuner(BaseModel):

    def __init__(self, config: DictConfig, data: list[pd.DataFrame], rnn_fpath: str = None) -> None:
        """
        Initialize a LSTM Model
        """
        super().__init__(config, data)

        self.finetuner = BaseFineTuner.make_finetuner(self, config).to(self.device)

        if rnn_fpath is not None:
            try:
                self.nn = BaseRNN.make_rnn(self, load_config(rnn_fpath)).to(self.device)
                self.load_rnn_model(f"{load_model_fpath(rnn_fpath, name="rnn_model_best.pt")}")
            except:
                self.nn = BaseRNN.make_rnn(self, config).to(self.device)
                print("Failure in model loading. Loading untrained model...")
        else:
            self.nn = BaseRNN.make_rnn(self, config).to(self.device)

        self.make_optimizer(self.finetuner)

    def load_rnn_model(self, path: str) -> None:
        """
        Load LSTM Model
        """
        self.nn.load_state_dict(torch.load(path, weights_only=True, map_location=self.device), strict=False)

    def save_model(self, path: str, name: str = "rnn_model_best.pt") -> None:
        """
        Save model
        """
        torch.save(self.finetuner.state_dict(), f"{path}/finetune_model.pt")
        torch.save(self.nn.state_dict(), f"{path}/{name}")

    def load_model(self, path: str, name: str = "rnn_model_best.pt") -> None:
        """
        Load Fine Tuner Model
        """
        self.finetuner.load_state_dict(
            torch.load(f"{path}/finetune_model.pt", weights_only=True, map_location=self.device)
        )
        self.nn.load_state_dict(torch.load(f"{path}/{name}", weights_only=True, map_location=self.device))

    @staticmethod
    def make_finetuner(model: nn.Module, config: DictConfig) -> nn.Module:
        """
        Make the Finetuner"""
        if config.RTMCConfig.arch == "FCGRU":
            finetuner = ErrorFCTuner(config, model)
        elif config.RTMCConfig.arch == "EmbedFCGRU":
            finetuner = EmbedErrorFCTuner(config, model)
        else:
            raise Exception(f"Unrecognized Model Architecture `{config.RTMCConfig.arch}`")
        return finetuner

    def make_optimizer(self, model: nn.Module) -> None:
        """Make the optimizer"""

        self.learning_rate = self.config.RTMCConfig.learning_rate
        self.batch_size = self.config.RTMCConfig.batch_size
        self.epochs = self.config.RTMCConfig.epochs

        if self.config.RTMCConfig.loss_func == "MSELoss":
            self.loss_func = nn.MSELoss(reduction="none")
        elif self.config.RTMCConfig.loss_func == "CrossEntropyLoss":
            self.loss_func = nn.CrossEntropyLoss(reduction="none")
        elif self.config.RTMCConfig.loss_func == "ScaledMSELoss":
            self.loss_func = ScaledMSELoss(weight=self.config.RTMCConfig.scale)
        else:
            msg = f"Unimplemented RTMCConfig Loss Function {self.config.RTMCConfig.loss_func}"
            raise NotImplementedError(msg)

        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.DConfig.lr_factor,
            patience=10,
            cooldown=10,
            threshold=1e-6,
        )

    def optimize(self) -> None:

        writer, run_name, log_path = setup_logging(self.config)
        os.makedirs(log_path, exist_ok=True)

        with open(f"{log_path}/config.yaml", "w", encoding="utf-8") as fp:
            OmegaConf.save(config=self.config, f=fp.name)
        fp.close()
        # torch.save(self.drange, f"{log_path}/model_drange.pt")

        self.best_cum_rmse = float("inf")
        self.best_eval_loss = float("inf")
        self.best_rmse = torch.zeros(size=(4,)).to(self.device)

        for param in self.nn.parameters():
            param.requires_grad = False

        for param in self.finetuner.parameters():
            param.requires_grad = True

        self.nn.eval()
        self.finetuner.train()
        train_name = "val" if self.config.DataConfig.val_set else "train"
        test_name = "test"
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
                days = (
                    torch.randint(0, self.max_dlen, size=(self.batch_size, 1), device=self.device)
                    if self.config.RTMCConfig.day_masking is not None
                    else None
                )

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

                output, _, _ = self.forward(
                    batch_data,
                    batch_dates,
                    batch_cultivars,
                    regions=batch_regions,
                    stations=batch_stations,
                    sites=batch_sites,
                    val_data=target,
                    days=days,
                )

                loss = self.loss_func(output, target.nan_to_num(nan=0.0))
                mask = ~torch.isnan(target)
                loss = (loss * mask).sum() / mask.sum()

                for name, param in self.finetuner.named_parameters():
                    if torch.isnan(param).any():
                        pass
                    if param.grad is not None and torch.isnan(param.grad).any():
                        pass

                loss.backward()

                self.optimizer.step()
                train_loss += loss.item()
                grad += get_grad_norm(self.finetuner)

                avg_ = cumulative_error(target, output, mask, device=self.device)
                train_avg[:3] += avg_[1:-1]
                train_avg[-1] += torch.sum(avg_[1:-1]).to(self.device)

            # Evaluation
            test_loss = 0
            test_avg = torch.zeros(size=(4,)).to(self.device)

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

                test_target = self.val[test_name][j : j + self.batch_size]
                test_output, _, _ = self.forward(
                    batch_data,
                    batch_dates,
                    cultivars=batch_cultivars,
                    regions=batch_regions,
                    stations=batch_stations,
                    sites=batch_sites,
                    val_data=test_target,
                )

                test_loss = self.loss_func(test_output, test_target.nan_to_num(nan=0.0))
                test_mask = ~torch.isnan(test_target)
                test_loss = (test_loss * test_mask).sum() / test_mask.sum()
                test_loss += test_loss.item()

                avg_ = cumulative_error(test_target, test_output, test_mask, device=self.device)
                test_avg[:3] += avg_[1:-1]
                test_avg[-1] += torch.sum(avg_[1:-1]).to(self.device)

            log_training(
                self,
                writer,
                log_path,
                epoch,
                train_loss,
                test_loss,
                train_avg,
                test_avg,
                grad,
            )

            self.scheduler.step(float(train_loss))

    def compute_error(
        self, i: int, b_size: int, output: torch.Tensor, val_data: torch.Tensor, days: int | torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute the error
        """
        if days is not None:
            error = torch.nan_to_num(output - val_data, nan=0.0).detach()
            if isinstance(days, int):
                if i >= days:
                    error = torch.zeros((b_size, len(self.output_vars))).to(self.device)
            else:
                error = torch.where(i >= days[:b_size], 0, error)

        else:
            error = torch.nan_to_num(output - val_data, nan=0.0).detach()
        return error


class FineTuner(BaseFineTuner):

    def __init__(self, config: DictConfig, data: list[pd.DataFrame], rnn_fpath: str = None) -> None:

        super(FineTuner, self).__init__(config, data, rnn_fpath=rnn_fpath)

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
        val_data: torch.Tensor = None,
        days: int = None,
        **kwargs,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]:

        data, dates, cultivars, b_size, dlen = self.handle_data(data, dates, cultivars)

        output_tens, param_tens, _ = self.setup_storage(b_size, dlen)
        param_tens_err = torch.empty(size=(b_size, dlen, 2, len(self.params))).to(self.device)
        self.store_error = torch.zeros(size=(b_size, 1)).to(self.device)
        error_tens = torch.zeros(size=(b_size, dlen, len(self.output_vars))).to(self.device)

        self.nn.zero_grad()
        self.finetuner.zero_grad()

        hn_cn = self.nn.get_init_state(batch_size=b_size)
        hn_cn_ft = self.finetuner.get_init_state(batch_size=b_size)
        batch_params = (
            self.task_params[cultivars.to(torch.long).squeeze()] if cultivars is not None else self.task_params[0]
        )
        self.model.set_model_params(batch_params, self.params)
        output = self.model.reset(b_size)
        # Run through entirety of time series predicting parameters for physical model at each step
        for i in range(dlen):
            params_predict_orig, hn_cn = self.nn(
                input=torch.cat((output.view(output.shape[0], -1).detach(), data[:, i]), dim=-1),
                hn=hn_cn,
                cultivars=cultivars,
                regions=regions,
                stations=stations,
                sites=sites,
            )
            params_predict_ft, hn_cn_ft = self.finetuner(
                error=error_tens[:, i - 1].clone().detach(),
                hn=hn_cn_ft,
                cultivars=cultivars,
                delta_t=torch.ones(size=(b_size, 1), device=self.device),
            )
            params_predict = self.param_cast(params_predict_orig + params_predict_ft)

            self.model.set_model_params(params_predict, self.params)
            output = self.model.run(dates=dates[:, i], cultivars=cultivars)

            error_tens[:, i] = self.compute_error(i, b_size, output, val_data[:, i], days=days)

            output_tens[:, i] = output
            param_tens[:, i] = params_predict.detach()
            param_tens_err[:, i, 0] = params_predict_orig.detach()
            param_tens_err[:, i, 1] = params_predict_ft.detach()

        return output_tens, param_tens, param_tens_err


class DeepFineTuner(BaseFineTuner):

    def __init__(self, config: DictConfig, data: list[pd.DataFrame], rnn_fpath: str = None) -> None:

        super(DeepFineTuner, self).__init__(config, data, rnn_fpath=rnn_fpath)

    def forward(
        self,
        data: torch.Tensor,
        dates: np.ndarray,
        cultivars: torch.Tensor = None,
        val_data: torch.Tensor = None,
        days: int = None,
        **kwargs,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]:

        data, dates, cultivars, b_size, dlen = self.handle_data(data, dates, cultivars)

        output_tens, _, _ = self.setup_storage(b_size, dlen)
        self.store_error = torch.zeros(size=(b_size, 1)).to(self.device)
        error_tens = torch.zeros(size=(b_size, dlen, len(self.output_vars))).to(self.device)

        self.nn.zero_grad()
        self.finetuner.zero_grad()

        hn_cn = self.nn.get_init_state(batch_size=b_size)
        hn_cn_ft = self.finetuner.get_init_state(batch_size=b_size)

        # Run through entirety of time series predicting parameters for physical model at each step
        for i in range(dlen):
            output_predict, hn_cn = self.nn(
                input=data[:, i],
                hn=hn_cn,
                cultivars=cultivars,
            )
            output_predict_ft, hn_cn_ft = self.finetuner(
                error=error_tens[:, i - 1].clone().detach(),
                hn=hn_cn_ft,
                cultivars=cultivars,
                delta_t=torch.ones(size=(b_size, 1), device=self.device),
            )
            output = output_predict + output_predict_ft

            error_tens[:, i] = self.compute_error(i, b_size, output, val_data[:, i], days=days)

            output_tens[:, i] = output

        return output_tens, None, None
