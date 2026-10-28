"""
rnn.py

Shared BaseRNN: architecture factory (make_rnn) and training loop (optimize).

Written by Will Solow, 2025
"""

import os
import time

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from train_algs.base.base import BaseModel
from train_algs.base.DMC_Base import (
    FCGRU,
    DeepEmbeddingGRU,
    DeepGRU,
    EmbeddingFCFF,
    EmbeddingFCGRU,
    EmbeddingTextGRU,
    EmbeddingVerboseTextGRU,
    FFTempResponse,
    FoundationEmbeddingGRU,
    FoundationParamEmbeddingGRU,
    FoundationParamModel,
    GHCNDeepEmbeddingGRU,
    OneHotEmbeddingFCGRU,
    ParamModel,
)
from train_algs.base.util import (
    cumulative_error,
    get_grad_norm,
    log_training,
    setup_logging,
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
        elif config.DConfig.arch == "DeepGRU":
            nn = DeepGRU(config, model)
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
        elif config.DConfig.arch == "EmbedTextGRU":
            nn = EmbeddingTextGRU(config, model)
        elif config.DConfig.arch == "EmbedTextVerbGRU":
            nn = EmbeddingVerboseTextGRU(config, model)
        elif config.DConfig.arch == "FoundationGRU":
            nn = FoundationEmbeddingGRU(config, model)
        elif config.DConfig.arch == "FoundationParamGRU":
            nn = FoundationParamEmbeddingGRU(config, model)
        elif config.DConfig.arch == "FoundationParamModel":
            nn = FoundationParamModel(config, model)
        elif config.DConfig.arch == "StationaryModel":
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
        test_name = "val" if self.config.DataConfig.val_set else "test"
        for epoch in range(self.epochs):
            st = time.time()

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
            all_ids = self.all_ids[train_name]

            # Training
            train_transfer = 0 if epoch < self.config.DConfig.transfer_start else 1
            train_embed = True
            embed = train_embed and "FoundationEmbed" in self.config.DConfig.type and train_transfer == 1
            if train_transfer == 1:
                for name, param in self.nn.named_parameters():
                    if any(name.startswith(p) for p in ["transfer"]):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

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

                output, val_out, model_output = self.forward(
                    batch_data,
                    batch_dates,
                    cultivars=batch_cultivars,
                    regions=batch_regions,
                    stations=batch_stations,
                    sites=batch_sites,
                    all_ids=all_ids,
                    transfer_rate=train_transfer,
                    ch=target,
                    name=train_name,
                    train_embed=train_embed,
                )
                # Train the embedding for foundation models
                if embed:
                    target = val_out

                # Compute PINN Loss
                odim = self.get_output_dim(self.config)
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
                    elif "Gaussian" in self.config.DConfig.loss_func:
                        loss = self.loss_func(output[:, :, :odim], target.nan_to_num(nan=0.0), output[:, :, odim:])
                    elif "VELBO" in self.config.DConfig.loss_func:
                        # Filter both gp_output and target to valid points only
                        mask = ~torch.isnan(target.squeeze())
                        loss = -self.loss_func(output, target.nan_to_num(nan=0.0).squeeze()).mean()
                    else:
                        loss = self.loss_func(output, target.nan_to_num(nan=0.0))
                    if "VELBO" not in self.config.DConfig.loss_func:
                        mask = ~torch.isnan(target)
                        loss = (loss * mask).sum() / mask.sum()
                    loss.backward()
                (
                    self.embed_optimizer.step()
                    if train_transfer == 1 and hasattr(self, "embed_optimizer")
                    else self.optimizer.step()
                )

                train_loss += loss.item()
                grad += get_grad_norm(self.nn)
                if "VELBO" in self.config.DConfig.loss_func:
                    output = output.mean
                else:
                    output = output[:, :, :odim] if output.shape[-1] != odim else output
                avg_ = cumulative_error(target, output, mask, device=self.device, embed=embed)
                train_avg[:3] += avg_[1:-1]
                train_avg[-1] += torch.sum(avg_[1:-1]).to(self.device)
            train_loss = train_loss if i == 0 else train_loss / i

            # Evaluation
            test_loss = 0
            eval_avg = torch.zeros(size=(4,)).to(self.device)
            test_transfer = 0 if epoch < self.config.DConfig.transfer_start else 1
            with torch.inference_mode():
                for j in range(0, len(self.data[test_name]), self.batch_size):
                    self.optimizer.zero_grad()
                    batch_data = self.data[test_name][j : j + self.batch_size]
                    batch_dates = self.dates[test_name][j : j + self.batch_size]
                    batch_cultivars = (
                        self.cultivars[test_name][j : j + self.batch_size] if self.cultivars is not None else None
                    )
                    batch_regions = (
                        self.regions[test_name][j : j + self.batch_size] if self.regions is not None else None
                    )
                    batch_stations = (
                        self.stations[test_name][j : j + self.batch_size] if self.stations is not None else None
                    )
                    batch_sites = self.sites[test_name][j : j + self.batch_size] if self.sites is not None else None

                    eval_target = self.val[test_name][j : j + self.batch_size]
                    test_all_ids = self.all_ids[test_name]
                    eval_output, eval_val_out, eval_model_output = self.forward(
                        batch_data,
                        batch_dates,
                        cultivars=batch_cultivars,
                        regions=batch_regions,
                        stations=batch_stations,
                        sites=batch_sites,
                        all_ids=test_all_ids,
                        transfer_rate=test_transfer,
                        ch=eval_target,
                        name=test_name,
                        train_embed=train_embed,
                    )

                    # Train the embedding for foundation models
                    if embed:
                        eval_target = eval_val_out

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
                        elif "Gaussian" in self.config.DConfig.loss_func:
                            eval_loss = self.loss_func(
                                eval_output[:, :, :odim], eval_target.nan_to_num(nan=0.0), eval_output[:, :, odim:]
                            )
                        elif "VELBO" in self.config.DConfig.loss_func:
                            # Filter both gp_output and target to valid points only
                            eval_mask = ~torch.isnan(eval_target.squeeze())
                            eval_loss = -self.loss_func(eval_output, eval_target.nan_to_num(nan=0.0).squeeze()).mean()

                        else:
                            eval_loss = self.loss_func(eval_output, eval_target.nan_to_num(nan=0.0))
                        if "VELBO" not in self.config.DConfig.loss_func:
                            eval_mask = ~torch.isnan(eval_target)
                            eval_loss = (eval_loss * eval_mask).sum() / eval_mask.sum()
                    test_loss += eval_loss.item()

                    if "VELBO" in self.config.DConfig.loss_func:
                        eval_output = eval_output.mean
                    else:
                        eval_output = eval_output[:, :, :odim] if eval_output.shape[-1] != odim else eval_output
                    avg_ = cumulative_error(eval_target, eval_output, eval_mask, device=self.device, embed=embed)
                    eval_avg[:3] += avg_[1:-1]
                    eval_avg[-1] += torch.sum(avg_[1:-1]).to(self.device)
                eval_loss = eval_loss if j == 0 else eval_loss / j
            log_training(
                self,
                writer,
                log_path,
                epoch,
                train_loss,
                test_loss,
                train_avg,
                eval_avg,
                grad,
                st,
            )

        self.scheduler.step(float(eval_loss if self.config.DataConfig.val_set else train_loss))
