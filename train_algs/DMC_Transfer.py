"""
DMC_Transfer.py

Contains classes for running Param RNN models. We assume that
the model predicts parameters for a physical model.

Written by Will Solow, 2025
"""

import numpy as np
import pandas as pd
import torch
from model_engine.engine import get_engine
from model_engine.util import per_task_param_loader
from omegaconf import DictConfig

from train_algs.DMC import BaseRNN
from utils import CROP_NAMES


class DeepTransferRNN(BaseRNN):
    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:
        super(DeepTransferRNN, self).__init__(config, data)

    def forward(
        self,
        data: torch.Tensor = None,
        dates: np.ndarray = None,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        transfer_rate: float = 0.0,
        all_ids: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data, dates, cultivars, b_size, dlen = self.handle_data(data, dates, cultivars)
        attn_map = None
        orig_pred = None
        output_tens, _, _ = self.setup_storage(b_size, dlen)

        self.nn.zero_grad()
        hn = self.nn.get_init_state(batch_size=b_size)

        transfer_inds = torch.rand(data.shape[0]) < transfer_rate
        if (~transfer_inds).sum() != 0:
            output_predict, _ = self.nn(
                data[~transfer_inds],
                cultivars=cultivars[~transfer_inds],
                regions=regions[~transfer_inds],
                stations=stations[~transfer_inds],
                sites=sites[~transfer_inds],
                hn=hn[:, ~transfer_inds],
            )
            output_tens[~transfer_inds, :] = output_predict

        if transfer_inds.sum() != 0:
            transfer_predict, attn_map, orig_pred = self.nn.forward_transfer(
                data[transfer_inds],
                hn=hn[:, transfer_inds],
                cultivars=cultivars[transfer_inds],
                regions=regions[transfer_inds],
                stations=stations[transfer_inds],
                sites=sites[transfer_inds],
                all_ids=all_ids,
            )
            output_tens[transfer_inds] = transfer_predict

        return output_tens, attn_map, orig_pred

    def make_optimizer(self, model):
        super().make_optimizer(model)

        self.embed_optimizer = torch.optim.Adam(
            [
                {"params": model.transfer_weather_rnn.parameters()},
                {"params": model.transfer_attn_fc.parameters()},
            ],
            lr=self.learning_rate,
        )


class DeepAdaptRNN(BaseRNN):
    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:
        super(DeepAdaptRNN, self).__init__(config, data)

    def forward(
        self,
        data: torch.Tensor = None,
        dates: np.ndarray = None,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        transfer_rate: float = 0.0,
        ch: torch.Tensor = None,
        name: str = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data, dates, cultivars, b_size, dlen = self.handle_data(data, dates, cultivars)
        attn_map = None
        orig_pred = None
        output_tens, _, _ = self.setup_storage(b_size, dlen)

        self.nn.zero_grad()
        hn = self.nn.get_init_state(batch_size=b_size)

        transfer_inds = torch.rand(data.shape[0]) < transfer_rate
        if (~transfer_inds).sum() != 0:
            output_predict, _ = self.nn(
                data[~transfer_inds],
                cultivars=cultivars[~transfer_inds],
                regions=regions[~transfer_inds],
                stations=stations[~transfer_inds],
                sites=sites[~transfer_inds],
                hn=hn[:, ~transfer_inds],
            )
            output_tens[~transfer_inds, :] = output_predict

        if transfer_inds.sum() != 0:
            b_all_inds = torch.cat((regions, stations, sites, cultivars), dim=1)[transfer_inds]
            all_inds = self.all_inds["train"]
            adapt_inds = []
            map_idx = []
            for i, ind in enumerate(b_all_inds):
                ids = torch.argwhere((ind == all_inds).all(axis=1)).flatten()
                """idx = ids[
                    torch.randperm(ids.shape[0])[
                        : torch.randint(
                            1, min(ids.shape[0] + 1, self.config.DConfig.num_transfer_years + 1), (1,)
                        ).item()
                    ]
                ]"""
                idx = ids[
                    torch.randperm(ids.shape[0])[: min(ids.shape[0] + 1, self.config.DConfig.num_transfer_years + 1)]
                ]
                adapt_inds.append(idx)
                map_idx.append(torch.ones_like(idx, device=idx.device) * i)
            adapt_inds = torch.cat(adapt_inds)
            map_idx = torch.cat(map_idx)
            embed = self.nn.compute_embed(
                self.data["train"][adapt_inds],
                ch=self.val["train"][adapt_inds],
                map_idx=map_idx,
                bsize=(transfer_inds).sum().item(),
            )
            transfer_predict, hn = self.nn.forward_embed(data[transfer_inds], embed=embed, hn=hn)
            output_tens[transfer_inds, :] = transfer_predict

        return output_tens, attn_map, orig_pred

    def make_optimizer(self, model):
        super().make_optimizer(model)

        embed_param_groups = [
            {"params": model.transfer_weather_rnn.parameters()} if hasattr(model, "transfer_weather_rnn") else None,
            {"params": model.transfer_rnn.parameters()} if hasattr(model, "transfer_rnn") else None,
            {"params": model.transfer_ch_rnn.parameters()} if hasattr(model, "transfer_ch_rnn") else None,
            {"params": model.transfer_pool_attn.parameters()} if hasattr(model, "transfer_pool_attn") else None,
            {"params": model.transfer_embed_attn.parameters()} if hasattr(model, "transfer_embed_attn") else None,
        ]

        self.embed_optimizer = torch.optim.Adam(
            [pg for pg in embed_param_groups if pg is not None],
            lr=self.learning_rate,
        )


class ParamTransferRNN(BaseRNN):
    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:
        super(ParamTransferRNN, self).__init__(config, data)

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

    def make_optimizer(self, model):
        super().make_optimizer(model)

        self.embed_optimizer = torch.optim.Adam(
            [
                {"params": model.transfer_weather_rnn.parameters()},
                {"params": model.transfer_attn_fc.parameters()},
            ],
            lr=self.learning_rate,
        )


class DeepFoundationRNN(BaseRNN):
    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:
        super(DeepFoundationRNN, self).__init__(config, data)

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
        transfer_rate: float = 0.0,
        ch: torch.Tensor = None,
        name: str = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data, dates, cultivars, b_size, dlen = self.handle_data(data, dates, cultivars)
        attn_map = None
        orig_pred = None
        output_tens, _, _ = self.setup_storage(b_size, dlen)

        self.nn.zero_grad()
        hn = self.nn.get_init_state(batch_size=b_size)

        # map_idx = torch.randint(0, 2, size=cultivars.shape)
        map_idx = torch.zeros_like(cultivars, device=self.device)

        transfer_inds = torch.rand(data.shape[0]) < transfer_rate
        if (~transfer_inds).sum() != 0:
            output_predict, _ = self.nn(
                data[~transfer_inds],
                cultivars=cultivars[~transfer_inds],
                regions=regions[~transfer_inds],
                stations=stations[~transfer_inds],
                sites=sites[~transfer_inds],
                hn=hn[:, ~transfer_inds],
                text_embedder=self.text_model,
                map_idx=map_idx[~transfer_inds],
            )
            output_tens[~transfer_inds, :] = output_predict

        if transfer_inds.sum() != 0:
            b_all_inds = torch.cat((regions, stations, sites, cultivars), dim=1)[transfer_inds]
            all_inds = self.all_inds["train"]
            adapt_inds = []
            map_idx = []
            for i, ind in enumerate(b_all_inds):
                ids = torch.argwhere((ind == all_inds).all(axis=1)).flatten()
                """idx = ids[
                    torch.randperm(ids.shape[0])[
                        : torch.randint(
                            1, min(ids.shape[0] + 1, self.config.DConfig.num_transfer_years + 1), (1,)
                        ).item()
                    ]
                ]"""
                idx = ids[
                    torch.randperm(ids.shape[0])[: min(ids.shape[0] + 1, self.config.DConfig.num_transfer_years + 1)]
                ]
                adapt_inds.append(idx)
                map_idx.append(torch.ones_like(idx, device=idx.device) * i)
            adapt_inds = torch.cat(adapt_inds)
            map_idx = torch.cat(map_idx)
            embed = self.nn.compute_embed(
                self.data["train"][adapt_inds],
                ch=self.val["train"][adapt_inds],
                map_idx=map_idx,
                bsize=(transfer_inds).sum().item(),
            )
            transfer_predict, hn = self.nn.forward_embed(data[transfer_inds], embed=embed, hn=hn)
            output_tens[transfer_inds, :] = transfer_predict

        return output_tens, attn_map, orig_pred

    def make_optimizer(self, model):
        super().make_optimizer(model)

        embed_param_groups = [
            {"params": model.transfer_weather_rnn.parameters()} if hasattr(model, "transfer_weather_rnn") else None,
            {"params": model.transfer_rnn.parameters()} if hasattr(model, "transfer_rnn") else None,
            {"params": model.transfer_ch_rnn.parameters()} if hasattr(model, "transfer_ch_rnn") else None,
            {"params": model.transfer_pool_attn.parameters()} if hasattr(model, "transfer_pool_attn") else None,
            {"params": model.transfer_embed_attn.parameters()} if hasattr(model, "transfer_embed_attn") else None,
        ]

        if all(p is not None for p in embed_param_groups):
            self.embed_optimizer = torch.optim.Adam(
                [pg for pg in embed_param_groups if pg is not None],
                lr=self.learning_rate,
            )


class ParamFoundationRNN(BaseRNN):
    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:
        super(ParamFoundationRNN, self).__init__(config, data)

        from sentence_transformers import SentenceTransformer

        self.text_model = SentenceTransformer("intfloat/e5-large-v2", device=self.device)

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
        transfer_rate: float = 0.0,
        ch: torch.Tensor = None,
        name: str = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data, dates, cultivars, b_size, dlen = self.handle_data(data, dates, cultivars)
        output_tens, param_tens, _ = self.setup_storage(b_size, dlen)

        batch_params = (
            self.task_params[cultivars.to(torch.long).squeeze()] if cultivars is not None else self.task_params[0]
        )
        self.model.set_model_params(batch_params, self.params)
        output = self.model.reset(b_size)

        self.nn.zero_grad()
        hn = self.nn.get_init_state(batch_size=b_size)

        map_idx = torch.randint(0, 2, size=cultivars.shape)
        # map_idx = torch.zeros_like(cultivars, device=self.device)

        transfer_inds = torch.rand(data.shape[0]) < transfer_rate

        b_all_inds = torch.cat((regions, stations, sites, cultivars), dim=1)[transfer_inds]
        if transfer_inds.sum() != 0:
            all_inds = self.all_inds["train"]
            adapt_inds = []
            map_idx = []
            for i, ind in enumerate(b_all_inds):
                ids = torch.argwhere((ind == all_inds).all(axis=1)).flatten()
                idx = ids[
                    torch.randperm(ids.shape[0])[: min(ids.shape[0] + 1, self.config.DConfig.num_transfer_years + 1)]
                ]
                adapt_inds.append(idx)
                map_idx.append(torch.ones_like(idx, device=idx.device) * i)
            adapt_inds = torch.cat(adapt_inds)
            map_idx = torch.cat(map_idx)
            embed = self.nn.compute_embed(
                self.data["train"][adapt_inds],
                ch=self.val["train"][adapt_inds],
                map_idx=map_idx,
                bsize=(transfer_inds).sum().item(),
            )
        if (~transfer_inds).sum() != 0:
            reg_embed, stat_embed, site_embed, cult_embed = self.nn.get_embeds(
                cultivars[~transfer_inds],
                regions[~transfer_inds],
                stations[~transfer_inds],
                sites[~transfer_inds],
                self.text_model,
                map_idx[~transfer_inds],
            )
        for i in range(dlen):
            if (~transfer_inds).sum() != 0:
                params_predict, hn_nt = self.nn.forward_with_embed(
                    torch.cat(
                        (output.view(output.shape[0], -1).detach()[~transfer_inds], data[~transfer_inds, i]), dim=-1
                    ),
                    hn=hn[:, ~transfer_inds],
                    reg_embed=reg_embed,
                    stat_embed=stat_embed,
                    site_embed=site_embed,
                    cult_embed=cult_embed,
                )

                params_predict = self.param_cast(params_predict, prev_params=batch_params)
                # param_tens[~transfer_inds, i] = params_predict.squeeze().clone()
                param_tens[:, i] = params_predict.squeeze().clone()
                # hn[:, ~transfer_inds] = hn_nt
                hn = hn_nt

            if transfer_inds.sum() != 0:
                transfer_params_predict, hn_t = self.nn.forward_embed(data[transfer_inds, i], embed=embed, hn=hn)
                param_tens[transfer_inds, i] = transfer_params_predict.squeeze().clone()
                hn[:, transfer_inds] = hn_t

            # self.model.set_model_params(param_tens[:, i], self.params)
            self.model.set_model_params(params_predict.squeeze(), self.params)
            output = self.model.run(dates=dates[:, i], cultivars=cultivars)
            output_tens[:, i] = output

        return output_tens, param_tens, None

    def make_optimizer(self, model):
        super().make_optimizer(model)

        embed_param_groups = [
            {"params": model.transfer_weather_rnn.parameters()} if hasattr(model, "transfer_weather_rnn") else None,
            {"params": model.transfer_rnn.parameters()} if hasattr(model, "transfer_rnn") else None,
            {"params": model.transfer_ch_rnn.parameters()} if hasattr(model, "transfer_ch_rnn") else None,
            {"params": model.transfer_pool_attn.parameters()} if hasattr(model, "transfer_pool_attn") else None,
            {"params": model.transfer_embed_attn.parameters()} if hasattr(model, "transfer_embed_attn") else None,
        ]

        if all(p is not None for p in embed_param_groups):
            self.embed_optimizer = torch.optim.Adam(
                [pg for pg in embed_param_groups if pg is not None],
                lr=self.learning_rate,
            )
