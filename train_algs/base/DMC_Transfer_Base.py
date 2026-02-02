"""Base class for DMC Transfer

Written by Will Solow, 2025"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig
from train_algs.base.util import set_embedding_op
from model_engine.util import CROP_NAMES, REGIONS, STATIONS, SITES


class BaseModule(nn.Module):
    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super().__init__()
        self.hidden_dim = c.DConfig.hidden_dim
        self.dim2 = int(self.hidden_dim / 2)
        self.input_dim = model.get_input_dim(c)
        self.output_dim = model.get_output_dim(c)
        self.embed_dim = set_embedding_op(self, c.DConfig.embed_op)

        self.embed_op_str = c.DConfig.embed_op
        self.synth_data = c.DataConfig.synth_data

        cult_orig = torch.unique(torch.concatenate(list(model.cultivars.values()), axis=0)).to(torch.int)
        self.cult_mapping = torch.zeros((int(cult_orig.max()) + 1,)).to(torch.int).to(model.device)
        self.cult_mapping[cult_orig] = torch.arange(len(cult_orig)).to(torch.int).to(model.device)

        if self.synth_data is None:
            reg_orig = torch.unique(torch.concatenate(list(model.regions.values()), axis=0)).to(torch.int)
            self.reg_mapping = torch.zeros((int(reg_orig.max()) + 1,)).to(torch.int).to(model.device)
            self.reg_mapping[reg_orig] = torch.arange(len(reg_orig)).to(torch.int).to(model.device)

            stat_orig = torch.unique(torch.concatenate(list(model.stations.values()), axis=0)).to(torch.int)
            self.stat_mapping = torch.zeros((int(stat_orig.max()) + 1,)).to(torch.int).to(model.device)
            self.stat_mapping[stat_orig] = torch.arange(len(stat_orig)).to(torch.int).to(model.device)

            site_orig = torch.unique(torch.concatenate(list(model.sites.values()), axis=0)).to(torch.int)
            self.site_mapping = torch.zeros((int(site_orig.max()) + 1,)).to(torch.int).to(model.device)
            self.site_mapping[site_orig] = torch.arange(len(site_orig)).to(torch.int).to(model.device)

    def get_init_state(self, batch_size: int = 1) -> torch.Tensor:
        return self.h0.repeat(1, batch_size, 1)


class EmbeddingTransferGRU(BaseModule):
    """
    Multi Task GRU for transfer
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super(EmbeddingTransferGRU, self).__init__(c, model)

        self.cult_embedding_layer = nn.Embedding(len(CROP_NAMES[c.DataConfig.dtype]), self.input_dim)
        self.reg_embedding_layer = nn.Embedding(len(REGIONS), self.input_dim) if self.synth_data is None else None
        self.stat_embedding_layer = nn.Embedding(len(STATIONS), self.input_dim) if self.synth_data is None else None
        self.site_embedding_layer = nn.Embedding(len(SITES), self.input_dim) if self.synth_data is None else None
        self.weather_output_dim = 128
        self.weather_output_dim2 = 64
        self.weather_output_dim3 = 32
        self.fc1 = nn.Linear(self.embed_dim, self.dim2)
        self.fc2 = nn.Linear(self.dim2, self.hidden_dim)
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.fc3 = nn.Linear(self.hidden_dim, self.dim2)
        self.hidden_to_params = nn.Linear(self.dim2, self.output_dim)
        self.weather_rnn = nn.GRU(self.input_dim, self.weather_output_dim, batch_first=True)
        self.attn_fc1 = nn.Linear(self.weather_output_dim + (self.embed_dim - self.input_dim), self.weather_output_dim2)
        self.attn_fc2 = nn.Linear(self.weather_output_dim2, self.weather_output_dim3)
        self.attn_fc3 = nn.Linear(self.weather_output_dim3, 1)

        # self.mha = nn.MultiheadAttention(embed_dim=self.hidden_dim, batch_first=True, num_heads=4)
        self.softmax = nn.Softmax(dim=1)

        self.h0 = nn.Parameter(torch.zeros(1, self.hidden_dim))

    def forward(
        self,
        input: torch.Tensor = None,
        hn: torch.Tensor = None,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cult_embed = self.cult_embedding_layer(self.cult_mapping[cultivars.flatten().to(torch.int)])
        reg_embed = (
            self.reg_embedding_layer(self.reg_mapping[regions.flatten().to(torch.int)])
            if self.synth_data is None
            else None
        )
        stat_embed = (
            self.stat_embedding_layer(self.stat_mapping[stations.flatten().to(torch.int)])
            if self.synth_data is None
            else None
        )
        site_embed = (
            self.site_embedding_layer(self.site_mapping[sites.flatten().to(torch.int)])
            if self.synth_data is None
            else None
        )

        if input.ndim == 2:
            input = input.unsqueeze(1)

        cult_embed = cult_embed.unsqueeze(1).expand_as(input)
        reg_embed = reg_embed.unsqueeze(1).expand_as(input) if self.synth_data is None else None
        stat_embed = stat_embed.unsqueeze(1).expand_as(input) if self.synth_data is None else None
        site_embed = site_embed.unsqueeze(1).expand_as(input) if self.synth_data is None else None
        if self.embed_op_str == "concat_reg":
            gru_input = self.embed_op(cult_embed, site_embed, stat_embed, reg_embed, input)
        elif self.embed_op_str == "concat_stat":
            gru_input = self.embed_op(cult_embed, site_embed, stat_embed, input)
        elif self.embed_op_str == "concat_site":
            gru_input = self.embed_op(cult_embed, site_embed, input)
        elif self.embed_op_str == "concat":
            gru_input = self.embed_op(cult_embed, input)

        x = self.fc1(gru_input)
        x = self.fc2(x)
        gru_out, hn = self.rnn(x, hn)
        out = F.relu(gru_out)
        out = F.relu(self.fc3(out))
        params = self.hidden_to_params(out)

        return params, hn

    def forward_transfer(
        self,
        input: torch.Tensor = None,
        hn: torch.Tensor = None,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        all_ids: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the forward pass for transfering.

        Args:
            input: torch tensor of weather
            hn: current state of the RNN
            cultivars: cultivar IDs of transfer target
            regions: region IDS of transfer target
            stations: station IDs of transfer target
            sites: site IDS of transfer target
            all_ids: tensor of (N, 4) of all possible region/station/site/cultivar IDs that can be used for transfer
        """

        if input.ndim == 2:
            input = input.unsqueeze(1)
        transfer_input = input.repeat_interleave(all_ids.shape[0], dim=0)
        transfer_hn = hn.repeat_interleave(all_ids.shape[0], dim=1)
        transfer_ids = all_ids.repeat(input.shape[0], 1)

        transfer_cult_embed = self.cult_embedding_layer(
            self.cult_mapping[transfer_ids[:, 3].flatten().to(torch.int)]
        ).detach()
        transfer_reg_embed = (
            self.reg_embedding_layer(self.reg_mapping[transfer_ids[:, 0].flatten().to(torch.int)]).detach()
            if self.synth_data is None
            else None
        )
        transfer_stat_embed = (
            self.stat_embedding_layer(self.stat_mapping[transfer_ids[:, 1].flatten().to(torch.int)]).detach()
            if self.synth_data is None
            else None
        )
        transfer_site_embed = (
            self.site_embedding_layer(self.site_mapping[transfer_ids[:, 2].flatten().to(torch.int)]).detach()
            if self.synth_data is None
            else None
        )

        transfer_cult_embed = transfer_cult_embed.unsqueeze(1).expand_as(transfer_input)
        transfer_reg_embed = (
            transfer_reg_embed.unsqueeze(1).expand_as(transfer_input) if self.synth_data is None else None
        )
        transfer_stat_embed = (
            transfer_stat_embed.unsqueeze(1).expand_as(transfer_input) if self.synth_data is None else None
        )
        transfer_site_embed = (
            transfer_site_embed.unsqueeze(1).expand_as(transfer_input) if self.synth_data is None else None
        )

        with torch.no_grad():
            transfer_pred_orig, _ = self.forward(
                transfer_input,
                hn=transfer_hn,
                regions=transfer_ids[:, 0],
                stations=transfer_ids[:, 1],
                sites=transfer_ids[:, 2],
                cultivars=transfer_ids[:, 3],
            )
            transfer_pred_orig = transfer_pred_orig.detach()

        # Attention mech
        weather_output, _ = self.weather_rnn(transfer_input)

        if self.embed_op_str == "concat_reg":
            transfer_input = self.embed_op(
                transfer_cult_embed, transfer_site_embed, transfer_stat_embed, transfer_reg_embed, weather_output
            )
        elif self.embed_op_str == "concat_stat":
            transfer_input = self.embed_op(
                transfer_cult_embed, transfer_site_embed, transfer_stat_embed, weather_output
            )
        elif self.embed_op_str == "concat_site":
            transfer_input = self.embed_op(transfer_cult_embed, transfer_site_embed, weather_output)
        elif self.embed_op_str == "concat":
            transfer_input = self.embed_op(transfer_cult_embed, weather_output)

        T = input.shape[1]
        B = input.shape[0]
        transfer_output = self.attn_fc1(transfer_input).relu()
        transfer_output = self.attn_fc2(transfer_output).relu()
        transfer_output = self.attn_fc3(transfer_output)
        transfer_attn = transfer_output.reshape(B, -1, T, self.output_dim)
        softm_attn = self.softmax(transfer_attn)
        transfer_pred = transfer_pred_orig.reshape(B, -1, T, self.output_dim)
        transfer_pred = torch.sum(softm_attn * transfer_pred, dim=1)

        return transfer_pred, softm_attn, transfer_pred_orig.reshape(B, -1, T, self.output_dim)


class DeepEmbeddingTransferGRU(BaseModule):
    """
    DeepRNN Embedding Transfer GRU
    Takes one hot embedding of cultivar and passes through RNN
    to create a regression prediction of the output
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super(DeepEmbeddingTransferGRU, self).__init__(c, model)

        self.cult_embedding_layer = nn.Embedding(len(CROP_NAMES[c.DataConfig.dtype]), self.input_dim)
        self.reg_embedding_layer = nn.Embedding(len(REGIONS), self.input_dim) if self.synth_data is None else None
        self.stat_embedding_layer = nn.Embedding(len(STATIONS), self.input_dim) if self.synth_data is None else None
        self.site_embedding_layer = nn.Embedding(len(SITES), self.input_dim) if self.synth_data is None else None
        self.fc1 = nn.Linear(self.embed_dim, self.dim2)
        self.fc2 = nn.Linear(self.dim2, self.hidden_dim)
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.fc3 = nn.Linear(self.hidden_dim, self.dim2)
        self.hidden_to_output = nn.Linear(self.dim2, self.output_dim)

        self.mha = nn.MultiheadAttention(embed_dim=self.hidden_dim, batch_first=True, num_heads=4)

        self.h0 = nn.Parameter(torch.zeros(1, self.hidden_dim))

        self.init_state = self.h0

    def forward(
        self,
        input: torch.Tensor = None,
        hn: torch.Tensor = None,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cult_embed = self.cult_embedding_layer(self.cult_mapping[cultivars.flatten().to(torch.int)])
        reg_embed = (
            self.reg_embedding_layer(self.reg_mapping[regions.flatten().to(torch.int)])
            if self.synth_data is None
            else None
        )
        stat_embed = (
            self.stat_embedding_layer(self.stat_mapping[stations.flatten().to(torch.int)])
            if self.synth_data is None
            else None
        )
        site_embed = (
            self.site_embedding_layer(self.site_mapping[sites.flatten().to(torch.int)])
            if self.synth_data is None
            else None
        )

        if input.ndim == 2:
            input = input.unsqueeze(1)

        cult_embed = cult_embed.unsqueeze(1).expand_as(input)
        reg_embed = reg_embed.unsqueeze(1).expand_as(input) if self.synth_data is None else None
        stat_embed = stat_embed.unsqueeze(1).expand_as(input) if self.synth_data is None else None
        site_embed = site_embed.unsqueeze(1).expand_as(input) if self.synth_data is None else None

        if self.embed_op_str == "concat_reg":
            gru_input = self.embed_op(cult_embed, site_embed, stat_embed, reg_embed, input)
        elif self.embed_op_str == "concat_stat":
            gru_input = self.embed_op(cult_embed, site_embed, stat_embed, input)
        elif self.embed_op_str == "concat_site":
            gru_input = self.embed_op(cult_embed, site_embed, input)
        elif self.embed_op_str == "concat":
            gru_input = self.embed_op(cult_embed, input)

        x = self.fc1(gru_input).relu()
        x = self.fc2(x).relu()
        gru_out, hn = self.rnn(x, hn)

        gru_out = self.fc3(gru_out).relu()
        output = self.hidden_to_output(gru_out).squeeze(1)

        return output, hn
