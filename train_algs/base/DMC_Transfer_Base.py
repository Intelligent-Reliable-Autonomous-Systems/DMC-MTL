"""Base class for DMC Transfer

Written by Will Solow, 2025"""

import torch
import torch.nn as nn
from omegaconf import DictConfig

from train_algs.base.DMC_Base import BaseModule


class TransferAttentionMixin:
    """Shared attention-based transfer forward used by Embedding/Deep transfer GRUs."""

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
        B, T, _ = input.shape

        # Embed weather
        weather_output, _ = self.transfer_weather_rnn(input)

        true_ids = torch.cat((regions, stations, sites, cultivars), dim=1)
        ex_all_ids = all_ids.unsqueeze(0).expand(input.shape[0], -1, -1)
        N, M, D = ex_all_ids.shape
        mask = (ex_all_ids != true_ids[:, None, :]).any(dim=-1)
        transfer_ids = ex_all_ids[mask].reshape(N, (M - 1), D)

        with torch.no_grad():
            transfer_preds = []
            for n in range(N):
                pred, _ = self.forward(
                    input[n].unsqueeze(0).expand(M - 1, -1, -1),
                    hn=hn[:, n].unsqueeze(1).expand(-1, M - 1, -1).contiguous(),
                    regions=transfer_ids[n, :, 0],
                    stations=transfer_ids[n, :, 1],
                    sites=transfer_ids[n, :, 2],
                    cultivars=transfer_ids[n, :, 3],
                )
                transfer_preds.append(pred)
        transfer_pred_orig = torch.stack(transfer_preds, dim=0).detach()

        transfer_cult_embed = (
            self.cult_embedding_layer(self.cult_mapping[transfer_ids[:, :, 3].flatten().to(torch.int)])
            .detach()
            .view(N, M - 1, -1)
        )
        transfer_reg_embed = (
            self.reg_embedding_layer(self.reg_mapping[transfer_ids[:, :, 0].flatten().to(torch.int)])
            .detach()
            .view(N, M - 1, -1)
        )
        transfer_stat_embed = (
            self.stat_embedding_layer(self.stat_mapping[transfer_ids[:, :, 1].flatten().to(torch.int)])
            .detach()
            .view(N, M - 1, -1)
        )
        transfer_site_embed = (
            self.site_embedding_layer(self.site_mapping[transfer_ids[:, :, 2].flatten().to(torch.int)])
            .detach()
            .view(N, M - 1, -1)
        )

        cult_embed = self.cult_embedding_layer(self.cult_mapping[cultivars.flatten().to(torch.int)]).detach()
        reg_embed = self.reg_embedding_layer(self.reg_mapping[regions.flatten().to(torch.int)]).detach()
        stat_embed = self.stat_embedding_layer(self.stat_mapping[stations.flatten().to(torch.int)]).detach()
        site_embed = self.site_embedding_layer(self.site_mapping[sites.flatten().to(torch.int)]).detach()

        transfer_cult_embed = transfer_cult_embed[:, :, None, :].expand(-1, -1, T, -1)
        transfer_reg_embed = transfer_reg_embed[:, :, None, :].expand(-1, -1, T, -1)
        transfer_stat_embed = transfer_stat_embed[:, :, None, :].expand(-1, -1, T, -1)
        transfer_site_embed = transfer_site_embed[:, :, None, :].expand(-1, -1, T, -1)
        cult_embed = cult_embed[:, None, None, :].expand(-1, M - 1, T, -1)
        reg_embed = reg_embed[:, None, None, :].expand(-1, M - 1, T, -1)
        stat_embed = stat_embed[:, None, None, :].expand(-1, M - 1, T, -1)
        site_embed = site_embed[:, None, None, :].expand(-1, M - 1, T, -1)
        weather_output = weather_output[:, None, :, :].expand(-1, M - 1, T, -1)

        transfer_input = self.apply_embed_op(
            torch.cat((transfer_cult_embed, cult_embed), dim=-1),
            torch.cat((transfer_site_embed, site_embed), dim=-1),
            torch.cat((transfer_stat_embed, stat_embed), dim=-1),
            torch.cat((transfer_reg_embed, reg_embed), dim=-1),
            weather_output,
        )

        softm_attn = self.transfer_attn_fc(transfer_input).softmax(dim=1)

        transfer_pred = torch.sum(softm_attn * transfer_pred_orig, dim=1)

        return (
            transfer_pred,
            softm_attn,
            transfer_pred_orig.reshape(B, -1, T, self.output_dim),
        )


class EmbeddingTransferGRU(TransferAttentionMixin, BaseModule):
    """
    Multi Task GRU for transfer
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super(EmbeddingTransferGRU, self).__init__(c, model)

        self._build_hierarchy_embeddings(c, embed_size_aware=False, synth_optional=True)
        self.weather_output_dim = 128
        self.weather_output_dim2 = 64
        self.weather_output_dim3 = 32
        self.fc1 = nn.Sequential(nn.Linear(self.embed_dim, self.dim2), nn.Linear(self.dim2, self.hidden_dim))
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.fc2 = nn.Sequential(
            nn.ReLU(), nn.Linear(self.hidden_dim, self.dim2), nn.ReLU(), nn.Linear(self.dim2, self.output_dim)
        )
        self.transfer_weather_rnn = nn.GRU(self.input_dim, self.weather_output_dim, batch_first=True)
        self.transfer_attn_fc = nn.Sequential(
            nn.Linear(
                self.weather_output_dim + (2 * self.embed_dim - 2 * self.input_dim),
                self.weather_output_dim2,
            ),
            nn.ReLU(),
            nn.Linear(self.weather_output_dim2, self.weather_output_dim3),
            nn.ReLU(),
            nn.Linear(self.weather_output_dim3, 1),
        )

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
        gru_input, _ = self._prep_hierarchy_input(
            input, regions, stations, sites, cultivars, synth_safe=False, expand="expand_as"
        )
        x = self.fc1(gru_input)
        gru_out, hn = self.rnn(x, hn)
        params = self.fc2(gru_out)

        return params, hn


class DeepEmbeddingTransferGRU(TransferAttentionMixin, BaseModule):
    """
    DeepRNN Embedding Transfer GRU
    Takes one hot embedding of cultivar and passes through RNN
    to create a regression prediction of the output
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super(DeepEmbeddingTransferGRU, self).__init__(c, model)

        self.weather_output_dim = 128
        self.weather_output_dim2 = 64
        self.weather_output_dim3 = 32

        self._build_hierarchy_embeddings(c, embed_size_aware=False, synth_optional=True)
        self.fc1 = nn.Sequential(
            nn.Linear(self.embed_dim, self.dim2), nn.ReLU(), nn.Linear(self.dim2, self.hidden_dim), nn.ReLU()
        )
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.fc2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.dim2), nn.ReLU(), nn.Linear(self.dim2, self.output_dim)
        )

        self.transfer_weather_rnn = nn.GRU(
            self.input_dim if "Deep" in c.DConfig.type else (self.input_dim - len(c.PConfig.output_vars)),
            self.weather_output_dim,
            batch_first=True,
        )
        self.transfer_attn_fc = nn.Sequential(
            nn.Linear(
                self.weather_output_dim + (2 * self.embed_dim - 2 * self.input_dim),
                self.weather_output_dim2,
            ),
            nn.ReLU(),
            nn.Linear(self.weather_output_dim2, self.weather_output_dim3),
            nn.ReLU(),
            nn.Linear(self.weather_output_dim3, 1),
        )

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
        gru_input, _ = self._prep_hierarchy_input(
            input, regions, stations, sites, cultivars, synth_safe=True, expand="expand_as"
        )
        x = self.fc1(gru_input)
        gru_out, hn = self.rnn(x, hn)
        output = self.fc2(gru_out)

        return output, hn


class DeepAdaptTransferSineGRU(BaseModule):
    """
    DeepRNN Adapt Transfer GRU. Learns a weather and cold-hardiness encoding
    and uses this to condition embedding
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super(DeepAdaptTransferSineGRU, self).__init__(c, model)

        self.weather_output_dim = 128

        self._build_hierarchy_embeddings(c, embed_size_aware=False, synth_optional=True)
        self.fc1 = nn.Sequential(
            nn.Linear(self.embed_dim, self.dim2), nn.ReLU(), nn.Linear(self.dim2, self.hidden_dim), nn.ReLU()
        )
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.fc2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.dim2), nn.ReLU(), nn.Linear(self.dim2, self.output_dim)
        )
        self.h0 = nn.Parameter(torch.zeros(1, self.hidden_dim))
        self.transfer_weather_rnn = nn.GRU(self.input_dim, self.weather_output_dim, batch_first=True)
        self.transfer_ch_rnn = nn.GRU(
            self.output_dim + 2, self.weather_output_dim, batch_first=True
        )  # Sine/Cosine embedding
        self.transfer_pool_attn = nn.Linear(self.weather_output_dim, 1)
        self.transfer_embed_attn = nn.Linear(self.weather_output_dim, self.embed_dim - self.input_dim)

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
        gru_input, _ = self._prep_hierarchy_input(
            input, regions, stations, sites, cultivars, synth_safe=True, expand="expand_as"
        )
        x = self.fc1(gru_input)
        gru_out, hn = self.rnn(x, hn)
        output = self.fc2(gru_out)

        return output, hn

    def compute_embed(
        self,
        weather_input: torch.Tensor = None,
        ch: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._weather_ch_cross_attn_embed(weather_input, ch, sine_day=True)

    def forward_embed(
        self,
        input: torch.Tensor = None,
        hn: torch.Tensor = None,
        embed: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if input.ndim == 2:
            input = input.unsqueeze(1)

        embed = embed.unsqueeze(1).expand(-1, input.shape[1], -1)

        gru_input = torch.cat((embed, input), dim=-1)

        x = self.fc1(gru_input)
        gru_out, hn = self.rnn(x, hn)
        output = self.fc2(gru_out)

        return output, hn


class DeepAdaptTransferGRU(BaseModule):
    """
    DeepRNN Adapt Transfer GRU. Learns a weather and cold-hardiness encoding
    and uses this to condition embedding
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super(DeepAdaptTransferGRU, self).__init__(c, model)

        self.weather_output_dim = 128

        self._build_hierarchy_embeddings(c, embed_size_aware=False, synth_optional=True)
        self.fc1 = nn.Sequential(
            nn.Linear(self.embed_dim, self.dim2), nn.ReLU(), nn.Linear(self.dim2, self.hidden_dim), nn.ReLU()
        )
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.fc2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.dim2), nn.ReLU(), nn.Linear(self.dim2, self.output_dim)
        )
        self.h0 = nn.Parameter(torch.zeros(1, self.hidden_dim))
        self.transfer_weather_rnn = nn.GRU(self.input_dim, self.weather_output_dim, batch_first=True)
        self.transfer_ch_rnn = nn.GRU(
            self.output_dim, self.weather_output_dim, batch_first=True
        )  # Sine/Cosine embedding
        self.transfer_pool_attn = nn.Linear(self.weather_output_dim, 1)
        self.transfer_embed_attn = nn.Linear(self.weather_output_dim, self.embed_dim - self.input_dim)

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
        gru_input, _ = self._prep_hierarchy_input(
            input, regions, stations, sites, cultivars, synth_safe=True, expand="expand_as"
        )
        x = self.fc1(gru_input)
        gru_out, hn = self.rnn(x, hn)
        output = self.fc2(gru_out)

        return output, hn

    def compute_embed(
        self,
        weather_input: torch.Tensor = None,
        ch: torch.Tensor = None,
        map_idx: torch.Tensor = None,
        bsize: int = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._weather_ch_cross_attn_embed(
            weather_input, ch, sine_day=False, map_idx=map_idx, bsize=bsize
        )

    def forward_embed(
        self,
        input: torch.Tensor = None,
        hn: torch.Tensor = None,
        embed: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if input.ndim == 2:
            input = input.unsqueeze(1)

        embed = embed.unsqueeze(1).expand(-1, input.shape[1], -1)

        gru_input = torch.cat((embed, input), dim=-1)

        x = self.fc1(gru_input)
        gru_out, hn = self.rnn(x, hn)
        output = self.fc2(gru_out)

        return output, hn


class DeepTransferAttnGRU(BaseModule):
    """
    DeepRNN Adapt Transfer GRU. Learns a weather and cold-hardiness encoding
    and uses this to condition embedding
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super(DeepTransferAttnGRU, self).__init__(c, model)

        self.weather_output_dim = 128

        self._build_hierarchy_embeddings(c, embed_size_aware=False, synth_optional=True)
        self.fc1 = nn.Sequential(
            nn.Linear(self.embed_dim, self.dim2), nn.ReLU(), nn.Linear(self.dim2, self.hidden_dim), nn.ReLU()
        )
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.fc2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.dim2), nn.ReLU(), nn.Linear(self.dim2, self.output_dim)
        )
        self.h0 = nn.Parameter(torch.zeros(1, self.hidden_dim))
        self.transfer_rnn = nn.GRU(self.input_dim + self.output_dim, self.weather_output_dim, batch_first=True)
        self.transfer_pool_attn = nn.Linear(self.weather_output_dim, 1)
        self.transfer_embed_attn = nn.Linear(self.weather_output_dim, self.embed_dim - self.input_dim)

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
        gru_input, _ = self._prep_hierarchy_input(
            input, regions, stations, sites, cultivars, synth_safe=True, expand="expand_as"
        )
        x = self.fc1(gru_input)
        gru_out, hn = self.rnn(x, hn)
        output = self.fc2(gru_out)

        return output, hn

    def compute_embed(
        self,
        weather_input: torch.Tensor = None,
        ch: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if weather_input.ndim == 2:
            weather_input = weather_input.unsqueeze(1)
        if ch.ndim == 2:
            ch = ch.unsqueeze(1)
        ch = torch.nan_to_num(ch, nan=0.0)
        ch_w_input = torch.cat((weather_input, ch), dim=-1)
        ch_w_encode, _ = self.transfer_rnn(ch_w_input)  # (T,d)

        attn_weights = self.transfer_pool_attn(ch_w_encode).softmax(dim=1)  # (B, T, 1)
        attn_pooled = (ch_w_encode * attn_weights).sum(dim=1)  # (B, d)
        embed = self.transfer_embed_attn(attn_pooled)  # (B,embed_dim)

        return embed

    def forward_embed(
        self,
        input: torch.Tensor = None,
        hn: torch.Tensor = None,
        embed: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if input.ndim == 2:
            input = input.unsqueeze(1)

        embed = embed.unsqueeze(1).expand(-1, input.shape[1], -1)

        gru_input = torch.cat((embed, input), dim=-1)

        x = self.fc1(gru_input)
        gru_out, hn = self.rnn(x, hn)
        output = self.fc2(gru_out)

        return output, hn


class DeepTransferGRU(DeepTransferAttnGRU):
    def __init__(self, c, model):
        super().__init__(c, model)

    def compute_embed(
        self,
        weather_input: torch.Tensor = None,
        ch: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if weather_input.ndim == 2:
            weather_input = weather_input.unsqueeze(1)
        if ch.ndim == 2:
            ch = ch.unsqueeze(1)
        ch = torch.nan_to_num(ch, nan=0.0)
        ch_w_input = torch.cat((weather_input, ch), dim=-1)
        ch_w_encode, _ = self.transfer_rnn(ch_w_input)  # (T,d)

        embed = self.transfer_embed_attn(ch_w_encode[:, -1])  # (B,embed_dim)

        return embed
