"""Base class for Param RNN

Written by Will Solow, 2025"""

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from assets import (
    CROP_NAMES,
    CULT_STR,
    CULT_STR_BARE,
    CULT_STR_MINI,
    CULT_STR_TEST,
    CULT_STR_VERB,
    REGION_STR,
    REGION_STR_BARE,
    REGION_STR_MINI,
    REGION_STR_TEST,
    REGION_STR_VERB,
    REGIONS,
    SITE_STR,
    SITE_STR_BARE,
    SITE_STR_MINI,
    SITE_STR_TEST,
    SITE_STR_VERB,
    SITES,
    STATION_STR,
    STATION_STR_BARE,
    STATION_STR_MINI,
    STATION_STR_TEST,
    STATION_STR_VERB,
    STATIONS,
)
from train_algs.base.util import set_embedding_op


class BaseModule(nn.Module):
    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super().__init__()
        self.config = c
        self.hidden_dim = c.DConfig.hidden_dim
        self.dim2 = int(self.hidden_dim / 2)
        self.input_dim = model.get_input_dim(c)
        self.output_dim = model.get_output_dim(c)
        self.embed_dim = set_embedding_op(self, c.DConfig.embed_op, c.DConfig.embed_size)

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

    def lookup_id_embeds(
        self,
        regions=None,
        stations=None,
        sites=None,
        cultivars=None,
        *,
        synth_safe: bool = False,
    ):
        """Lookup cult/reg/stat/site embedding vectors via ID mappings.

        When ``synth_safe`` and ``synth_data`` is set, geo embeds are None
        (matches DeepEmbeddingGRU / Adapt convention).
        """
        cult_embed = self.cult_embedding_layer(self.cult_mapping[cultivars.flatten().to(torch.int)])
        if synth_safe and self.synth_data is not None:
            return cult_embed, None, None, None
        reg_embed = self.reg_embedding_layer(self.reg_mapping[regions.flatten().to(torch.int)])
        stat_embed = self.stat_embedding_layer(self.stat_mapping[stations.flatten().to(torch.int)])
        site_embed = self.site_embedding_layer(self.site_mapping[sites.flatten().to(torch.int)])
        return cult_embed, reg_embed, stat_embed, site_embed

    def expand_embeds_to_T(self, cult_embed, reg_embed, stat_embed, site_embed, T: int, *, mode: str = "repeat"):
        """Expand ID embeds along time. mode: 'repeat' | 'expand_as' (needs weather for expand_as)."""
        if mode == "repeat":
            cult_embed = cult_embed.unsqueeze(1).repeat(1, T, 1)
            reg_embed = reg_embed.unsqueeze(1).repeat(1, T, 1) if reg_embed is not None else None
            stat_embed = stat_embed.unsqueeze(1).repeat(1, T, 1) if stat_embed is not None else None
            site_embed = site_embed.unsqueeze(1).repeat(1, T, 1) if site_embed is not None else None
        else:
            raise ValueError(f"Unknown expand mode `{mode}`")
        return cult_embed, reg_embed, stat_embed, site_embed

    def apply_embed_op(
        self, cult_embed, site_embed=None, stat_embed=None, reg_embed=None, weather=None, *, ids_only: bool = False
    ):
        """Single dispatch for concat_reg / concat_stat / concat_site / concat.

        When ``ids_only``, concatenate hierarchy embeds without weather
        (FoundationParamModel).
        """
        if ids_only:
            if self.embed_op_str == "concat_reg":
                return torch.cat((cult_embed, site_embed, stat_embed, reg_embed), dim=-1)
            if self.embed_op_str == "concat_stat":
                return torch.cat((cult_embed, site_embed, stat_embed), dim=-1)
            if self.embed_op_str == "concat_site":
                return torch.cat((cult_embed, site_embed), dim=-1)
            if self.embed_op_str == "concat":
                return cult_embed
            raise ValueError(f"Unsupported embed_op `{self.embed_op_str}`")
        if self.embed_op_str == "concat_reg":
            return self.embed_op(cult_embed, site_embed, stat_embed, reg_embed, weather)
        if self.embed_op_str == "concat_stat":
            return self.embed_op(cult_embed, site_embed, stat_embed, weather)
        if self.embed_op_str == "concat_site":
            return self.embed_op(cult_embed, site_embed, weather)
        if self.embed_op_str == "concat":
            return self.embed_op(cult_embed, weather)
        raise ValueError(f"Unsupported embed_op `{self.embed_op_str}`")

    def _build_hierarchy_embeddings(
        self, c: DictConfig, *, embed_size_aware: bool = True, synth_optional: bool = False
    ):
        """Create cult/reg/stat/site Embedding layers (shared constructor pattern)."""
        edim = self.input_dim if (not embed_size_aware or c.DConfig.embed_size is None) else c.DConfig.embed_size
        self.cult_embedding_layer = nn.Embedding(len(CROP_NAMES[c.DataConfig.dtype]), edim)
        if synth_optional and self.synth_data is not None:
            self.reg_embedding_layer = None
            self.stat_embedding_layer = None
            self.site_embedding_layer = None
        else:
            self.reg_embedding_layer = nn.Embedding(len(REGIONS), edim)
            self.stat_embedding_layer = nn.Embedding(len(STATIONS), edim)
            self.site_embedding_layer = nn.Embedding(len(SITES), edim)

    def _prep_hierarchy_input(
        self,
        input,
        regions=None,
        stations=None,
        sites=None,
        cultivars=None,
        *,
        synth_safe: bool = True,
        expand: str = "repeat",
    ):
        """Lookup → expand → apply_embed_op. Returns (gru_input, input)."""
        cult_embed, reg_embed, stat_embed, site_embed = self.lookup_id_embeds(
            regions, stations, sites, cultivars, synth_safe=synth_safe
        )
        if input.ndim == 2:
            input = input.unsqueeze(1)
        if expand == "repeat":
            cult_embed, reg_embed, stat_embed, site_embed = self.expand_embeds_to_T(
                cult_embed, reg_embed, stat_embed, site_embed, input.shape[1]
            )
        elif expand == "expand_as":
            cult_embed = cult_embed.unsqueeze(1).expand_as(input)
            reg_embed = reg_embed.unsqueeze(1).expand_as(input) if reg_embed is not None else None
            stat_embed = stat_embed.unsqueeze(1).expand_as(input) if stat_embed is not None else None
            site_embed = site_embed.unsqueeze(1).expand_as(input) if site_embed is not None else None
        else:
            raise ValueError(f"Unknown expand mode `{expand}`")
        gru_input = self.apply_embed_op(cult_embed, site_embed, stat_embed, reg_embed, input)
        return gru_input, input

    def forward_embed_direct(
        self,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        **kwargs,
    ):
        """Shared direct ID embedding lookup (reg, stat, site, cult)."""
        cult_embed, reg_embed, stat_embed, site_embed = self.lookup_id_embeds(
            regions, stations, sites, cultivars, synth_safe=True
        )
        return reg_embed, stat_embed, site_embed, cult_embed

    def _build_text_projectors(self, out_dim: int):
        """MLP towers projecting 1024-d text embeds → out_dim (cult/site/stat; reg reuses stat)."""

        def _tower():
            return nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, out_dim),
            )

        self.cult_embedder = _tower()
        self.site_embedder = _tower()
        self.stat_embedder = _tower()

    def _resolve_text_tables(self):
        """String tables for text embedding (verbose_level or EmbeddingTextGRU.verbose)."""
        level = getattr(self.config.DConfig, "verbose_level", None)
        if level is None:
            level = "verb" if getattr(self, "verbose", False) else "base"
        tables = {
            "verb": (CULT_STR_VERB, SITE_STR_VERB, STATION_STR_VERB, REGION_STR_VERB),
            "mini": (CULT_STR_MINI, SITE_STR_MINI, STATION_STR_MINI, REGION_STR_MINI),
            "bare": (CULT_STR_BARE, SITE_STR_BARE, STATION_STR_BARE, REGION_STR_BARE),
            "base": (CULT_STR, SITE_STR, STATION_STR, REGION_STR),
            "test": (CULT_STR_TEST, SITE_STR_TEST, STATION_STR_TEST, REGION_STR_TEST),
        }
        if level not in tables:
            raise ValueError(f"Unknown verbose_level `{level}`")
        return tables[level]

    def forward_embed_text(
        self,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        text_embedder: nn.Module = None,
        **kwargs,
    ):
        """Shared text→MLP embedding (crossed cult/site wiring; reg uses stat_embedder)."""
        cult_str, site_str, station_str, region_str = self._resolve_text_tables()
        device = cultivars.device
        with torch.no_grad():
            cult_embed_text = text_embedder.encode(
                cult_str[self.cult_mapping[cultivars.flatten().to(torch.int)].cpu().numpy()],
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=device,
            ).clone()
            site_embed_text = text_embedder.encode(
                site_str[self.site_mapping[sites.flatten().to(torch.int)].cpu().numpy()],
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=device,
            ).clone()
            stat_embed_text = text_embedder.encode(
                station_str[self.stat_mapping[stations.flatten().to(torch.int)].cpu().numpy()],
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=device,
            ).clone()
            reg_embed_text = text_embedder.encode(
                region_str[self.reg_mapping[regions.flatten().to(torch.int)].cpu().numpy()],
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=device,
            ).clone()
        cult_embed = self.cult_embedder(site_embed_text)
        site_embed = self.site_embedder(cult_embed_text)
        stat_embed = self.stat_embedder(stat_embed_text)
        reg_embed = self.stat_embedder(reg_embed_text)
        return reg_embed, stat_embed, site_embed, cult_embed

    def _weather_ch_cross_attn_embed(
        self,
        weather_input: torch.Tensor,
        ch: torch.Tensor,
        *,
        sine_day: bool = False,
        map_idx: torch.Tensor = None,
        bsize: int = None,
    ):
        """Shared weather–CH cross-attention embed (Foundation / DeepAdapt / Sine)."""
        if weather_input.ndim == 2:
            weather_input = weather_input.unsqueeze(1)
        if ch.ndim == 2:
            ch = ch.unsqueeze(1)
        w_encode, _ = self.transfer_weather_rnn(weather_input)

        B, T, d = ch.shape
        ch_mask = ~torch.isnan(ch)
        if sine_day:
            ch_inds = torch.arange(T, device=ch.device).unsqueeze(1).repeat(1, 2) * (2 * torch.pi) / 365
            ch_inds[:, 0] = torch.sin(ch_inds[:, 0])
            ch_inds[:, 1] = torch.cos(ch_inds[:, 1])
            ch_inds = ch_inds.unsqueeze(0).repeat(B, 1, 1)
            ch_num = torch.where(ch_mask, ch, torch.zeros_like(ch))
            ch_clean = torch.cat((ch_num, ch_inds), dim=-1)
            feat_dim = d + 2
        else:
            ch_clean = torch.where(ch_mask, ch, torch.zeros_like(ch))
            feat_dim = d

        ch_lengths = ch_mask.sum(dim=(1, 2))
        max_len = ch_lengths.max().item()
        if max_len == 0:
            return torch.zeros(B, self.embed_dim, device=ch.device, requires_grad=True)

        ch_arr_list = []
        for i in range(B):
            mask_i = ch_mask[i].any(dim=-1)
            valid_data = ch_clean[i][mask_i]
            valid_len = valid_data.shape[0]
            if valid_len < max_len:
                pad = torch.zeros((max_len - valid_len, feat_dim), device=ch.device, dtype=ch.dtype)
                valid_data = torch.cat([valid_data, pad], dim=0)
            ch_arr_list.append(valid_data)
        ch_arr = torch.stack(ch_arr_list)

        ch_embed, _ = self.transfer_ch_rnn(ch_arr)
        attn_mask = torch.arange(max_len, device=ch.device).unsqueeze(0) < ch_lengths.unsqueeze(1)
        ch_encode = torch.where(~attn_mask.unsqueeze(-1), torch.zeros(ch_embed.shape[-1], device=ch.device), ch_embed)

        attn_scores = torch.bmm(w_encode, ch_encode.transpose(1, 2))
        attn_scores = attn_scores.masked_fill(~attn_mask.unsqueeze(1), float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=2)
        ch_aligned = torch.bmm(attn_weights, ch_encode)

        attn_weights = self.transfer_pool_attn(ch_aligned).softmax(dim=1)
        attn_pooled = (ch_aligned * attn_weights).sum(dim=1)
        embed_all = self.transfer_embed_attn(attn_pooled)

        if map_idx is not None and bsize is not None:
            embed = torch.zeros((bsize, embed_all.shape[-1]), device=ch.device)
            embed = embed.scatter_reduce(
                0, map_idx.unsqueeze(1).expand(-1, embed_all.shape[-1]), embed_all, reduce="mean", include_self=False
            )
            return embed
        return embed_all


class FCGRU(BaseModule):
    """
    Single Task GRU
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super().__init__(c, model)

        self.fc1 = nn.Linear(self.input_dim, self.dim2)
        self.fc2 = nn.Linear(self.dim2, self.hidden_dim)
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.fc3 = nn.Linear(self.hidden_dim, self.dim2)
        if model.cultivars is None:
            self.param_heads = nn.ModuleList([nn.Linear(self.dim2, self.output_dim) for _ in range(1)])
        else:
            self.param_heads = nn.ModuleList(
                [nn.Linear(self.dim2, self.output_dim) for _ in range(len(CROP_NAMES[c.DataConfig.dtype]))]
            )

        self.h0 = nn.Parameter(torch.zeros(1, self.hidden_dim))
        self.init_state = self.h0

    def forward(
        self, input: torch.Tensor = None, hn: torch.Tensor = None, cultivars: torch.Tensor = None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input.ndim == 2:
            input = input.unsqueeze(1)
        x = self.fc1(input)
        x = self.fc2(x)
        _, hn = self.rnn(x, hn)
        out = F.relu(hn)
        out = F.relu(self.fc3(out)).squeeze(0)

        params = torch.empty(input.size(0), self.param_heads[0].out_features).to(input.device)
        if cultivars is None:
            for i in range(input.size(0)):
                params[i] = self.param_heads[0](out[i])
        else:
            for i in range(input.size(0)):
                params[i] = self.param_heads[cultivars[i][0].to(torch.int)](out[i])

        return params, hn


class EmbeddingFCGRU(BaseModule):
    """
    Multi Task GRU
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super().__init__(c, model)
        self._build_hierarchy_embeddings(c, embed_size_aware=True)
        self.fc1 = nn.Linear(self.embed_dim, self.dim2)
        self.fc2 = nn.Linear(self.dim2, self.hidden_dim)
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.fc3 = nn.Linear(self.hidden_dim, self.dim2)
        self.hidden_to_params = nn.Linear(self.dim2, self.output_dim)
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
            input, regions, stations, sites, cultivars, synth_safe=False, expand="repeat"
        )
        x = self.fc1(gru_input)
        x = self.fc2(x)
        _, hn = self.rnn(x, hn)
        out = F.relu(hn)
        out = F.relu(self.fc3(out))
        params = self.hidden_to_params(out).squeeze(0)
        return params, hn

    def forward_embed(
        self,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cult_embed, reg_embed, stat_embed, site_embed = self.lookup_id_embeds(
            regions, stations, sites, cultivars, synth_safe=False
        )
        return reg_embed, stat_embed, site_embed, cult_embed


class EmbeddingTextGRU(BaseModule):
    """
    Multi Task GRU with text embeddings. Set verbose=True for verbose string tables.
    """

    verbose = False

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super().__init__(c, model)
        self._build_text_projectors(self.input_dim)
        self.fc1 = nn.Linear(self.embed_dim, self.dim2)
        self.fc2 = nn.Linear(self.dim2, self.hidden_dim)
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.fc3 = nn.Linear(self.hidden_dim, self.dim2)
        self.hidden_to_output = nn.Linear(self.dim2, self.output_dim)
        self.h0 = nn.Parameter(torch.zeros(1, self.hidden_dim))

    def forward(
        self,
        input: torch.Tensor = None,
        hn: torch.Tensor = None,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        text_embedder: nn.Module = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input.ndim == 2:
            input = input.unsqueeze(1)
        reg_embed, stat_embed, site_embed, cult_embed = self.forward_embed_text(
            cultivars=cultivars,
            regions=regions,
            stations=stations,
            sites=sites,
            text_embedder=text_embedder,
        )
        cult_embed = cult_embed.unsqueeze(1).repeat(1, input.shape[1], 1)
        site_embed = site_embed.unsqueeze(1).repeat(1, input.shape[1], 1)
        stat_embed = stat_embed.unsqueeze(1).repeat(1, input.shape[1], 1)
        reg_embed = reg_embed.unsqueeze(1).repeat(1, input.shape[1], 1)
        gru_input = self.apply_embed_op(cult_embed, site_embed, stat_embed, reg_embed, input)
        x = self.fc1(gru_input).relu()
        x = self.fc2(x).relu()
        gru_out, hn = self.rnn(x, hn)
        gru_out = self.fc3(gru_out).relu()
        output = self.hidden_to_output(gru_out)
        return output, hn

    def forward_embed(
        self,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        text_embedder: nn.Module = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_embed_text(
            cultivars=cultivars,
            regions=regions,
            stations=stations,
            sites=sites,
            text_embedder=text_embedder,
            **kwargs,
        )


class EmbeddingVerboseTextGRU(EmbeddingTextGRU):
    """Text GRU using verbose CULT/SITE/STATION/REGION string tables."""

    verbose = True


class DeepEmbeddingGRU(BaseModule):
    """
    DeepRNN Embedding GRU
    Takes one hot embedding of cultivar and passes through RNN
    to create a regression prediction of the output
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super().__init__(c, model)
        self._build_hierarchy_embeddings(c, embed_size_aware=True)
        self.fc1 = nn.Linear(self.embed_dim, self.dim2)
        self.fc2 = nn.Linear(self.dim2, self.hidden_dim)
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.fc3 = nn.Linear(self.hidden_dim, self.dim2)
        self.hidden_to_output = nn.Linear(self.dim2, self.output_dim)
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
            input, regions, stations, sites, cultivars, synth_safe=True, expand="repeat"
        )
        x = self.fc1(gru_input).relu()
        x = self.fc2(x).relu()
        gru_out, hn = self.rnn(x, hn)
        gru_out = self.fc3(gru_out).relu()
        output = self.hidden_to_output(gru_out)
        return output, hn

    def forward_embed(
        self,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_embed_direct(cultivars=cultivars, regions=regions, stations=stations, sites=sites, **kwargs)


class DeepGRU(BaseModule):
    """
    DeepRNN GRU without hierarchy embeddings.
    Passes weather input directly through the FC → GRU trunk.
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super().__init__(c, model)
        self.embed_dim = self.input_dim
        self._build_hierarchy_embeddings(c, embed_size_aware=True)
        self.fc1 = nn.Linear(self.embed_dim, self.dim2)
        self.fc2 = nn.Linear(self.dim2, self.hidden_dim)
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.fc3 = nn.Linear(self.hidden_dim, self.dim2)
        self.hidden_to_output = nn.Linear(self.dim2, self.output_dim)
        self.h0 = nn.Parameter(torch.zeros(1, self.hidden_dim))
        self.init_state = self.h0

    def _build_hierarchy_embeddings(self, c: DictConfig, *, embed_size_aware: bool = True, synth_optional: bool = False):
        """No-op: DeepGRU does not use cultivar/region/station/site embeddings."""
        return

    def _prep_hierarchy_input(
        self,
        input,
        regions=None,
        stations=None,
        sites=None,
        cultivars=None,
        *,
        synth_safe: bool = True,
        expand: str = "repeat",
    ):
        """Return weather only (no embedding concat)."""
        if input.ndim == 2:
            input = input.unsqueeze(1)
        return input, input

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
            input, regions, stations, sites, cultivars, synth_safe=True, expand="repeat"
        )
        x = self.fc1(gru_input).relu()
        x = self.fc2(x).relu()
        gru_out, hn = self.rnn(x, hn)
        gru_out = self.fc3(gru_out).relu()
        output = self.hidden_to_output(gru_out)
        return output, hn


class EmbeddingFCFF(BaseModule):
    """
    Multi Task FF
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super().__init__(c, model)

        self.embedding_layer = nn.Embedding(len(CROP_NAMES[c.DataConfig.dtype]), self.input_dim)
        self.fc1 = nn.Linear(self.embed_dim, self.dim2)
        self.fc2 = nn.Linear(self.dim2, self.hidden_dim)
        self.rnn = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
        )
        self.fc3 = nn.Linear(self.hidden_dim, self.dim2)
        self.hidden_to_params = nn.Linear(self.dim2, self.output_dim)

        self.h0 = nn.Parameter(torch.zeros(1, self.hidden_dim))

    def forward(
        self,
        input: torch.Tensor = None,
        cultivars: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.embedding_layer(self.cult_mapping[cultivars.flatten().to(torch.int)])
        gru_input = self.embed_op(embed, input)
        x = self.fc1(gru_input)
        x = self.fc2(x)
        ff_out = self.rnn(x)
        ff_out = F.relu(ff_out)
        ff_out = F.relu(self.fc3(ff_out))
        params = self.hidden_to_params(ff_out)

        return params, None


class OneHotEmbeddingFCGRU(BaseModule):
    """
    Multi Task GRU
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super().__init__(c, model)

        self.fc1 = nn.Linear(1 + self.input_dim, self.dim2)
        self.fc2 = nn.Linear(self.dim2, self.hidden_dim)
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.fc3 = nn.Linear(self.hidden_dim, self.dim2)
        self.hidden_to_params = nn.Linear(self.dim2, self.output_dim)

        self.h0 = nn.Parameter(torch.zeros(1, self.hidden_dim))

    def forward(
        self,
        input: torch.Tensor = None,
        hn: torch.Tensor = None,
        cultivars: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input.ndim == 2:
            input = input.unsqueeze(1)
        gru_input = torch.concatenate((cultivars, input), dim=-1)
        x = self.fc1(gru_input)
        x = self.fc2(x)
        _, hn = self.rnn(x, hn)
        out = F.relu(hn)
        out = F.relu(self.fc3(out))
        params = self.hidden_to_params(out).squeeze(0)

        return params, hn


class GHCNDeepEmbeddingGRU(BaseModule):
    """
    GHCNDeepRNN Embedding GRU
    Takes one hot embedding of cultivar and passes through RNN
    to create a regression prediction of the output
    Assumes 3 outputs (LTE10, LTE50, LTE90) on different prediction heads
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super().__init__(c, model)

        self.embedding_layer = nn.Embedding(len(CROP_NAMES[c.DataConfig.dtype]), self.input_dim)
        self.fc1 = nn.Linear(self.embed_dim, self.dim2)
        self.fc2 = nn.Linear(self.dim2, self.hidden_dim)
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.fc3 = nn.Linear(self.hidden_dim, self.dim2)
        self.hidden_to_output_1 = nn.Linear(self.dim2, int(self.output_dim / 3))
        self.hidden_to_output_2 = nn.Linear(self.dim2, int(self.output_dim / 3))
        self.hidden_to_output_3 = nn.Linear(self.dim2, int(self.output_dim / 3))

        self.h0 = nn.Parameter(torch.zeros(1, self.hidden_dim))

    def forward(
        self, input: torch.Tensor = None, hn: torch.Tensor = None, cultivars: torch.Tensor = None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.embedding_layer(cultivars.flatten().to(torch.long))
        if input.ndim == 2:
            input = input.unsqueeze(1)
        embed = embed.unsqueeze(1).repeat(1, input.shape[1], 1)
        gru_input = self.embed_op(embed, input)

        x = F.relu(self.fc1(gru_input))
        x = F.relu(self.fc2(x))
        _, hn = self.rnn(x, hn)
        out = F.relu(self.fc3(hn))
        output1 = self.hidden_to_output_1(out).squeeze(1)
        output2 = self.hidden_to_output_2(out).squeeze(1)
        output3 = self.hidden_to_output_3(out).squeeze(1)

        return torch.cat((output1, output2, output3), axis=-1), hn


class FFTempResponse(BaseModule):
    """
    Hybrid model for replacing the temperature response function
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super().__init__(c, model)

        self.temp_response = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=self.output_dim),
        )

        self.model_params = nn.Parameter(torch.rand(len(c.params)) - 1)

    def forward(self, input: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning temperature response and parameters
        """
        return self.temp_response(input), self.model_params


class ParamModel(BaseModule):
    """
    Model for performing gradient descent on the parameters
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super().__init__(c, model)
        self.model_params = nn.Parameter(torch.rand(len(c.params)) - 1)

    def forward(self, input: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning temperature response and parameters
        """
        return self.model_params, None


class FoundationParamModel(BaseModule):
    """
    Model for performing gradient descent on the parameters
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super().__init__(c, model)
        self.embed_dim = self.embed_dim - self.input_dim
        edim = self.input_dim if c.DConfig.embed_size is None else c.DConfig.embed_size
        self._build_text_projectors(edim)
        self._build_hierarchy_embeddings(c, embed_size_aware=True)
        self.fc1 = nn.Linear(self.embed_dim, self.dim2)
        self.fc2 = nn.Linear(self.dim2, self.dim2)
        self.hidden_to_params = nn.Linear(self.dim2, self.output_dim)

    def forward(
        self,
        input: torch.Tensor = None,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        text_embedder: nn.Module = None,
        map_idx: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if input.ndim == 2:
            input = input.unsqueeze(1)

        direct_mask = map_idx == 0
        text_mask = map_idx == 1

        reg_embed = torch.zeros(
            size=(
                input.shape[0],
                self.input_dim if self.config.DConfig.embed_size is None else self.config.DConfig.embed_size,
            ),
            device=input.device,
        )
        stat_embed = torch.zeros(
            size=(
                input.shape[0],
                self.input_dim if self.config.DConfig.embed_size is None else self.config.DConfig.embed_size,
            ),
            device=input.device,
        )
        site_embed = torch.zeros(
            size=(
                input.shape[0],
                self.input_dim if self.config.DConfig.embed_size is None else self.config.DConfig.embed_size,
            ),
            device=input.device,
        )
        cult_embed = torch.zeros(
            size=(
                input.shape[0],
                self.input_dim if self.config.DConfig.embed_size is None else self.config.DConfig.embed_size,
            ),
            device=input.device,
        )

        if text_mask.sum() != 0:
            reg_embed_text, stat_embed_text, site_embed_text, cult_embed_text = self.forward_embed_text(
                cultivars[text_mask],
                regions[text_mask],
                stations[text_mask],
                sites[text_mask],
                text_embedder=text_embedder,
            )
            reg_embed[text_mask.flatten()] = reg_embed_text
            stat_embed[text_mask.flatten()] = stat_embed_text
            site_embed[text_mask.flatten()] = site_embed_text
            cult_embed[text_mask.flatten()] = cult_embed_text

        if direct_mask.sum() != 0:
            reg_embed_direct, stat_embed_direct, site_embed_direct, cult_embed_direct = self.forward_embed_direct(
                cultivars[direct_mask], regions[direct_mask], stations[direct_mask], sites[direct_mask]
            )

            reg_embed[direct_mask.flatten()] = reg_embed_direct
            stat_embed[direct_mask.flatten()] = stat_embed_direct
            site_embed[direct_mask.flatten()] = site_embed_direct
            cult_embed[direct_mask.flatten()] = cult_embed_direct

        cult_embed = cult_embed.unsqueeze(1).repeat(1, input.shape[1], 1)
        site_embed = site_embed.unsqueeze(1).repeat(1, input.shape[1], 1)
        stat_embed = stat_embed.unsqueeze(1).repeat(1, input.shape[1], 1)
        reg_embed = reg_embed.unsqueeze(1).repeat(1, input.shape[1], 1)

        gru_input = self.apply_embed_op(cult_embed, site_embed, stat_embed, reg_embed, ids_only=True)

        x = self.fc1(gru_input).relu()
        x = self.fc2(x).relu()
        output = self.hidden_to_params(x).squeeze(-1)
        return output, None


class FoundationEmbeddingGRU(BaseModule):
    """
    Multi Task GRU
    """

    param_style = False

    def _id_embed_dim(self):
        if self.param_style:
            return self.input_dim
        return self.input_dim if self.config.DConfig.embed_size is None else self.config.DConfig.embed_size

    def _run_trunk(self, gru_input, hn):
        if self.param_style:
            x = self.fc1(gru_input)
            x = self.fc2(x)
            gru_out, hn = self.rnn(x, hn)
            gru_out = self.fc3(gru_out.relu()).relu()
        else:
            x = self.fc1(gru_input).relu()
            x = self.fc2(x).relu()
            gru_out, hn = self.rnn(x, hn)
            gru_out = self.fc3(gru_out).relu()
        return self.hidden_to_output(gru_out), hn

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super().__init__(c, model)

        self.weather_output_dim = 128
        edim = self.input_dim if c.DConfig.embed_size is None else c.DConfig.embed_size
        self._build_hierarchy_embeddings(c, embed_size_aware=True)
        self._build_text_projectors(edim)

        self.fc1 = nn.Linear(self.embed_dim, self.dim2)
        self.fc2 = nn.Linear(self.dim2, self.hidden_dim)
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.fc3 = nn.Linear(self.hidden_dim, self.dim2)
        self.hidden_to_output = nn.Linear(self.dim2, self.output_dim)
        self.h0 = nn.Parameter(torch.zeros(1, self.hidden_dim))

        self.transfer_weather_rnn = nn.GRU(
            self.input_dim if "Deep" in c.DConfig.type else (self.input_dim - len(c.PConfig.output_vars)),
            self.weather_output_dim,
            batch_first=True,
        )
        self.transfer_ch_rnn = nn.GRU(
            self.output_dim if "Deep" in c.DConfig.type else len(c.PConfig.output_vars),
            self.weather_output_dim,
            batch_first=True,
        )
        self.transfer_pool_attn = nn.Linear(self.weather_output_dim, 1)
        self.transfer_embed_attn = nn.Linear(self.weather_output_dim, self.embed_dim - self.input_dim)

    def get_embeds(
        self,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        text_embedder: nn.Module = None,
        map_idx: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        direct_mask = map_idx == 0
        text_mask = map_idx == 1
        transfer_mask = map_idx == 2

        reg_embed = torch.zeros(
            size=(
                cultivars.shape[0],
                self.input_dim if self.config.DConfig.embed_size is None else self.config.DConfig.embed_size,
            ),
            device=cultivars.device,
        )
        stat_embed = torch.zeros(
            size=(
                cultivars.shape[0],
                self.input_dim if self.config.DConfig.embed_size is None else self.config.DConfig.embed_size,
            ),
            device=cultivars.device,
        )
        site_embed = torch.zeros(
            size=(
                cultivars.shape[0],
                self.input_dim if self.config.DConfig.embed_size is None else self.config.DConfig.embed_size,
            ),
            device=cultivars.device,
        )
        cult_embed = torch.zeros(
            size=(
                cultivars.shape[0],
                self.input_dim if self.config.DConfig.embed_size is None else self.config.DConfig.embed_size,
            ),
            device=cultivars.device,
        )

        if text_mask.sum() != 0:
            reg_embed_text, stat_embed_text, site_embed_text, cult_embed_text = self.forward_embed_text(
                cultivars[text_mask],
                regions[text_mask],
                stations[text_mask],
                sites[text_mask],
                text_embedder=text_embedder,
            )
            reg_embed[text_mask.flatten()] = reg_embed_text
            stat_embed[text_mask.flatten()] = stat_embed_text
            site_embed[text_mask.flatten()] = site_embed_text
            cult_embed[text_mask.flatten()] = cult_embed_text

        if direct_mask.sum() != 0:
            reg_embed_direct, stat_embed_direct, site_embed_direct, cult_embed_direct = self.forward_embed_direct(
                cultivars[direct_mask], regions[direct_mask], stations[direct_mask], sites[direct_mask]
            )

            reg_embed[direct_mask.flatten()] = reg_embed_direct
            stat_embed[direct_mask.flatten()] = stat_embed_direct
            site_embed[direct_mask.flatten()] = site_embed_direct
            cult_embed[direct_mask.flatten()] = cult_embed_direct

        return reg_embed, stat_embed, site_embed, cult_embed

    def forward_with_embed(
        self,
        input: torch.Tensor = None,
        hn: torch.Tensor = None,
        reg_embed: torch.Tensor = None,
        stat_embed: torch.Tensor = None,
        site_embed: torch.Tensor = None,
        cult_embed: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if input.ndim == 2:
            input = input.unsqueeze(1)

        cult_embed = cult_embed.unsqueeze(1).repeat(1, input.shape[1], 1)
        site_embed = site_embed.unsqueeze(1).repeat(1, input.shape[1], 1)
        stat_embed = stat_embed.unsqueeze(1).repeat(1, input.shape[1], 1)
        reg_embed = reg_embed.unsqueeze(1).repeat(1, input.shape[1], 1)

        gru_input = self.apply_embed_op(cult_embed, site_embed, stat_embed, reg_embed, input)

        output, hn = self._run_trunk(gru_input, hn)

        return output, hn

    def forward(
        self,
        input: torch.Tensor = None,
        hn: torch.Tensor = None,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        text_embedder: nn.Module = None,
        map_idx: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if input.ndim == 2:
            input = input.unsqueeze(1)

        direct_mask = map_idx == 0
        text_mask = map_idx == 1

        embed_dim = self._id_embed_dim()
        reg_embed = torch.zeros(size=(input.shape[0], embed_dim), device=input.device)
        stat_embed = torch.zeros(size=(input.shape[0], embed_dim), device=input.device)
        site_embed = torch.zeros(size=(input.shape[0], embed_dim), device=input.device)
        cult_embed = torch.zeros(size=(input.shape[0], embed_dim), device=input.device)

        if text_mask.sum() != 0:
            reg_embed_text, stat_embed_text, site_embed_text, cult_embed_text = self.forward_embed_text(
                cultivars[text_mask],
                regions[text_mask],
                stations[text_mask],
                sites[text_mask],
                text_embedder=text_embedder,
            )
            reg_embed[text_mask.flatten()] = reg_embed_text
            stat_embed[text_mask.flatten()] = stat_embed_text
            site_embed[text_mask.flatten()] = site_embed_text
            cult_embed[text_mask.flatten()] = cult_embed_text

        if direct_mask.sum() != 0:
            reg_embed_direct, stat_embed_direct, site_embed_direct, cult_embed_direct = self.forward_embed_direct(
                cultivars[direct_mask], regions[direct_mask], stations[direct_mask], sites[direct_mask]
            )

            reg_embed[direct_mask.flatten()] = reg_embed_direct
            stat_embed[direct_mask.flatten()] = stat_embed_direct
            site_embed[direct_mask.flatten()] = site_embed_direct
            cult_embed[direct_mask.flatten()] = cult_embed_direct

        cult_embed = cult_embed.unsqueeze(1).repeat(1, input.shape[1], 1)
        site_embed = site_embed.unsqueeze(1).repeat(1, input.shape[1], 1)
        stat_embed = stat_embed.unsqueeze(1).repeat(1, input.shape[1], 1)
        reg_embed = reg_embed.unsqueeze(1).repeat(1, input.shape[1], 1)

        gru_input = self.apply_embed_op(cult_embed, site_embed, stat_embed, reg_embed, input)

        output, hn = self._run_trunk(gru_input, hn)

        return output, hn

    def forward_embed_adapt(
        self,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        calibrator=None,
        **kwargs,
    ):

        transfer_inds = torch.arange(cultivars.shape[0])

        b_all_inds = torch.cat((regions, stations, sites, cultivars), dim=1)[transfer_inds]
        all_inds = calibrator.all_inds["train"]
        adapt_inds = []
        map_idx = []
        for i, ind in enumerate(b_all_inds):
            ids = torch.argwhere((ind == all_inds).all(axis=1)).flatten()
            idx = ids[
                torch.randperm(ids.shape[0])[: min(ids.shape[0] + 1, calibrator.config.DConfig.num_transfer_years + 1)]
            ]
            adapt_inds.append(idx)
            map_idx.append(torch.ones_like(idx, device=idx.device) * i)
        adapt_inds = torch.cat(adapt_inds)
        map_idx = torch.cat(map_idx)
        embed = self.compute_embed(
            calibrator.data["train"][adapt_inds],
            ch=calibrator.val["train"][adapt_inds],
            map_idx=map_idx,
            bsize=(transfer_inds).sum().item(),
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

        output, hn = self._run_trunk(gru_input, hn)

        return output, hn

    def compute_embed(
        self,
        weather_input: torch.Tensor = None,
        ch: torch.Tensor = None,
        map_idx: torch.Tensor = None,
        bsize: int = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._weather_ch_cross_attn_embed(weather_input, ch, sine_day=False, map_idx=map_idx, bsize=bsize)


class FoundationParamEmbeddingGRU(FoundationEmbeddingGRU):
    """Param-coupled foundation GRU; shares embed paths, uses param-style trunk."""

    param_style = True
