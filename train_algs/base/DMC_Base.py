"""Base class for Param RNN

Written by Will Solow, 2025"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from assets import (
    CROP_NAMES,
    CULT_STR,
    CULT_STR_VERB,
    REGION_STR,
    REGION_STR_VERB,
    REGIONS,
    SITE_STR,
    SITE_STR_VERB,
    SITES,
    STATION_STR,
    STATION_STR_VERB,
    STATIONS,
)
from train_algs.base.util import set_embedding_op


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


class FCGRU(BaseModule):
    """
    Single Task GRU
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super(FCGRU, self).__init__(c, model)

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
        super(EmbeddingFCGRU, self).__init__(c, model)

        self.cult_embedding_layer = nn.Embedding(len(CROP_NAMES[c.DataConfig.dtype]), self.input_dim)
        self.reg_embedding_layer = nn.Embedding(len(REGIONS), self.input_dim) if self.synth_data is None else None
        self.stat_embedding_layer = nn.Embedding(len(STATIONS), self.input_dim) if self.synth_data is None else None
        self.site_embedding_layer = nn.Embedding(len(SITES), self.input_dim) if self.synth_data is None else None

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

        return reg_embed, stat_embed, site_embed, cult_embed


class EmbeddingTextGRU(BaseModule):
    """
    Multi Task GRU
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super(EmbeddingTextGRU, self).__init__(c, model)

        self.cult_embedder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_dim),
        )
        self.site_embedder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_dim),
        )

        self.stat_embedder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_dim),
        )

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

        with torch.no_grad():
            cult_embed_text = text_embedder.encode(
                CULT_STR[self.cult_mapping[cultivars.flatten().to(torch.int)].cpu().numpy()],
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=input.device,
            ).clone()
            site_embed_text = text_embedder.encode(
                SITE_STR[self.site_mapping[sites.flatten().to(torch.int)].cpu().numpy()],
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=input.device,
            ).clone()
            stat_embed_text = text_embedder.encode(
                STATION_STR[self.stat_mapping[stations.flatten().to(torch.int)].cpu().numpy()],
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=input.device,
            ).clone()
            reg_embed_text = text_embedder.encode(
                REGION_STR[self.reg_mapping[regions.flatten().to(torch.int)].cpu().numpy()],
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=input.device,
            ).clone()
        cult_embed = self.cult_embedder(site_embed_text)
        site_embed = self.site_embedder(cult_embed_text)
        stat_embed = self.stat_embedder(stat_embed_text)
        reg_embed = self.stat_embedder(reg_embed_text)

        cult_embed = cult_embed.unsqueeze(1).expand_as(input)
        site_embed = site_embed.unsqueeze(1).expand_as(input)
        stat_embed = stat_embed.unsqueeze(1).expand_as(input)
        reg_embed = reg_embed.unsqueeze(1).expand_as(input)

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

        with torch.no_grad():
            cult_embed_text = text_embedder.encode(
                CULT_STR[self.cult_mapping[cultivars.flatten().to(torch.int)].cpu().numpy()],
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=cultivars.device,
            ).clone()
            site_embed_text = text_embedder.encode(
                SITE_STR[self.site_mapping[sites.flatten().to(torch.int)].cpu().numpy()],
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=cultivars.device,
            ).clone()
            stat_embed_text = text_embedder.encode(
                STATION_STR[self.stat_mapping[stations.flatten().to(torch.int)].cpu().numpy()],
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=cultivars.device,
            ).clone()
            reg_embed_text = text_embedder.encode(
                REGION_STR[self.reg_mapping[regions.flatten().to(torch.int)].cpu().numpy()],
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=cultivars.device,
            ).clone()
        cult_embed = self.cult_embedder(site_embed_text)
        site_embed = self.site_embedder(cult_embed_text)
        stat_embed = self.stat_embedder(stat_embed_text)
        reg_embed = self.stat_embedder(reg_embed_text)

        return reg_embed, stat_embed, site_embed, cult_embed


class EmbeddingVerboseTextGRU(BaseModule):
    """
    Multi Task GRU
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super(EmbeddingVerboseTextGRU, self).__init__(c, model)

        self.cult_embedder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_dim),
        )
        self.site_embedder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_dim),
        )

        self.stat_embedder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_dim),
        )

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

        with torch.no_grad():
            cult_embed_text = text_embedder.encode(
                CULT_STR_VERB[self.cult_mapping[cultivars.flatten().to(torch.int)].cpu().numpy()],
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=input.device,
            ).clone()
            site_embed_text = text_embedder.encode(
                SITE_STR_VERB[self.site_mapping[sites.flatten().to(torch.int)].cpu().numpy()],
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=input.device,
            ).clone()
            stat_embed_text = text_embedder.encode(
                STATION_STR_VERB[self.stat_mapping[stations.flatten().to(torch.int)].cpu().numpy()],
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=input.device,
            ).clone()
            reg_embed_text = text_embedder.encode(
                REGION_STR_VERB[self.reg_mapping[regions.flatten().to(torch.int)].cpu().numpy()],
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=input.device,
            ).clone()
        cult_embed = self.cult_embedder(site_embed_text)
        site_embed = self.site_embedder(cult_embed_text)
        stat_embed = self.stat_embedder(stat_embed_text)
        reg_embed = self.stat_embedder(reg_embed_text)

        cult_embed = cult_embed.unsqueeze(1).expand_as(input)
        site_embed = site_embed.unsqueeze(1).expand_as(input)
        stat_embed = stat_embed.unsqueeze(1).expand_as(input)
        reg_embed = reg_embed.unsqueeze(1).expand_as(input)

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

        with torch.no_grad():
            cult_embed_text = text_embedder.encode(
                CULT_STR_VERB[self.cult_mapping[cultivars.flatten().to(torch.int)].cpu().numpy()],
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=cultivars.device,
            ).clone()
            site_embed_text = text_embedder.encode(
                SITE_STR_VERB[self.site_mapping[sites.flatten().to(torch.int)].cpu().numpy()],
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=cultivars.device,
            ).clone()
            stat_embed_text = text_embedder.encode(
                STATION_STR_VERB[self.stat_mapping[stations.flatten().to(torch.int)].cpu().numpy()],
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=cultivars.device,
            ).clone()
            reg_embed_text = text_embedder.encode(
                REGION_STR_VERB[self.reg_mapping[regions.flatten().to(torch.int)].cpu().numpy()],
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=cultivars.device,
            ).clone()
        cult_embed = self.cult_embedder(site_embed_text)
        site_embed = self.site_embedder(cult_embed_text)
        stat_embed = self.stat_embedder(stat_embed_text)
        reg_embed = self.stat_embedder(reg_embed_text)

        return reg_embed, stat_embed, site_embed, cult_embed


class DeepEmbeddingGRU(BaseModule):
    """
    DeepRNN Embedding GRU
    Takes one hot embedding of cultivar and passes through RNN
    to create a regression prediction of the output
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super(DeepEmbeddingGRU, self).__init__(c, model)

        self.cult_embedding_layer = nn.Embedding(len(CROP_NAMES[c.DataConfig.dtype]), self.input_dim)
        self.reg_embedding_layer = nn.Embedding(len(REGIONS), self.input_dim) if self.synth_data is None else None
        self.stat_embedding_layer = nn.Embedding(len(STATIONS), self.input_dim) if self.synth_data is None else None
        self.site_embedding_layer = nn.Embedding(len(SITES), self.input_dim) if self.synth_data is None else None

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

        return reg_embed, stat_embed, site_embed, cult_embed


class EmbeddingFCFF(BaseModule):
    """
    Multi Task FF
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super(EmbeddingFCFF, self).__init__(c, model)

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
        super(OneHotEmbeddingFCGRU, self).__init__(c, model)

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
        super(GHCNDeepEmbeddingGRU, self).__init__(c, model)

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
        embed = embed.unsqueeze(1).expand_as(input)
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
        super(FFTempResponse, self).__init__(c, model)

        self.temp_response = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=self.output_dim),
        )

        self.model_params = nn.Parameter(torch.rand((len(c.params))) - 1)

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
        super(ParamModel, self).__init__(c, model)
        self.model_params = nn.Parameter(torch.rand((len(c.params))) - 1)

    def forward(self, input: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning temperature response and parameters
        """
        return self.model_params, None


class DeepEmbeddingMuSigmaGRU(BaseModule):
    """
    DeepRNN Embedding GRU
    Takes one hot embedding of cultivar and passes through RNN
    to create a regression prediction of the output
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super(DeepEmbeddingMuSigmaGRU, self).__init__(c, model)

        self.cult_embedding_layer = nn.Embedding(len(CROP_NAMES[c.DataConfig.dtype]), self.input_dim)
        self.reg_embedding_layer = nn.Embedding(len(REGIONS), self.input_dim) if self.synth_data is None else None
        self.stat_embedding_layer = nn.Embedding(len(STATIONS), self.input_dim) if self.synth_data is None else None
        self.site_embedding_layer = nn.Embedding(len(SITES), self.input_dim) if self.synth_data is None else None

        self.fc1 = nn.Linear(self.embed_dim, self.dim2)
        self.fc2 = nn.Linear(self.dim2, self.hidden_dim)
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.fc3 = nn.Linear(self.hidden_dim, self.dim2)
        self.hidden_to_mu = nn.Linear(self.dim2, self.output_dim)
        self.hidden_to_log_sigma = nn.Linear(self.dim2, self.output_dim)
        self.hidden_to_log_sigma.bias.data.fill_(-3)

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
        mu = self.hidden_to_mu(gru_out)
        sigma = torch.exp(self.hidden_to_log_sigma(gru_out))

        return torch.cat((mu, sigma), dim=-1), hn


class FoundationEmbeddingGRU(BaseModule):
    """
    Multi Task GRU
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super(FoundationEmbeddingGRU, self).__init__(c, model)

        self.weather_output_dim = 128

        self.cult_embedder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_dim),
        )
        self.site_embedder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_dim),
        )

        self.stat_embedder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_dim),
        )

        self.cult_embedding_layer = nn.Embedding(len(CROP_NAMES[c.DataConfig.dtype]), self.input_dim)
        self.reg_embedding_layer = nn.Embedding(len(REGIONS), self.input_dim) if self.synth_data is None else None
        self.stat_embedding_layer = nn.Embedding(len(STATIONS), self.input_dim) if self.synth_data is None else None
        self.site_embedding_layer = nn.Embedding(len(SITES), self.input_dim) if self.synth_data is None else None

        self.fc1 = nn.Linear(self.embed_dim, self.dim2)
        self.fc2 = nn.Linear(self.dim2, self.hidden_dim)
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.fc3 = nn.Linear(self.hidden_dim, self.dim2)
        self.hidden_to_output = nn.Linear(self.dim2, self.output_dim)

        self.transfer_weather_rnn = nn.GRU(self.input_dim, self.weather_output_dim, batch_first=True)
        self.transfer_ch_rnn = nn.GRU(
            self.output_dim, self.weather_output_dim, batch_first=True
        )  # Sine/Cosine embedding
        self.transfer_pool_attn = nn.Linear(self.weather_output_dim, 1)
        self.transfer_embed_attn = nn.Linear(self.weather_output_dim, self.embed_dim - self.input_dim)

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
        map_idx: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if input.ndim == 2:
            input = input.unsqueeze(1)

        direct_mask = map_idx == 0
        text_mask = map_idx == 1
        transfer_mask = map_idx == 2

        reg_embed = torch.zeros(size=(input.shape[0], self.input_dim), device=input.device)
        stat_embed = torch.zeros(size=(input.shape[0], self.input_dim), device=input.device)
        site_embed = torch.zeros(size=(input.shape[0], self.input_dim), device=input.device)
        cult_embed = torch.zeros(size=(input.shape[0], self.input_dim), device=input.device)

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

        cult_embed = cult_embed.unsqueeze(1).expand_as(input)
        site_embed = site_embed.unsqueeze(1).expand_as(input)
        stat_embed = stat_embed.unsqueeze(1).expand_as(input)
        reg_embed = reg_embed.unsqueeze(1).expand_as(input)

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
        output = self.hidden_to_output(gru_out)

        return output, hn

    def forward_embed_text(
        self,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        text_embedder: nn.Module = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        with torch.no_grad():
            cult_embed_text = text_embedder.encode(
                CULT_STR_VERB[self.cult_mapping[cultivars.flatten().to(torch.int)].cpu().numpy()],
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=cultivars.device,
            ).clone()
            site_embed_text = text_embedder.encode(
                SITE_STR_VERB[self.site_mapping[sites.flatten().to(torch.int)].cpu().numpy()],
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=cultivars.device,
            ).clone()
            stat_embed_text = text_embedder.encode(
                STATION_STR_VERB[self.stat_mapping[stations.flatten().to(torch.int)].cpu().numpy()],
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=cultivars.device,
            ).clone()
            reg_embed_text = text_embedder.encode(
                REGION_STR_VERB[self.reg_mapping[regions.flatten().to(torch.int)].cpu().numpy()],
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=cultivars.device,
            ).clone()
        cult_embed = self.cult_embedder(site_embed_text)
        site_embed = self.site_embedder(cult_embed_text)
        stat_embed = self.stat_embedder(stat_embed_text)
        reg_embed = self.stat_embedder(reg_embed_text)

        return reg_embed, stat_embed, site_embed, cult_embed

    def forward_embed_direct(
        self,
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

        return reg_embed, stat_embed, site_embed, cult_embed

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

        x = self.fc1(gru_input).relu()
        x = self.fc2(x).relu()
        gru_out, hn = self.rnn(x, hn)

        gru_out = self.fc3(gru_out).relu()
        output = self.hidden_to_output(gru_out)

        return output, hn

    def compute_embed(
        self,
        weather_input: torch.Tensor = None,
        ch: torch.Tensor = None,
        map_idx: torch.Tensor = None,
        bsize: int = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if weather_input.ndim == 2:
            weather_input = weather_input.unsqueeze(1)
        if ch.ndim == 2:
            ch = ch.unsqueeze(1)
        w_encode, _ = self.transfer_weather_rnn(weather_input)  # (T,d)

        B, T, d = ch.shape
        # Create mask for CH
        ch_mask = ~torch.isnan(ch)
        ch_clean = torch.where(ch_mask, ch, torch.zeros_like(ch))

        ch_lengths = ch_mask.sum(dim=(1, 2))
        max_len = ch_lengths.max().item()
        if max_len == 0:
            return torch.zeros(B, self.embed_dim, device=ch.device, requires_grad=True)
        ch_arr = torch.zeros(size=(B, max_len), device=ch.device, requires_grad=True)

        ch_arr_list = []
        # Go through batch and pad data, loop avoids breaking gradient flow via masking
        for i in range(B):
            mask_i = ch_mask[i].any(dim=-1)
            valid_data = ch_clean[i][mask_i]
            valid_len = valid_data.shape[0]
            if valid_len < max_len:
                pad = torch.zeros((max_len - valid_len, d), device=ch.device, dtype=ch.dtype)
                valid_data = torch.cat([valid_data, pad], dim=0)
            ch_arr_list.append(valid_data)
        ch_arr = torch.stack(ch_arr_list)

        ch_embed, _ = self.transfer_ch_rnn(ch_arr)  # (ch_max_len, d), padded with zeros

        attn_mask = torch.arange(max_len, device=ch.device).unsqueeze(0) < ch_lengths.unsqueeze(1)
        ch_encode = torch.where(~attn_mask.unsqueeze(-1), torch.zeros(ch_embed.shape[-1], device=ch.device), ch_embed)

        attn_scores = torch.bmm(w_encode, ch_encode.transpose(1, 2))
        attn_scores = attn_scores.masked_fill(
            ~attn_mask.unsqueeze(1),  # (B, 1, h_max_len)
            float("-inf"),
        )
        attn_weights = torch.softmax(attn_scores, dim=2)  # (B, T, ch_max_len)
        ch_aligned = torch.bmm(attn_weights, ch_encode)  # (B, T, d)

        attn_weights = self.transfer_pool_attn(ch_aligned).softmax(dim=1)  # (B, T, 1)
        attn_pooled = (ch_aligned * attn_weights).sum(dim=1)  # (B, d)
        embed_all = self.transfer_embed_attn(attn_pooled)  # (B,embed_dim)

        embed = torch.zeros((bsize, embed_all.shape[-1]), device=ch.device)
        embed = embed.scatter_reduce(
            0, map_idx.unsqueeze(1).expand(-1, embed_all.shape[-1]), embed_all, reduce="mean", include_self=False
        )
        return embed
