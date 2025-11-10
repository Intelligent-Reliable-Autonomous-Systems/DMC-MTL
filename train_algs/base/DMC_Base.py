"""Base class for Param RNN

Written by Will Solow, 2025"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig
from train_algs.base.util import set_embedding_op
from model_engine.util import CROP_NAMES


class BaseModule(nn.Module):

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super().__init__()
        self.hidden_dim = c.DConfig.hidden_dim
        self.dim2 = int(self.hidden_dim / 2)
        self.input_dim = model.get_input_dim(c)
        self.output_dim = model.get_output_dim(c)
        self.embed_dim = set_embedding_op(self, c.DConfig.embed_op)

        orig = torch.unique(torch.concatenate(list(model.cultivars.values()), axis=0)).to(torch.int)
        self.mapping = torch.zeros((int(orig.max()) + 1,)).to(torch.int).to(model.device)
        self.mapping[orig] = torch.arange(len(orig)).to(torch.int).to(model.device)

    def get_init_state(self, batch_size: int = 1) -> torch.Tensor:

        return self.h0.repeat(1, batch_size, 1)

    def reinit_weights(self, m):
        """
        Reinitilize weights for transformer layers"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


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
                [nn.Linear(self.dim2, self.output_dim) for _ in range(len(CROP_NAMES[c.dtype]))]
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

        self.embedding_layer = nn.Embedding(len(CROP_NAMES[c.dtype]), self.input_dim)
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
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.embedding_layer(self.mapping[cultivars.flatten().to(torch.int)])
        
        if input.ndim == 2:
            input = input.unsqueeze(1)

        embed = embed.unsqueeze(1).expand_as(input)
        gru_input = self.embed_op(embed, input)
        x = self.fc1(gru_input)
        x = self.fc2(x)
        _, hn = self.rnn(x, hn)
        out = F.relu(hn)
        out = F.relu(self.fc3(out))
        params = self.hidden_to_params(out).squeeze(0)

        return params, hn


class EmbeddingFCFF(BaseModule):
    """
    Multi Task FF
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:

        super(EmbeddingFCFF, self).__init__(c, model)

        self.embedding_layer = nn.Embedding(len(CROP_NAMES[c.dtype]), self.input_dim)
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
        embed = self.embedding_layer(self.mapping[cultivars.flatten().to(torch.int)])
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


class DeepEmbeddingGRU(BaseModule):
    """
    DeepRNN Embedding GRU
    Takes one hot embedding of cultivar and passes through RNN
    to create a regression prediction of the output
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:

        super(DeepEmbeddingGRU, self).__init__(c, model)

        self.embedding_layer = nn.Embedding(len(CROP_NAMES[c.dtype]), self.input_dim)
        self.fc1 = nn.Linear(self.embed_dim, self.dim2)
        self.fc2 = nn.Linear(self.dim2, self.hidden_dim)
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.fc3 = nn.Linear(self.hidden_dim, self.dim2)
        self.hidden_to_output = nn.Linear(self.dim2, self.output_dim)

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
        output = self.hidden_to_output(out).squeeze(0)

        return output, hn


class GHCNDeepEmbeddingGRU(BaseModule):
    """
    GHCNDeepRNN Embedding GRU
    Takes one hot embedding of cultivar and passes through RNN
    to create a regression prediction of the output
    Assumes 3 outputs (LTE10, LTE50, LTE90) on different prediction heads
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:

        super(GHCNDeepEmbeddingGRU, self).__init__(c, model)

        self.embedding_layer = nn.Embedding(len(CROP_NAMES[c.dtype]), self.input_dim)
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
        return None, self.model_params
