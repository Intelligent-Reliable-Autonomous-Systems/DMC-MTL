"""Base class for Param RNN

Written by Will Solow, 2025"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig
from train_algs.base.util import set_embedding_op



class BaseModule(nn.Module):

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super().__init__()
        self.hidden_dim = c.DConfig.hidden_dim
        self.dim2 = int(self.hidden_dim / 2)
        self.input_dim = model.get_input_dim(c)
        self.output_dim = model.get_output_dim(c)
        self.embed_dim = set_embedding_op(self, c.DConfig.embed_op)

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
            self.param_heads = nn.ModuleList([nn.Linear(self.dim2, self.output_dim) for _ in range(model.num_cultivars)])

        self.h0 = nn.Parameter(torch.zeros(1, self.hidden_dim))
        self.init_state = self.h0

    def forward(
        self, input: torch.Tensor = None, hn: torch.Tensor = None, cultivars:torch.Tensor=None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.fc1(input)
        x = self.fc2(x)
        gru_out, hn = self.rnn(x.unsqueeze(1), hn)
        gru_out = F.relu(gru_out)
        gru_out = F.relu(self.fc3(gru_out)).squeeze(1)

        params = torch.empty(input.size(0), self.param_heads[0].out_features).to(input.device)
        if cultivars is None:
            for i in range(input.size(0)):
                params[i] = self.param_heads[0](gru_out[i])
        else:
            for i in range(input.size(0)):
                params[i] = self.param_heads[int(cultivars[i][0])](gru_out[i])

        return params, hn


class EmbeddingFCGRU(BaseModule):
    """
    Multi Task GRU
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:

        super(EmbeddingFCGRU, self).__init__(c, model)

        self.embedding_layer = nn.Embedding(model.num_cultivars, self.input_dim)
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

        embed = self.embedding_layer(cultivars.flatten().to(torch.long))
        gru_input = self.embed_op(embed, input)
        x = self.fc1(gru_input)
        x = self.fc2(x)
        gru_out, hn = self.rnn(x.unsqueeze(1), hn)
        gru_out = F.relu(gru_out)
        gru_out = F.relu(self.fc3(gru_out))
        params = self.hidden_to_params(gru_out).squeeze(1)

        return params, hn


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

        gru_input = torch.concatenate((cultivars, input), dim=-1)
        x = self.fc1(gru_input)
        x = self.fc2(x)
        gru_out, hn = self.rnn(x.unsqueeze(1), hn)
        gru_out = F.relu(gru_out)
        gru_out = F.relu(self.fc3(gru_out))
        params = self.hidden_to_params(gru_out).squeeze(1)

        return params, hn


class DeepEmbeddingGRU(BaseModule):
    """
    DeepRNN Embedding GRU
    Takes one hot embedding of cultivar and passes through RNN
    to create a regression prediction of the output
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:

        super(DeepEmbeddingGRU, self).__init__(c, model)

        self.embedding_layer = nn.Embedding(model.num_cultivars, self.input_dim)
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
        gru_input = gru_input = self.embed_op(embed, input)
        x = F.relu(self.fc1(gru_input))
        x = F.relu(self.fc2(x))
        gru_out, hn = self.rnn(x.unsqueeze(1), hn)
        gru_out = F.relu(self.fc3(gru_out))
        output = self.hidden_to_output(gru_out).squeeze(1)

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

        self.embedding_layer = nn.Embedding(model.num_cultivars, self.input_dim)
        self.fc1 = nn.Linear(self.embed_dim, self.dim2)
        self.fc2 = nn.Linear(self.dim2, self.hidden_dim)
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.fc3 = nn.Linear(self.hidden_dim, self.dim2)
        self.hidden_to_output_1 = nn.Linear(self.dim2, int(self.output_dim/3))
        self.hidden_to_output_2 = nn.Linear(self.dim2, int(self.output_dim/3))
        self.hidden_to_output_3 = nn.Linear(self.dim2, int(self.output_dim/3))

        self.h0 = nn.Parameter(torch.zeros(1, self.hidden_dim))

    def forward(
        self, input: torch.Tensor = None, hn: torch.Tensor = None, cultivars: torch.Tensor = None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:

        embed = self.embedding_layer(cultivars.flatten().to(torch.long))
        gru_input = self.embed_op(embed, input)

        x = F.relu(self.fc1(gru_input))
        x = F.relu(self.fc2(x))
        gru_out, hn = self.rnn(x.unsqueeze(1), hn)
        gru_out = F.relu(self.fc3(gru_out))
        output1 = self.hidden_to_output_1(gru_out).squeeze(1)
        output2 = self.hidden_to_output_2(gru_out).squeeze(1)
        output3 = self.hidden_to_output_3(gru_out).squeeze(1)

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
