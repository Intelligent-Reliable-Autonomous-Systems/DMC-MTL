"""
Base class for Finetuning param prediction

Written by Will Solow, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class BaseFTModule(nn.Module):

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super().__init__()
        self.hidden_dim = c.RTMCConfig.hidden_dim
        self.dim2 = int(self.hidden_dim / 2)
        self.input_dim = model.get_ft_input_dim(c)
        self.bias = c.RTMCConfig.bias
        self.output_dim = model.get_output_dim(c)

    def get_init_state(self, batch_size: int = 1) -> torch.Tensor:

        return self.h0.repeat(1, batch_size, 1)


class EmbedErrorFCTuner(BaseFTModule):
    """
    GRU error encoder using single error signal
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:

        super(EmbedErrorFCTuner, self).__init__(c, model)

        self.embedding_layer = nn.Embedding(model.num_cultivars, self.input_dim)
        self.fc1 = nn.Linear(2 * self.input_dim, self.dim2, bias=self.bias)
        self.fc2 = nn.Linear(self.dim2, self.hidden_dim, bias=self.bias)
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True, bias=self.bias)
        self.fc3 = nn.Linear(self.hidden_dim, self.dim2, bias=self.bias)
        self.hidden_to_params = nn.Linear(self.dim2, self.output_dim, bias=self.bias)

        if self.bias:
            self.h0 = nn.Parameter(torch.zeros(1, self.hidden_dim))
        else:
            self.h0 = torch.zeros(1, self.hidden_dim).to(model.device)

    def forward(
        self, error: torch.Tensor = None, hn: torch.Tensor = None, cultivars: torch.Tensor = None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:

        embed = self.embedding_layer(cultivars.flatten().to(torch.long))
        embed = torch.where(error == 0.0, 0.0, embed)
        gru_input = torch.concatenate((embed, error), dim=-1)
        x = self.fc1(gru_input)
        x = self.fc2(x)
        gru_out, hn = self.rnn(x.unsqueeze(1), hn)
        gru_out = F.relu(gru_out)
        gru_out = F.relu(self.fc3(gru_out))
        params = self.hidden_to_params(gru_out).squeeze(1)

        return params, hn


class ErrorFCTuner(BaseFTModule):
    """
    GRU error embedding without cultivar information
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:

        super(ErrorFCTuner, self).__init__(c, model)

        self.fc1 = nn.Linear(self.input_dim, self.dim2, bias=self.bias)
        self.fc2 = nn.Linear(self.dim2, self.hidden_dim, bias=self.bias)
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True, bias=self.bias)
        self.fc3 = nn.Linear(self.hidden_dim, self.dim2, bias=self.bias)
        self.hidden_to_params = nn.Linear(self.dim2, self.output_dim, bias=self.bias)

        if self.bias:
            self.h0 = nn.Parameter(torch.zeros(1, self.hidden_dim))
        else:
            self.h0 = torch.zeros(1, self.hidden_dim).to(model.device)

    def forward(
        self, error: torch.Tensor = None, hn: torch.Tensor = None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:

        x = self.fc1(error)
        x = self.fc2(x)
        gru_out, hn = self.rnn(x.unsqueeze(1), hn)
        gru_out = F.relu(gru_out)
        gru_out = F.relu(self.fc3(gru_out))
        params = self.hidden_to_params(gru_out).squeeze(1)

        return params, hn
