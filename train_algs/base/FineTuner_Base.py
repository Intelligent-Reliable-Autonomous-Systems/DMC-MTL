"""
Base class for Finetuning param prediction

Written by Will Solow, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer
from train_algs.base.base import TLSTM
from omegaconf import DictConfig


class BaseFTModule(nn.Module):

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super().__init__()
        self.hidden_dim = c.RTMCConfig.hidden_dim
        self.dim2 = int(self.hidden_dim / 2)
        self.input_dim = model.get_ft_input_dim(c)
        self.bias = c.RTMCConfig.bias
        self.output_dim = model.get_output_dim(c)
        self.num_heads = c.RTMCConfig.num_heads
        self.num_layers = c.RTMCConfig.num_layers
        self.attn_pool = c.RTMCConfig.attn_pool

    def get_init_state(self, batch_size: int = 1) -> torch.Tensor:

        return self.h0.repeat(1, batch_size, 1)

    def reinit_weights(self, m):
        """
        Reinitilize transformer layer weights
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class CausalAttentionPooling(nn.Module):
    """
    Casual attnetion pooling from transformer output
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, L, D = x.shape

        query_expanded = self.query.unsqueeze(0).unsqueeze(0).expand(B, L, D)
        scores = torch.matmul(x, query_expanded.transpose(1, 2))
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        pooled = torch.matmul(attn_weights, x)

        return pooled


class CausalTransformerFCTuner(BaseFTModule):
    """
    Transformer error encoding using single error signal
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:

        super().__init__(c, model)

        self.embedding_layer = nn.Embedding(model.num_cultivars, self.input_dim)
        self.error_embedding_layer = nn.Linear(1, self.input_dim, bias=self.bias)
        self.fc1 = nn.Linear(2 * self.input_dim, self.hidden_dim, bias=self.bias)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=self.num_heads, batch_first=True, bias=self.bias, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.transformer.apply(self.reinit_weights)
        self.hidden_to_output = nn.Linear(self.hidden_dim, self.output_dim, bias=self.bias)
        self.attn_pooling = CausalAttentionPooling(self.hidden_dim)

    def forward(
        self, error: torch.Tensor = None, cultivars: torch.Tensor = None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, seq_len, _ = error.shape
        mask = Transformer.generate_square_subsequent_mask(seq_len, device=error.device)

        embed = self.embedding_layer(cultivars.flatten().to(torch.long)).unsqueeze(1).expand_as(error)
        error_embed = self.error_embedding_layer(error)
        embed = torch.where(error == 0.0, 0.0, embed)
        x = torch.concatenate((embed, error_embed), dim=-1)
        x = self.fc1(x)

        x = self.transformer(x, mask=mask)
        x = self.attn_pooling(x)

        return self.hidden_to_output(x)


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
            self.h0 = torch.zeros(1, self.hidden_dim)

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
            self.h0 = torch.zeros(1, self.hidden_dim)

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
