"""Base class for GP architectures (DeepEmbedGP).

Written by Will Solow, 2025"""

import gpytorch
import torch
import torch.nn as nn
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from omegaconf import DictConfig

from assets import (
    CROP_NAMES,
    REGIONS,
    SITES,
    STATIONS,
)
from train_algs.base.DMC_Base import BaseModule


class GPRegressionHead(ApproximateGP):
    """Gaussian Process prediction head"""

    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DeepEmbedGP(BaseModule):
    """
    DeepRNN Embedding GRU
    Takes one hot embedding of cultivar and passes through RNN
    to create a regression prediction of the output
    """

    def __init__(self, c: DictConfig, model: nn.Module) -> None:
        super(DeepEmbedGP, self).__init__(c, model)

        self.num_inducing = 100
        self.latent_dim = 16

        self.cult_embedding_layer = nn.Embedding(len(CROP_NAMES[c.DataConfig.dtype]), self.input_dim)
        self.reg_embedding_layer = nn.Embedding(len(REGIONS), self.input_dim) if self.synth_data is None else None
        self.stat_embedding_layer = nn.Embedding(len(STATIONS), self.input_dim) if self.synth_data is None else None
        self.site_embedding_layer = nn.Embedding(len(SITES), self.input_dim) if self.synth_data is None else None

        self.fc1 = nn.Sequential(
            nn.Linear(self.embed_dim, self.dim2), nn.ReLU(), nn.Linear(self.dim2, self.hidden_dim), nn.ReLU()
        )
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.fc2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.dim2), nn.ReLU(), nn.Linear(self.dim2, self.latent_dim)
        )
        self.hidden_to_output = nn.Linear(self.latent_dim, self.output_dim)

        self.h0 = nn.Parameter(torch.zeros(1, self.hidden_dim))

        self.inducing_points = torch.randn(self.num_inducing, self.latent_dim)
        self.gp_head = GPRegressionHead(self.inducing_points)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

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

        reg_embed, stat_embed, site_embed, cult_embed = self.forward_embed(
            cultivars=cultivars, regions=regions, stations=stations, sites=sites
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
        gru_out, hn = self.rnn(x, hn)
        features = self.fc2(gru_out)
        output = self.hidden_to_output(features)
        gp_output = self.gp_head(features)

        return gp_output, hn

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
