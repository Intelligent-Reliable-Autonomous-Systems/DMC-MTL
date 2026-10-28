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

from assets import CROP_NAMES
from train_algs.base.rnn import BaseRNN


class DeepTransferRNN(BaseRNN):
    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:
        super().__init__(config, data)

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
        super().__init__(config, data)

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
        super().__init__(config, data)

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
    _FIT_DIRECT_LR = 1e-2
    _FIT_DIRECT_STEPS = 50
    # Cap activation memory during embed GD (eval batches can be large × num_transfer_years).
    _FIT_DIRECT_MICROBATCH = 16

    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:
        super().__init__(config, data)
        # Lazy: e5-large is unused for fit_direct/mean/adapt/direct and is a large VRAM cost.
        self._text_model = None

    @property
    def text_model(self):
        if self._text_model is None:
            from sentence_transformers import SentenceTransformer

            self._text_model = SentenceTransformer("intfloat/e5-large-v2", device=self.device)
        return self._text_model

    def _ensure_fit_direct(self, n_batch: int) -> None:
        """Lazily create/reset per-batch fit_direct embed Parameter + Adam optimizer."""
        edim = self.nn.embed_dim - self.nn.input_dim
        self._fit_embed = torch.nn.Parameter(torch.zeros(n_batch, edim, device=self.device))
        self._fit_optimizer = torch.optim.Adam([self._fit_embed], lr=self._FIT_DIRECT_LR)

    def _fit_direct_embed(
        self,
        weather: torch.Tensor,
        target: torch.Tensor,
        map_idx: torch.Tensor,
        bsize: int,
        hn: torch.Tensor = None,
    ) -> torch.Tensor:
        """Freeze self.nn and fit a per-batch embed via short Adam/MSE GD; return detached embed.

        cuDNN RNN backward requires training mode even when weights are frozen, so we
        temporarily flip self.nn.train() around the GD loop. Micro-batching keeps peak
        activation memory bounded for large adapt sets.
        """
        self._ensure_fit_direct(bsize)
        grad_flags = [p.requires_grad for p in self.nn.parameters()]
        for p in self.nn.parameters():
            p.requires_grad_(False)

        was_training = self.nn.training
        # Eval/compile paths call calibrator.eval(); cuDNN refuses RNN backward in eval.
        self.nn.train()

        n_seq = weather.shape[0]
        if hn is None:
            hn = self.nn.get_init_state(batch_size=n_seq)
        hn = hn.detach()

        mask = ~torch.isnan(target)
        target_clean = target.nan_to_num(nan=0.0)
        denom = mask.sum().clamp_min(1)
        micro = max(1, min(self._FIT_DIRECT_MICROBATCH, n_seq))

        try:
            with torch.enable_grad():
                for _ in range(self._FIT_DIRECT_STEPS):
                    self._fit_optimizer.zero_grad(set_to_none=True)
                    for start in range(0, n_seq, micro):
                        sl = slice(start, start + micro)
                        # hn is (num_layers, batch, hidden) for nn.GRU
                        hn_sl = hn[:, sl]
                        pred, _ = self.nn.forward_embed(
                            weather[sl], embed=self._fit_embed[map_idx[sl]], hn=hn_sl
                        )
                        # Same mean as full-batch: accumulate ∂(Σ err² / denom).
                        loss = ((pred - target_clean[sl]) ** 2 * mask[sl]).sum() / denom
                        loss.backward()
                    self._fit_optimizer.step()
        finally:
            self.nn.train(was_training)
            for p, flag in zip(self.nn.parameters(), grad_flags):
                p.requires_grad_(flag)
            if weather.is_cuda:
                torch.cuda.empty_cache()

        return self._fit_embed.detach()

    def _id_text_embed(
        self,
        cultivars: torch.Tensor,
        regions: torch.Tensor,
        stations: torch.Tensor,
        sites: torch.Tensor,
        *,
        use_text: bool,
    ) -> torch.Tensor:
        """Build concatenated ID or text embeds for forward_embed (ids_only)."""
        if use_text:
            reg_e, stat_e, site_e, cult_e = self.nn.forward_embed_text(
                cultivars=cultivars,
                regions=regions,
                stations=stations,
                sites=sites,
                text_embedder=self.text_model,
            )
        else:
            reg_e, stat_e, site_e, cult_e = self.nn.forward_embed_direct(
                cultivars=cultivars,
                regions=regions,
                stations=stations,
                sites=sites,
            )
        return self.nn.apply_embed_op(cult_e, site_e, stat_e, reg_e, ids_only=True)

    def _resolve_transfer_embed(
        self,
        data: torch.Tensor,
        cultivars: torch.Tensor,
        regions: torch.Tensor,
        stations: torch.Tensor,
        sites: torch.Tensor,
        transfer_inds: torch.Tensor,
    ) -> torch.Tensor | None:
        """Resolve forward_embed inputs for transfer_inds samples. Returns (N, E) or None."""
        n_transfer = int(transfer_inds.sum().item())
        if n_transfer == 0:
            return None

        ttype = self.config.DConfig.transfer_type
        if ttype == "mean":
            if getattr(self, "_mean_embed", None) is None:
                all_ids = self.all_ids["train"]
                reg_e, stat_e, site_e, cult_e = self.nn.forward_embed_direct(
                    cultivars=all_ids[:, 3:4],
                    regions=all_ids[:, 0:1],
                    stations=all_ids[:, 1:2],
                    sites=all_ids[:, 2:3],
                )
                self._mean_embed = self.nn.apply_embed_op(cult_e, site_e, stat_e, reg_e, ids_only=True).mean(dim=0)
            return self._mean_embed.unsqueeze(0).expand(n_transfer, -1)

        if ttype in ("direct", "text"):
            return self._id_text_embed(
                cultivars[transfer_inds],
                regions[transfer_inds],
                stations[transfer_inds],
                sites[transfer_inds],
                use_text=(ttype == "text"),
            )

        b_all_inds = torch.cat((regions, stations, sites, cultivars), dim=1)[transfer_inds]
        all_inds = self.all_inds["train"]
        adapt_inds = []
        map_idx = []
        for i, ind in enumerate(b_all_inds):
            ids = torch.argwhere((ind == all_inds).all(axis=1)).flatten()
            idx = ids[torch.randperm(ids.shape[0])[: min(ids.shape[0] + 1, self.config.DConfig.num_transfer_years + 1)]]
            adapt_inds.append(idx)
            map_idx.append(torch.ones_like(idx, device=idx.device) * i)
        adapt_inds = torch.cat(adapt_inds)
        map_idx = torch.cat(map_idx)

        if ttype == "fit_direct":
            return self._fit_direct_embed(
                self.data["train"][adapt_inds],
                self.val["train"][adapt_inds],
                map_idx=map_idx,
                bsize=n_transfer,
            )

        adapt_embed = self.nn.compute_embed(
            self.data["train"][adapt_inds],
            ch=self.val["train"][adapt_inds],
            map_idx=map_idx,
            bsize=n_transfer,
        )
        if ttype != "adapt_text":
            return adapt_embed

        # Hybrid: direct cult | adapt (site+stat) | text reg  (== concat_reg layout)
        reg_e, _, _, _ = self.nn.forward_embed_text(
            cultivars=cultivars[transfer_inds],
            regions=regions[transfer_inds],
            stations=stations[transfer_inds],
            sites=sites[transfer_inds],
            text_embedder=self.text_model,
        )
        _, _, _, cult_e = self.nn.forward_embed_direct(
            cultivars=cultivars[transfer_inds],
            regions=regions[transfer_inds],
            stations=stations[transfer_inds],
            sites=sites[transfer_inds],
        )
        q = adapt_embed.shape[-1] // 4
        middle = adapt_embed[..., q : 3 * q]
        return torch.cat((cult_e, middle, reg_e), dim=-1)

    def _noise_prepare_ctx(
        self,
        data: torch.Tensor,
        dates: np.ndarray = None,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        b_size: int = None,
        dlen: int = None,
        transfer_rate: float = 1.0,
        **kwargs,
    ):
        """Resolve a full-batch embed for noisy K-day rollouts via forward_embed."""
        transfer_inds = torch.rand(data.shape[0], device=data.device) < transfer_rate
        if transfer_inds.sum() == 0:
            transfer_inds = torch.ones(data.shape[0], dtype=torch.bool, device=data.device)

        embed = self._resolve_transfer_embed(data, cultivars, regions, stations, sites, transfer_inds)
        if embed is None or (~transfer_inds).any():
            full_embed = torch.zeros(
                data.shape[0],
                self.nn.embed_dim - self.nn.input_dim,
                device=data.device,
                dtype=data.dtype,
            )
            if embed is not None:
                full_embed[transfer_inds] = embed
            if (~transfer_inds).any():
                full_embed[~transfer_inds] = self._id_text_embed(
                    cultivars[~transfer_inds],
                    regions[~transfer_inds],
                    stations[~transfer_inds],
                    sites[~transfer_inds],
                    use_text=(self.config.DConfig.transfer_type == "text"),
                )
            embed = full_embed
        elif embed.shape[0] != data.shape[0]:
            full_embed = torch.zeros(
                data.shape[0],
                embed.shape[-1],
                device=data.device,
                dtype=embed.dtype,
            )
            full_embed[transfer_inds] = embed
            embed = full_embed
        return embed

    def _noise_step(
        self,
        weather_t: torch.Tensor,
        state,
        ctx,
        *,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        dates_t: np.ndarray = None,
        **kwargs,
    ):
        return self.nn.forward_embed(weather_t, embed=ctx, hn=state)

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

        if self.config.DConfig.noise:
            return self.forward_with_noise(
                data,
                dates,
                cultivars,
                regions,
                stations,
                sites,
                transfer_rate=transfer_rate,
                ch=ch,
                name=name,
                b_size=b_size,
                dlen=dlen,
                **kwargs,
            )

        self.nn.zero_grad()
        hn = self.nn.get_init_state(batch_size=b_size)

        map_idx = torch.randint(0, 2, size=cultivars.shape)
        if self.config.DConfig.transfer_type == "direct":
            map_idx = torch.zeros_like(cultivars, device=self.device)
        elif self.config.DConfig.transfer_type == "text":
            map_idx = torch.ones_like(cultivars, device=self.device)
        elif self.config.DConfig.transfer_type in ("adapt", "mean", "fit_direct", "adapt_text"):
            map_idx = torch.ones_like(cultivars, device=self.device) + 1

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
            embed = self._resolve_transfer_embed(data, cultivars, regions, stations, sites, transfer_inds)
            transfer_predict, hn = self.nn.forward_embed(data[transfer_inds], embed=embed, hn=hn[:, transfer_inds])
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
        super().__init__(config, data)

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
        if self.config.DConfig.transfer_type == "direct":
            map_idx = torch.zeros_like(cultivars, device=self.device)
        elif self.config.DConfig.transfer_type == "text":
            map_idx = torch.ones_like(cultivars, device=self.device)
        elif self.config.DConfig.transfer_type == "adapt":
            map_idx = torch.ones_like(cultivars, device=self.device) + 1

        transfer_inds = torch.rand(data.shape[0]) < transfer_rate

        if transfer_inds.sum() != 0:
            all_inds = self.all_inds["train"]
            adapt_inds = []
            map_idx = []
            b_all_inds = torch.cat((regions, stations, sites, cultivars), dim=1)[transfer_inds]
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

            if transfer_inds.sum() != 0:
                transfer_params_predict, hn_t = self.nn.forward_embed(
                    torch.cat(
                        (output.view(output.shape[0], -1).detach()[transfer_inds], data[transfer_inds, i]), dim=-1
                    ),
                    embed=embed,
                    hn=hn,
                )

            step_params = torch.zeros(b_size, param_tens.shape[-1], device=self.device)

            if (~transfer_inds).sum() != 0 and transfer_inds.sum() != 0:
                # Expand masks to match param shape
                mask = transfer_inds.unsqueeze(-1).expand_as(step_params)
                step_params = torch.where(mask, transfer_params_predict.squeeze(), params_predict.squeeze())

                hn_mask = transfer_inds.unsqueeze(0).unsqueeze(-1).expand_as(hn)
                hn_next = torch.where(hn_mask, hn_t, hn_nt)

            elif (~transfer_inds).sum() != 0:
                step_params = params_predict.squeeze()
                hn_next = hn_nt
            else:
                step_params = transfer_params_predict.squeeze()
                hn_next = hn_t
            step_params = self.param_cast(step_params, prev_params=batch_params)
            param_tens[:, i] = step_params
            hn = hn_next

            self.model.set_model_params(step_params, self.params)
            # self.model.set_model_params(params_predict.squeeze(), self.params)
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


class FoundationModel(BaseRNN):
    def __init__(self, config: DictConfig, data: list[pd.DataFrame]) -> None:

        super().__init__(config, data)
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
        data: torch.Tensor,
        dates: np.ndarray,
        cultivars: torch.Tensor = None,
        regions: torch.Tensor = None,
        stations: torch.Tensor = None,
        sites: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        map_idx = torch.randint(0, 2, size=cultivars.shape)
        if self.config.DConfig.transfer_type == "direct":
            map_idx = torch.zeros_like(cultivars, device=self.device)
        elif self.config.DConfig.transfer_type == "text":
            map_idx = torch.ones_like(cultivars, device=self.device)

        data, dates, cultivars, b_size, dlen = self.handle_data(data, dates, cultivars)

        output_tens, _, _ = self.setup_storage(b_size, dlen)
        batch_params = (
            self.task_params[cultivars.to(torch.long).squeeze()] if cultivars is not None else self.task_params[0]
        )

        self.model.set_model_params(batch_params, self.params)
        params, _ = self.nn(
            input=data[:, 0],
            cultivars=cultivars,
            sites=sites,
            stations=stations,
            regions=regions,
            map_idx=map_idx,
            text_embedder=self.text_model,
        )
        self.model.set_model_params(self.param_cast(params), self.params)
        output = self.model.reset(b_size)

        for i in range(dlen):
            output = self.model.run(dates=dates[:, i], cultivars=cultivars)
            output_tens[:, i] = output

        return output_tens, None, None
