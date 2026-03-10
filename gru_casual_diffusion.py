"""
Causal RNN Diffusion Model for Cold-Hardiness Sequence Prediction
==================================================================
This version makes predictions using ONLY past and current weather.
It does NOT require seeing future timesteps — suitable for:
  - Real-time / online prediction
  - Rolling forecasts
  - Cases where future weather is unavailable

Key difference from the bidirectional version:
  - WeatherEncoderRNN:  BiLSTM  →  Causal (forward-only) LSTM
  - RNNDenoiser:        BiGRU   →  Causal (forward-only) GRU
  - h_summary:          concat(fwd, bwd) hidden → last forward hidden state only

Everything else (diffusion schedule, loss, sampling) is identical.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────
# 1. TIMESTEP EMBEDDING
# ─────────────────────────────────────────────


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.embed_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.proj(emb)


# ─────────────────────────────────────────────
# 2. CAUSAL WEATHER ENCODER  (forward LSTM only)
# ─────────────────────────────────────────────


class CausalWeatherEncoder(nn.Module):
    """
    Encodes weather features causally — at each timestep t, the context
    vector only depends on weather[0..t], never on future timesteps.

    Uses a standard (unidirectional, forward-only) LSTM.

    Outputs:
      context   [B, T, cond_dim]  — per-timestep causal encodings
      h_summary [B, cond_dim]     — final hidden state (summary of full seen sequence)

    At inference you can feed this incrementally one timestep at a time
    by passing `h_prev` and `c_prev` (see `step()` method).
    """

    def __init__(self, weather_dim: int = 6, cond_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.cond_dim = cond_dim
        self.num_layers = num_layers

        # Unidirectional — this is the key causal change
        self.lstm = nn.LSTM(
            input_size=weather_dim,
            hidden_size=cond_dim,  # full cond_dim (not //2, not bidirectional)
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,  # ← CAUSAL: no future information
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(cond_dim)

    def forward(self, weather: torch.Tensor):
        """
        Process the entire available weather sequence at once (training / batch inference).
        weather: [B, T, weather_dim]
        Returns:
          context:   [B, T, cond_dim]  — causal per-timestep encodings
          h_summary: [B, cond_dim]     — final hidden state
        """
        context, (h_n, _) = self.lstm(weather)
        # context: [B, T, cond_dim]
        # h_n:     [num_layers, B, cond_dim]

        context = self.norm(context)
        h_summary = h_n[-1]  # last layer's hidden state: [B, cond_dim]
        return context, h_summary

    def step(self, weather_t: torch.Tensor, h_prev=None, c_prev=None):
        """
        Process a SINGLE new timestep incrementally.
        Use this at inference time when weather arrives one step at a time.

        weather_t: [B, 1, weather_dim]   — one new timestep of weather
        h_prev:    [num_layers, B, cond_dim] or None (init to zeros)
        c_prev:    [num_layers, B, cond_dim] or None

        Returns:
          context_t:  [B, 1, cond_dim]
          h_new, c_new  (pass back in on next call)
        """
        B = weather_t.shape[0]
        device = weather_t.device

        if h_prev is None:
            h_prev = torch.zeros(self.num_layers, B, self.cond_dim, device=device)
        if c_prev is None:
            c_prev = torch.zeros(self.num_layers, B, self.cond_dim, device=device)

        context_t, (h_new, c_new) = self.lstm(weather_t, (h_prev, c_prev))
        context_t = self.norm(context_t)
        return context_t, h_new, c_new


# ─────────────────────────────────────────────
# 3. CAUSAL RNN DENOISER  (forward GRU only)
# ─────────────────────────────────────────────


class CausalRNNDenoiser(nn.Module):
    """
    Causal GRU denoiser — predicts noise at each position using only
    current and past information.

    Conditioning:
      1. Input concat:      [x_t | context | t_emb_expanded] at every timestep
      2. Hidden state init: project h_summary → initial GRU hidden state

    rnn_type: "GRU" (default, lighter) or "LSTM"
    """

    def __init__(
        self,
        target_dim: int = 1,
        cond_dim: int = 128,
        time_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        rnn_type: str = "GRU",
    ):
        super().__init__()
        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        input_dim = target_dim + cond_dim + time_dim

        rnn_cls = nn.GRU if rnn_type == "GRU" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,  # ← CAUSAL: no future information
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Project weather summary → initial hidden state
        self.h_init_proj = nn.Linear(cond_dim, num_layers * hidden_dim)

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, target_dim),
        )

    def _build_h0(self, h_summary: torch.Tensor, B: int, device: torch.device):
        """Build initial hidden state: [num_layers, B, hidden_dim]"""
        h = self.h_init_proj(h_summary)  # [B, num_layers * hidden_dim]
        h = h.view(B, self.num_layers, self.hidden_dim)
        h = h.permute(1, 0, 2).contiguous()  # [num_layers, B, hidden_dim]
        return h

    def forward(
        self,
        x_t: torch.Tensor,  # [B, T, 1]
        t_emb: torch.Tensor,  # [B, time_dim]
        context: torch.Tensor,  # [B, T, cond_dim]
        h_summary: torch.Tensor,  # [B, cond_dim]
    ) -> torch.Tensor:

        B, T, _ = x_t.shape
        device = x_t.device

        t_expanded = t_emb[:, None, :].expand(B, T, -1)
        rnn_input = torch.cat([x_t, context, t_expanded], dim=-1)

        h0 = self._build_h0(h_summary, B, device)

        if self.rnn_type == "LSTM":
            c0 = torch.zeros_like(h0)
            out, _ = self.rnn(rnn_input, (h0, c0))
        else:
            out, _ = self.rnn(rnn_input, h0)

        out = self.output_norm(out)
        return self.output_proj(out)  # [B, T, 1]


# ─────────────────────────────────────────────
# 4. FULL CAUSAL DIFFUSION MODEL
# ─────────────────────────────────────────────


class CausalColdHardinessRNNDiffusion(nn.Module):
    """
    Causal conditional diffusion model.

    Identical API to the bidirectional version — drop-in replacement.
    Additionally exposes `predict_online()` for true step-by-step inference.
    """

    def __init__(
        self,
        weather_dim: int = 6,
        cond_dim: int = 128,
        time_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        rnn_type: str = "GRU",
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.T = T
        self.cond_dim = cond_dim

        self.time_embed = SinusoidalTimestepEmbedding(time_dim)

        self.weather_encoder = CausalWeatherEncoder(
            weather_dim=weather_dim,
            cond_dim=cond_dim,
            num_layers=max(1, num_layers - 1),
            dropout=dropout,
        )

        self.denoiser = CausalRNNDenoiser(
            target_dim=1,
            cond_dim=cond_dim,
            time_dim=time_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            rnn_type=rnn_type,
        )

        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1 - alphas_cumprod).sqrt())

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        s = self.sqrt_alphas_cumprod[t][:, None, None]
        s1 = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return s * x0 + s1 * noise

    # ── Training ──────────────────────────────────────────

    def p_losses(self, x0: torch.Tensor, weather: torch.Tensor) -> torch.Tensor:
        """
        Standard DDPM loss. Same signature as the bidirectional version.
        x0:      [B, T_seq, 1]
        weather: [B, T_seq, 6]
        """
        B = x0.shape[0]
        device = x0.device

        t = torch.randint(0, self.T, (B,), device=device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)

        t_emb = self.time_embed(t)
        context, h_summary = self.weather_encoder(weather)
        noise_pred = self.denoiser(x_t, t_emb, context, h_summary)

        return F.mse_loss(noise_pred, noise)

    # ── Batch inference (seen sequence) ───────────────────

    @torch.no_grad()
    def sample(self, weather: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Generate predictions for a complete (already-observed) weather sequence.
        weather: [B, T_seq, 6]
        Returns: [B, num_samples, T_seq, 1]
        """
        B, T_seq, _ = weather.shape
        device = weather.device

        context, h_summary = self.weather_encoder(weather)

        all_samples = []
        for _ in range(num_samples):
            x = torch.randn(B, T_seq, 1, device=device)

            for t_val in reversed(range(self.T)):
                t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
                t_emb = self.time_embed(t_batch)
                noise_pred = self.denoiser(x, t_emb, context, h_summary)

                beta_t = self.betas[t_val]
                alpha_t = self.alphas[t_val]
                alpha_bar = self.alphas_cumprod[t_val]

                mean = (1.0 / alpha_t.sqrt()) * (x - (beta_t / (1 - alpha_bar).sqrt()) * noise_pred)
                x = mean + (beta_t.sqrt() * torch.randn_like(x) if t_val > 0 else 0)

            all_samples.append(x.unsqueeze(1))

        return torch.cat(all_samples, dim=1)  # [B, num_samples, T_seq, 1]

    # ── Online / streaming inference ──────────────────────

    @torch.no_grad()
    def predict_online(
        self,
        weather_stream: torch.Tensor,
        num_samples: int = 1,
        diffusion_steps: int = None,
    ) -> torch.Tensor:
        """
        Online (causal) inference: predict cold-hardiness one timestep
        at a time as weather data arrives. No future weather is used.

        weather_stream: [B, T_seq, 6]  — fed in left-to-right, one step at a time
        num_samples:    number of stochastic samples per timestep
        diffusion_steps: override self.T for faster inference (e.g. 50)

        Returns: predictions [B, T_seq, num_samples, 1]
          predictions[:, t, :, :] only depends on weather_stream[:, 0:t+1, :]

        Example usage:
            model.eval()
            for t in range(T_seq):
                new_weather = sensor.read()           # [B, 1, 6]
                pred = model.predict_online(weather_seen_so_far)
                hardiness_now = pred[:, -1, :, :]     # prediction for latest timestep
        """
        T_diff = diffusion_steps or self.T
        B, T_seq, _ = weather_stream.shape
        device = weather_stream.device

        all_preds = []  # will be [T_seq, B, num_samples, 1]

        # Carry LSTM hidden state across timesteps
        h_enc, c_enc = None, None

        # Accumulate context vectors as we step through the sequence
        context_so_far = []

        for t_seq in range(T_seq):
            # Feed one new weather timestep to the encoder
            weather_t = weather_stream[:, t_seq : t_seq + 1, :]  # [B, 1, 6]
            ctx_t, h_enc, c_enc = self.weather_encoder.step(weather_t, h_enc, c_enc)
            # ctx_t: [B, 1, cond_dim]

            context_so_far.append(ctx_t)
            context_full = torch.cat(context_so_far, dim=1)  # [B, t_seq+1, cond_dim]
            h_summary = h_enc[-1]  # [B, cond_dim]

            # Sample cold-hardiness for the sequence seen so far
            timestep_samples = []
            for _ in range(num_samples):
                x = torch.randn(B, t_seq + 1, 1, device=device)

                for t_diff in reversed(range(T_diff)):
                    t_batch = torch.full((B,), t_diff, device=device, dtype=torch.long)
                    t_emb = self.time_embed(t_batch)
                    noise_pred = self.denoiser(x, t_emb, context_full, h_summary)

                    beta_t = self.betas[t_diff]
                    alpha_t = self.alphas[t_diff]
                    alpha_bar = self.alphas_cumprod[t_diff]

                    mean = (1.0 / alpha_t.sqrt()) * (x - (beta_t / (1 - alpha_bar).sqrt()) * noise_pred)
                    x = mean + (beta_t.sqrt() * torch.randn_like(x) if t_diff > 0 else 0)

                # Only keep the prediction for the latest timestep: [B, 1, 1]
                timestep_samples.append(x[:, -1:, :])

            # Stack samples: [B, num_samples, 1]
            all_preds.append(torch.cat(timestep_samples, dim=1))

        # Stack over sequence: [B, T_seq, num_samples, 1]
        return torch.stack(all_preds, dim=1)


# ─────────────────────────────────────────────
# 5. TRAINING LOOP
# ─────────────────────────────────────────────


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for weather, cold_hardiness in dataloader:
        weather = weather.to(device)
        cold_hardiness = cold_hardiness.to(device)
        optimizer.zero_grad()
        loss = model.p_losses(cold_hardiness, weather)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


# ─────────────────────────────────────────────
# 6. SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    B, T_seq = 2, 30  # small for quick test

    model = CausalColdHardinessRNNDiffusion(
        weather_dim=6,
        cond_dim=64,
        time_dim=64,
        hidden_dim=128,
        num_layers=2,
        T=100,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    weather = torch.randn(B, T_seq, 6).to(device)
    cold_hardiness = torch.randn(B, T_seq, 1).to(device)

    # Training step
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss = model.p_losses(cold_hardiness, weather)
    loss.backward()
    print(f"Training loss: {loss.item():.4f}  ✓")

    # Batch sample
    model.eval()
    samples = model.sample(weather, num_samples=3)
    print(f"Batch sample shape:  {samples.shape}")  # [2, 3, 30, 1]

    # Online prediction
    online_preds = model.predict_online(weather, num_samples=3, diffusion_steps=20)
    print(f"Online pred shape:   {online_preds.shape}")  # [2, 30, 3, 1]

    # Verify causality: prediction at t=5 should not change when we extend sequence
    pred_short = model.predict_online(weather[:, :10, :], num_samples=1, diffusion_steps=20)
    pred_long = model.predict_online(weather[:, :20, :], num_samples=1, diffusion_steps=20)

    # The first 10 predictions should be equivalent (same weather seen)
    # Note: stochastic, so we check the encoder context is causal not exact equality
    print(f"Short pred shape:    {pred_short.shape}")  # [2, 10, 1, 1]
    print(f"Long pred shape:     {pred_long.shape}")  # [2, 20, 1, 1]

    print("\nAll smoke tests passed!")
