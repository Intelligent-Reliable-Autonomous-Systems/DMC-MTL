"""
RNN-Based Conditional Diffusion Model for Cold-Hardiness Sequence Prediction
=============================================================================
Architecture: Bidirectional GRU denoiser with LSTM weather encoder
- Target:    cold-hardiness sequence  [B, T, 1]
- Condition: weather features         [B, T, 6]

Replaces the 1D U-Net with a BiGRU denoiser.
Weather context is encoded by a BiLSTM and injected via:
  - Concatenation at the input
  - Cross-timestep hidden state initialization

Switch between GRU and LSTM via `rnn_type` argument.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────
# 1. TIMESTEP EMBEDDING  (same as U-Net version)
# ─────────────────────────────────────────────


class SinusoidalTimestepEmbedding(nn.Module):
    """Encodes diffusion timestep t → dense vector [B, embed_dim]."""

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
# 2. WEATHER ENCODER  (BiLSTM)
# ─────────────────────────────────────────────


class WeatherEncoderRNN(nn.Module):
    """
    Encodes the weather feature sequence [B, T, 6] using a BiLSTM.

    Outputs:
      - context:    [B, T, cond_dim]  per-timestep encodings (for concatenation)
      - h_summary:  [B, cond_dim]     global summary (for denoiser init)

    Using BiLSTM (vs BiGRU) here because LSTM's cell state is well-suited
    for capturing long-range seasonal patterns in weather.
    """

    def __init__(self, weather_dim: int = 6, cond_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        assert cond_dim % 2 == 0, "cond_dim must be even for bidirectional RNN"

        self.lstm = nn.LSTM(
            input_size=weather_dim,
            hidden_size=cond_dim // 2,  # //2 because bidirectional doubles it
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(cond_dim)

    def forward(self, weather: torch.Tensor):
        # weather: [B, T, 6]
        context, (h_n, _) = self.lstm(weather)
        # context:  [B, T, cond_dim]
        # h_n:      [num_layers*2, B, cond_dim//2]

        context = self.norm(context)

        # Global summary: concat last forward + last backward hidden states
        # h_n[-2] = last forward layer, h_n[-1] = last backward layer
        h_summary = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # [B, cond_dim]

        return context, h_summary


# ─────────────────────────────────────────────
# 3. RNN DENOISER  (BiGRU)
# ─────────────────────────────────────────────


class RNNDenoiser(nn.Module):
    """
    Bidirectional GRU denoiser.

    At each diffusion step it takes:
      - x_t:      [B, T, 1]        noisy cold-hardiness sequence
      - t_emb:    [B, time_dim]    diffusion timestep embedding
      - context:  [B, T, cond_dim] per-timestep weather encodings
      - h_init:   [B, cond_dim]    global weather summary for hidden state init

    Conditioning strategy:
      1. Input concatenation: [x_t | context | t_emb_expanded] at every timestep
      2. Hidden state init: project h_init → initial hidden state of GRU

    Why BiGRU for the denoiser?
      - Bidirectional lets it see both past and future context when denoising
      - GRU is lighter than LSTM (fewer params) — the weather encoder handles
        the heavy lifting, so the denoiser doesn't need LSTM's cell state
    """

    def __init__(
        self,
        target_dim: int = 1,
        cond_dim: int = 128,
        time_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        rnn_type: str = "GRU",  # "GRU" or "LSTM"
    ):
        super().__init__()
        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input: noisy target + per-timestep context + timestep emb (broadcast)
        input_dim = target_dim + cond_dim + time_dim

        rnn_cls = nn.GRU if rnn_type == "GRU" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,  # //2 for bidirectional
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Project global weather summary → initial hidden state for each layer
        # Shape needed: [num_layers*2, B, hidden_dim//2]
        self.h_init_proj = nn.Linear(cond_dim, num_layers * hidden_dim)

        # Output projection: hidden → predicted noise
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, target_dim),
        )

    def _build_h0(self, h_summary: torch.Tensor, batch_size: int, device: torch.device):
        """
        Build initial hidden state from global weather summary.
        h_summary: [B, cond_dim]
        Returns:   [num_layers*2, B, hidden_dim//2]  (×2 for bidirectional)
        """
        h = self.h_init_proj(h_summary)  # [B, num_layers * hidden_dim]
        h = h.view(batch_size, self.num_layers, 2, self.hidden_dim // 2)
        h = h.permute(1, 2, 0, 3)  # [num_layers, 2, B, hidden_dim//2]
        h = h.contiguous().view(self.num_layers * 2, batch_size, self.hidden_dim // 2)
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

        # Expand timestep embedding across sequence: [B, time_dim] → [B, T, time_dim]
        t_expanded = t_emb[:, None, :].expand(B, T, -1)

        # Concatenate all inputs: [B, T, target_dim + cond_dim + time_dim]
        rnn_input = torch.cat([x_t, context, t_expanded], dim=-1)

        # Build initial hidden state from global weather summary
        h0 = self._build_h0(h_summary, B, device)

        if self.rnn_type == "LSTM":
            # LSTM needs (h0, c0); init c0 to zeros
            c0 = torch.zeros_like(h0)
            out, _ = self.rnn(rnn_input, (h0, c0))
        else:
            out, _ = self.rnn(rnn_input, h0)

        # out: [B, T, hidden_dim]
        out = self.output_norm(out)
        return self.output_proj(out)  # [B, T, 1]  — predicted noise


# ─────────────────────────────────────────────
# 4. FULL CONDITIONAL DIFFUSION MODEL
# ─────────────────────────────────────────────


class ColdHardinessRNNDiffusion(nn.Module):
    """
    Full RNN-based conditional diffusion model.

    Components:
      - WeatherEncoderRNN  (BiLSTM) — encodes 6 weather features
      - RNNDenoiser        (BiGRU or BiLSTM) — predicts noise at each diffusion step
      - DDPM noise schedule

    Args:
        weather_dim:  number of weather input features (default 6)
        cond_dim:     hidden dim of weather encoder & context vectors
        time_dim:     diffusion timestep embedding dimension
        hidden_dim:   hidden dim of denoiser RNN
        num_layers:   number of layers in both encoder and denoiser RNNs
        T:            number of diffusion timesteps
        rnn_type:     "GRU" or "LSTM" for the denoiser
        beta_start:   linear schedule start
        beta_end:     linear schedule end
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

        self.time_embed = SinusoidalTimestepEmbedding(time_dim)

        self.weather_encoder = WeatherEncoderRNN(
            weather_dim=weather_dim,
            cond_dim=cond_dim,
            num_layers=max(1, num_layers - 1),  # slightly shallower encoder
            dropout=dropout,
        )

        self.denoiser = RNNDenoiser(
            target_dim=1,
            cond_dim=cond_dim,
            time_dim=time_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            rnn_type=rnn_type,
        )

        # ── DDPM linear noise schedule ────────────────────
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1 - alphas_cumprod).sqrt())

    # ── Forward diffusion ─────────────────────────────────

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """Add noise to x0 at diffusion timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_a = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_1a = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return sqrt_a * x0 + sqrt_1a * noise

    # ── Training loss ─────────────────────────────────────

    def p_losses(self, x0: torch.Tensor, weather: torch.Tensor) -> torch.Tensor:
        """
        DDPM training loss.

        x0:      [B, T_seq, 1]   normalized cold-hardiness
        weather: [B, T_seq, 6]   normalized weather features
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

    # ── Reverse diffusion (sampling) ──────────────────────

    @torch.no_grad()
    def sample(self, weather: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Generate cold-hardiness predictions conditioned on weather.

        weather:     [B, T_seq, 6]
        num_samples: how many stochastic draws (for uncertainty quantification)

        Returns: [B, num_samples, T_seq, 1]
        """
        B, T_seq, _ = weather.shape
        device = weather.device

        # Encode weather once — reuse across all samples
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

                coef1 = 1.0 / alpha_t.sqrt()
                coef2 = beta_t / (1.0 - alpha_bar).sqrt()
                mean = coef1 * (x - coef2 * noise_pred)

                if t_val > 0:
                    x = mean + beta_t.sqrt() * torch.randn_like(x)
                else:
                    x = mean

            all_samples.append(x.unsqueeze(1))

        return torch.cat(all_samples, dim=1)  # [B, num_samples, T_seq, 1]


# ─────────────────────────────────────────────
# 5. TRAINING LOOP
# ─────────────────────────────────────────────


def train_one_epoch(model, dataloader, optimizer, device):
    """
    dataloader yields: (weather [B,T,6], cold_hardiness [B,T,1])
    Both should be normalized (zero mean, unit variance) beforehand.
    """
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

    B, T_seq = 4, 365

    for rnn_type in ["GRU", "LSTM"]:
        print(f"\n── Testing rnn_type={rnn_type} ──")
        model = ColdHardinessRNNDiffusion(
            weather_dim=6,
            cond_dim=128,
            time_dim=128,
            hidden_dim=256,
            num_layers=3,
            dropout=0.1,
            rnn_type=rnn_type,
            T=1000,
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {total_params:,}")

        weather = torch.randn(B, T_seq, 6).to(device)
        cold_hardiness = torch.randn(B, T_seq, 1).to(device)

        # Training step
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss = model.p_losses(cold_hardiness, weather)
        print(f"Training loss (random data): {loss.item():.4f}")
        loss.backward()
        print("Backward pass: OK")

        # Inference (reduced steps for speed)
        model.T = 50
        samples = model.sample(weather, num_samples=3)
        print(f"Sample output shape: {samples.shape}")  # [4, 3, 365, 1]

    print("\nAll smoke tests passed!")
