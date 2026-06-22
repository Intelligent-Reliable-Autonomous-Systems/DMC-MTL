"""
Diffusion Model for Sequence Prediction
========================================
Given: a sequence of feature vectors  (B, T, D_in)
Predict: a sequence of target values   (B, T, D_out)

Architecture: Conditional DDPM
  - The input feature sequence is the conditioning signal.
  - The diffusion process operates on the TARGET sequence.
  - A Transformer-based denoising network predicts the noise.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ─────────────────────────────────────────────────────────────
# 1. NOISE SCHEDULE
# ─────────────────────────────────────────────────────────────


class CosineNoiseSchedule:
    """
    Cosine noise schedule (Nichol & Dhariwal 2021).
    Precomputes alpha_bar (cumulative product of (1 - beta_t)).
    """

    def __init__(self, T: int = 1000, s: float = 0.008, device="cpu"):
        self.T = T
        self.device = device

        # alpha_bar_t = cos²( (t/T + s) / (1 + s) * π/2 )
        steps = torch.arange(T + 1, dtype=torch.float32, device=device)
        f = torch.cos(((steps / T + s) / (1 + s)) * math.pi / 2) ** 2
        alpha_bar = f / f[0]  # (T+1,)

        beta = 1 - alpha_bar[1:] / alpha_bar[:-1]  # (T,)
        beta = beta.clamp(0.0, 0.999)

        self.alpha_bar = alpha_bar[1:]  # (T,)  indexed 1..T
        self.beta = beta
        self.alpha = 1.0 - beta
        self.sqrt_alpha_bar = self.alpha_bar.sqrt()
        self.sqrt_one_minus_ab = (1.0 - self.alpha_bar).sqrt()

    # ── forward diffusion: add noise at arbitrary timestep ──────
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """
        x0    : (B, T_seq, D_out)  — clean target sequence
        t     : (B,)               — timestep indices ∈ [0, T-1]
        Returns noisy x_t and the noise added.
        """
        if noise is None:
            noise = torch.randn_like(x0)

        # gather per-sample schedule values → (B, 1, 1) for broadcasting
        def gather(arr):
            return arr[t].view(-1, 1, 1)

        x_t = gather(self.sqrt_alpha_bar) * x0 + gather(self.sqrt_one_minus_ab) * noise
        return x_t, noise

    # ── reverse: one DDPM denoising step ────────────────────────
    @torch.no_grad()
    def p_sample_step(self, model, x_t, t_idx: int, condition: torch.Tensor):
        """Single reverse step  x_t → x_{t-1}."""
        B = x_t.shape[0]
        t_tensor = torch.full((B,), t_idx, device=self.device, dtype=torch.long)

        # predict noise
        eps_pred = model(x_t, t_tensor, condition)

        # DDPM posterior mean
        alpha = self.alpha[t_idx]
        alpha_bar = self.alpha_bar[t_idx]
        beta = self.beta[t_idx]

        coef = (1 - alpha) / (1 - alpha_bar).sqrt()
        mean = (1 / alpha.sqrt()) * (x_t - coef * eps_pred)

        if t_idx == 0:
            return mean
        # posterior variance
        alpha_bar_prev = self.alpha_bar[t_idx - 1]
        var = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
        return mean + var.sqrt() * torch.randn_like(x_t)

    # ── full reverse chain: noise → prediction ──────────────────
    @torch.no_grad()
    def p_sample_loop(self, model, condition: torch.Tensor, target_shape: tuple):
        """
        Denoise from pure Gaussian noise to a clean prediction.
        condition : (B, T_seq, D_in)
        target_shape : (B, T_seq, D_out)
        """
        x = torch.randn(target_shape, device=self.device)
        for t in reversed(range(self.T)):
            x = self.p_sample_step(model, x, t, condition)
        return x


# ─────────────────────────────────────────────────────────────
# 2. SINUSOIDAL TIMESTEP EMBEDDING
# ─────────────────────────────────────────────────────────────


class SinusoidalTimeEmbedding(nn.Module):
    """Maps scalar timestep t → dense embedding vector."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        # projection MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t : (B,) int64  →  (B, dim)"""
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, dim)
        return self.mlp(emb)


# ─────────────────────────────────────────────────────────────
# 3. DENOISING NETWORK (Transformer)
# ─────────────────────────────────────────────────────────────


class DenoisingTransformer(nn.Module):
    """
    Predicts the noise ε added at timestep t.

    Inputs
    ──────
    x_t       : (B, T_seq, D_out)  noisy target sequence
    t         : (B,)               diffusion timestep
    condition : (B, T_seq, D_in)   input feature sequence (conditioning)

    Output
    ──────
    eps_pred  : (B, T_seq, D_out)  predicted noise
    """

    def __init__(
        self,
        d_in: int = 16,  # feature dimension of condition
        d_out: int = 1,  # target dimension
        d_model: int = 128,  # transformer hidden size
        n_heads: int = 4,
        n_layers: int = 4,
        seq_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # ── input projections ───────────────────────────────────
        self.cond_proj = nn.Linear(d_in, d_model)  # condition
        self.noisy_proj = nn.Linear(d_out, d_model)  # noisy target

        # ── timestep embedding ──────────────────────────────────
        self.time_emb = SinusoidalTimeEmbedding(d_model)

        # ── positional encoding ─────────────────────────────────
        self.register_buffer("pos_enc", self._build_pos_enc(seq_len, d_model))

        # ── cross-attention Transformer ─────────────────────────
        #   query  = noisy target + time
        #   key/value = condition
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-LN for stability
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # ── output projection ───────────────────────────────────
        self.out_proj = nn.Linear(d_model, d_out)

    @staticmethod
    def _build_pos_enc(max_len: int, d_model: int) -> torch.Tensor:
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        return pe

    def forward(
        self,
        x_t: torch.Tensor,  # (B, T, D_out)
        t: torch.Tensor,  # (B,)
        condition: torch.Tensor,  # (B, T, D_in)
    ) -> torch.Tensor:
        T = x_t.size(1)

        # project condition + add positional encoding
        cond = self.cond_proj(condition) + self.pos_enc[:, :T]  # (B, T, d_model)

        # project noisy target, inject timestep + positional encoding
        t_emb = self.time_emb(t).unsqueeze(1)  # (B, 1, d_model)
        tgt = self.noisy_proj(x_t) + self.pos_enc[:, :T] + t_emb  # (B, T, d_model)

        # cross-attend: tgt queries condition memory
        out = self.transformer(tgt=tgt, memory=cond)  # (B, T, d_model)
        return self.out_proj(out)  # (B, T, D_out)


# ─────────────────────────────────────────────────────────────
# 4. SYNTHETIC DATASET (for demonstration)
# ─────────────────────────────────────────────────────────────


class SyntheticSeqDataset(Dataset):
    """
    Toy task: target = weighted sum of input features + small noise.
    Replace with your real dataset.
    """

    def __init__(self, n_samples=2000, seq_len=64, d_in=16, d_out=1):
        self.x = torch.randn(n_samples, seq_len, d_in)
        w = torch.randn(d_in, d_out) * 0.1
        # target is a linear mix of features + noise
        self.y = self.x @ w + 0.05 * torch.randn(n_samples, seq_len, d_out)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────
# 5. TRAINING LOOP
# ─────────────────────────────────────────────────────────────


def train(
    # model / diffusion
    d_in: int = 16,
    d_out: int = 1,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    seq_len: int = 64,
    T: int = 1000,  # diffusion steps
    # training
    epochs: int = 30,
    batch: int = 64,
    lr: float = 3e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"Training on {device}")

    # ── build components ────────────────────────────────────────
    schedule = CosineNoiseSchedule(T=T, device=device)

    model = DenoisingTransformer(
        d_in=d_in,
        d_out=d_out,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        seq_len=seq_len,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # cosine LR decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    # ── dataset ─────────────────────────────────────────────────
    dataset = SyntheticSeqDataset(n_samples=4000, seq_len=seq_len, d_in=d_in, d_out=d_out)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=0)

    # ── training ────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y in loader:
            x, y = x.to(device), y.to(device)  # (B, T, D_in/out)
            B = x.size(0)

            # ① sample random timesteps
            t = torch.randint(0, T, (B,), device=device, dtype=torch.long)

            # ② forward diffuse the TARGET
            x_t, noise = schedule.q_sample(y, t)

            # ③ predict the noise (condition on input features x)
            eps_pred = model(x_t, t, condition=x)

            # ④ simple MSE loss on noise prediction
            loss = F.mse_loss(eps_pred, noise)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            total_loss += loss.item() * B

        scheduler.step()
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch:3d}/{epochs}  loss={avg_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

    return model, schedule


# ─────────────────────────────────────────────────────────────
# 6. INFERENCE HELPER
# ─────────────────────────────────────────────────────────────


@torch.no_grad()
def predict(
    model: DenoisingTransformer,
    schedule: CosineNoiseSchedule,
    condition: torch.Tensor,  # (B, T_seq, D_in) — your input features
    d_out: int = 1,
) -> torch.Tensor:
    """
    Run the full reverse diffusion chain to produce a prediction.
    Returns (B, T_seq, D_out).
    """
    model.eval()
    B, T_seq, _ = condition.shape
    target_shape = (B, T_seq, d_out)
    return schedule.p_sample_loop(model, condition, target_shape)


# ─────────────────────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── train ───────────────────────────────────────────────────
    model, schedule = train(
        d_in=16,
        d_out=1,
        seq_len=64,
        d_model=128,
        n_heads=4,
        n_layers=4,
        T=1000,
        epochs=30,
        batch=64,
        lr=3e-4,
    )

    # ── inference example ────────────────────────────────────────
    device = next(model.parameters()).device
    test_condition = torch.randn(4, 64, 16, device=device)  # 4 samples
    prediction = predict(model, schedule, test_condition, d_out=1)
    print(f"\nPrediction shape: {prediction.shape}")  # → (4, 64, 1)
