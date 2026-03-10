"""
Conditional Diffusion Model for Cold-Hardiness Sequence Prediction
====================================================================
Architecture: 1D U-Net with Cross-Attention Conditioning
- Target:    cold-hardiness sequence  [B, T, 1]
- Condition: weather features         [B, T, 6]

Diffusion follows DDPM (Ho et al., 2020) with epsilon-parameterization.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────
# 1. TIMESTEP EMBEDDING
# ─────────────────────────────────────────────


class SinusoidalTimestepEmbedding(nn.Module):
    """
    Encodes the diffusion timestep t into a vector using sinusoidal
    positional encoding, then projects it to `embed_dim`.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] integer timesteps
        half = self.embed_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))  # [half]
        args = t[:, None].float() * freqs[None, :]  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, embed_dim]
        return self.proj(emb)  # [B, embed_dim]


# ─────────────────────────────────────────────
# 2. WEATHER FEATURE ENCODER (Conditioning)
# ─────────────────────────────────────────────


class WeatherEncoder(nn.Module):
    """
    Encodes the weather feature sequence [B, T, 6] into a richer
    context representation [B, T, cond_dim] using a Transformer encoder.
    This context is used by cross-attention layers inside the U-Net.
    """

    def __init__(self, weather_dim: int = 6, cond_dim: int = 128, num_layers: int = 2, nhead: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(weather_dim, cond_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cond_dim,
            nhead=nhead,
            dim_feedforward=cond_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, weather: torch.Tensor) -> torch.Tensor:
        # weather: [B, T, 6]  →  context: [B, T, cond_dim]
        x = self.input_proj(weather)
        return self.encoder(x)


# ─────────────────────────────────────────────
# 3. BUILDING BLOCKS
# ─────────────────────────────────────────────


class ResidualBlock1D(nn.Module):
    """
    1D residual block with:
      - Two Conv1d layers with GroupNorm + SiLU
      - Timestep embedding injected via FiLM (scale + shift)
      - Optional cross-attention over weather context
    """

    def __init__(
        self, in_channels: int, out_channels: int, time_dim: int, cond_dim: int, nhead: int = 4, use_attn: bool = True
    ):
        super().__init__()

        # Time embedding → FiLM scale and shift
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels * 2),  # scale and shift
        )

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        # Residual projection if channels differ
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        # Cross-attention over weather context
        self.use_attn = use_attn
        if use_attn:
            self.cross_attn = CrossAttention1D(out_channels, cond_dim, nhead)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x:       [B, C, T]  — noisy sequence (channels-first for Conv1d)
        t_emb:   [B, time_dim]
        context: [B, T, cond_dim]  — weather context
        """
        h = self.conv1(F.silu(self.norm1(x)))

        # FiLM conditioning from timestep
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=-1)  # each [B, out_channels]
        h = h * (scale[:, :, None] + 1) + shift[:, :, None]  # broadcast over T

        h = self.conv2(F.silu(self.norm2(h)))

        h = h + self.res_conv(x)  # residual connection

        # Cross-attention with weather context
        if self.use_attn:
            h = self.cross_attn(h, context)

        return h


class CrossAttention1D(nn.Module):
    """
    Cross-attention: sequence x attends to weather context.
    x:       [B, C, T]   (channels-first)
    context: [B, T, cond_dim]
    """

    def __init__(self, query_dim: int, context_dim: int, nhead: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, query_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=nhead,
            kdim=context_dim,
            vdim=context_dim,
            batch_first=True,
            dropout=0.0,
        )
        self.out_proj = nn.Conv1d(query_dim, query_dim, 1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)  # [B, C, T]
        q = x_norm.permute(0, 2, 1)  # [B, T, C]  ← queries from noisy sequence
        k = v = context  # [B, T, cond_dim] ← keys/values from weather
        attn_out, _ = self.attn(q, k, v)  # [B, T, C]
        attn_out = attn_out.permute(0, 2, 1)  # [B, C, T]
        return residual + self.out_proj(attn_out)


class Downsample1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


# ─────────────────────────────────────────────
# 4. 1D U-NET DENOISER
# ─────────────────────────────────────────────


class UNet1D(nn.Module):
    """
    1D U-Net denoiser for the diffusion model.

    Encoder: progressively downsamples the sequence
    Bottleneck: deepest processing with full attention
    Decoder: upsamples with skip connections from encoder

    Args:
        in_channels:   number of target channels (1 for cold-hardiness)
        base_channels: base width, doubled at each level
        channel_mults: multipliers per U-Net level
        time_dim:      dimension of timestep embedding
        cond_dim:      dimension of weather context from WeatherEncoder
        nhead:         attention heads in cross-attention
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4),
        time_dim: int = 128,
        cond_dim: int = 128,
        nhead: int = 4,
    ):
        super().__init__()

        self.time_embed = SinusoidalTimestepEmbedding(time_dim)

        channels = [base_channels * m for m in channel_mults]
        # e.g. [64, 128, 256] for base_channels=64, mults=(1,2,4) → oops, fixed below

        # Input projection: [B, 1, T] → [B, base_channels, T]
        self.input_conv = nn.Conv1d(in_channels, channels[0], kernel_size=3, padding=1)

        # ── Encoder ──────────────────────────────────────
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        in_ch = channels[0]
        skip_channels = [in_ch]  # track skip channels for decoder

        for i, out_ch in enumerate(channels):
            use_attn = True  # use cross-attention at every level
            self.encoder_blocks.append(ResidualBlock1D(in_ch, out_ch, time_dim, cond_dim, nhead, use_attn))
            skip_channels.append(out_ch)
            if i < len(channels) - 1:
                self.downsamples.append(Downsample1D(out_ch))
            else:
                self.downsamples.append(nn.Identity())  # no downsample at last encoder level
            in_ch = out_ch

        # ── Bottleneck ────────────────────────────────────
        self.bottleneck = nn.Sequential(
            ResidualBlock1D(channels[-1], channels[-1], time_dim, cond_dim, nhead, use_attn=True),
        )
        # Store bottleneck as a single module for forward call
        self.bottleneck = ResidualBlock1D(channels[-1], channels[-1], time_dim, cond_dim, nhead, use_attn=True)

        # ── Decoder ──────────────────────────────────────
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        reversed_channels = list(reversed(channels))
        for i, out_ch in enumerate(reversed_channels):
            skip_ch = skip_channels[-(i + 1)]
            self.decoder_blocks.append(
                ResidualBlock1D(reversed_channels[i] + skip_ch, out_ch, time_dim, cond_dim, nhead, use_attn=True)
            )
            if i < len(reversed_channels) - 1:
                self.upsamples.append(Upsample1D(out_ch))
            else:
                self.upsamples.append(nn.Identity())  # no upsample at last decoder level

        # Output projection: [B, base_channels, T] → [B, in_channels, T]
        self.output_norm = nn.GroupNorm(8, channels[0])
        self.output_conv = nn.Conv1d(channels[0], in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x:       [B, T, 1]      — noisy cold-hardiness sequence
        t:       [B]            — diffusion timestep indices
        context: [B, T, cond_dim] — encoded weather features

        Returns: predicted noise [B, T, 1]
        """
        # Convert to channels-first for Conv1d
        x = x.permute(0, 2, 1)  # [B, 1, T]

        # Timestep embedding
        t_emb = self.time_embed(t)  # [B, time_dim]

        # Input projection
        x = self.input_conv(x)  # [B, base_channels, T]

        # Encoder
        skips = [x]
        for block, down in zip(self.encoder_blocks, self.downsamples):
            x = block(x, t_emb, context)
            skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.bottleneck(x, t_emb, context)

        # Decoder
        for block, up in zip(self.decoder_blocks, self.upsamples):
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)  # skip connection
            x = block(x, t_emb, context)
            x = up(x)

        # Output
        x = self.output_conv(F.silu(self.output_norm(x)))  # [B, 1, T]
        return x.permute(0, 2, 1)  # [B, T, 1]


# ─────────────────────────────────────────────
# 5. FULL CONDITIONAL DIFFUSION MODEL
# ─────────────────────────────────────────────


class ColdHardinesseDiffusionModel(nn.Module):
    """
    Full conditional diffusion model wrapping:
      - WeatherEncoder  (encodes 6 weather features into context)
      - UNet1D          (predicts noise given noisy target + context + timestep)
      - DDPM noise schedule

    Args:
        weather_dim:   number of weather input features (default 6)
        T:             number of diffusion steps (default 1000)
        beta_start:    start of noise schedule
        beta_end:      end of noise schedule
    """

    def __init__(
        self,
        weather_dim: int = 6,
        cond_dim: int = 128,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4),
        time_dim: int = 128,
        nhead: int = 4,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()

        self.T = T

        # Sub-modules
        self.weather_encoder = WeatherEncoder(weather_dim, cond_dim, num_layers=2, nhead=nhead)
        self.unet = UNet1D(
            in_channels=1,
            base_channels=base_channels,
            channel_mults=channel_mults,
            time_dim=time_dim,
            cond_dim=cond_dim,
            nhead=nhead,
        )

        # ── DDPM noise schedule (linear) ─────────────────
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1 - alphas_cumprod).sqrt())

    # ── Forward diffusion (add noise) ────────────────────

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """
        Diffuse x0 to x_t by adding noise.
        x0:    [B, T_seq, 1]
        t:     [B]  integer diffusion timesteps
        Returns x_t: [B, T_seq, 1]
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]  # [B, 1, 1]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]

        return sqrt_alpha * x0 + sqrt_one_minus * noise

    # ── Training loss ─────────────────────────────────────

    def p_losses(self, x0: torch.Tensor, weather: torch.Tensor) -> torch.Tensor:
        """
        Compute the DDPM training loss (MSE on predicted vs actual noise).

        x0:      [B, T_seq, 1]   — normalized cold-hardiness targets
        weather: [B, T_seq, 6]  — normalized weather features
        """
        B = x0.shape[0]
        device = x0.device

        # Sample random diffusion timesteps
        t = torch.randint(0, self.T, (B,), device=device)

        # Sample noise and create noisy input
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)

        # Encode weather condition
        context = self.weather_encoder(weather)  # [B, T_seq, cond_dim]

        # Predict noise
        noise_pred = self.unet(x_t, t, context)  # [B, T_seq, 1]

        return F.mse_loss(noise_pred, noise)

    # ── Reverse diffusion (sampling / inference) ──────────

    @torch.no_grad()
    def sample(self, weather: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Generate cold-hardiness predictions conditioned on weather.

        weather:     [B, T_seq, 6]  — normalized weather features
        num_samples: how many stochastic samples to draw (for uncertainty)

        Returns: [B, num_samples, T_seq, 1]
        """
        B, T_seq, _ = weather.shape
        device = weather.device

        # Encode condition once
        context = self.weather_encoder(weather)  # [B, T_seq, cond_dim]

        all_samples = []
        for _ in range(num_samples):
            # Start from pure Gaussian noise
            x = torch.randn(B, T_seq, 1, device=device)

            # DDPM reverse process
            for t_val in reversed(range(self.T)):
                t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)

                noise_pred = self.unet(x, t_batch, context)

                # Compute x_{t-1} from x_t and predicted noise
                beta_t = self.betas[t_val]
                alpha_t = self.alphas[t_val]
                alpha_bar = self.alphas_cumprod[t_val]

                # Mean of p(x_{t-1} | x_t)
                coef1 = 1.0 / alpha_t.sqrt()
                coef2 = beta_t / (1.0 - alpha_bar).sqrt()
                mean = coef1 * (x - coef2 * noise_pred)

                if t_val > 0:
                    # Variance: use beta_t as posterior variance
                    noise = torch.randn_like(x)
                    x = mean + beta_t.sqrt() * noise
                else:
                    x = mean

            all_samples.append(x.unsqueeze(1))  # [B, 1, T_seq, 1]

        return torch.cat(all_samples, dim=1)  # [B, num_samples, T_seq, 1]


# ─────────────────────────────────────────────
# 6. TRAINING LOOP EXAMPLE
# ─────────────────────────────────────────────


def train_one_epoch(model, dataloader, optimizer, device):
    """
    Example training loop for one epoch.

    Expected dataloader to yield:
        weather:       [B, T_seq, 6]   — normalized weather features
        cold_hardiness:[B, T_seq, 1]   — normalized cold-hardiness labels
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
# 7. QUICK SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    B, T_seq = 4, 365  # batch size, sequence length (e.g. 1 year of daily data)
    weather_dim = 6
    T_diff = 1000  # diffusion steps

    # Build model
    model = ColdHardinesseDiffusionModel(
        weather_dim=weather_dim,
        cond_dim=128,
        base_channels=64,
        channel_mults=(1, 2, 4),
        time_dim=128,
        nhead=4,
        T=T_diff,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # Dummy data
    weather = torch.randn(B, T_seq, weather_dim).to(device)
    cold_hardiness = torch.randn(B, T_seq, 1).to(device)

    # Training step test
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss = model.p_losses(cold_hardiness, weather)
    print(f"Training loss (random data): {loss.item():.4f}")
    loss.backward()
    print("Backward pass: OK")

    # Inference test (use fewer diffusion steps for speed in testing)
    model.T = 50  # override for quick test
    samples = model.sample(weather, num_samples=3)
    print(f"Sample output shape: {samples.shape}")  # [4, 3, 365, 1]
    print("Smoke test passed!")
