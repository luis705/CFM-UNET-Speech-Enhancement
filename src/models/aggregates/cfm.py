"""Split-band CFM velocity network and ODE samplers.

This module implements the core of the speech enhancement model.  Instead of
operating in a latent space (with a separate encoder/decoder), everything
happens directly on the complex STFT representation:

    Audio → STFT → VelocityNet v_θ(x_t, x_noisy, t) → ODE integration → iSTFT → Audio

The velocity network is *split-band*: frequency bins are divided at a
cutoff (f_split, typically 4 kHz) into a low band and a high band.  Each
band is processed by a specialised sub-network:

    - Low band (0-4 kHz): Self-attention Transformer blocks — good at
      modelling long-range temporal patterns like pitch contours and
      harmonic structure.
    - High band (4-8 kHz): Dilated CNN residual blocks — efficient at
      capturing local textures, transients, and noise patterns.

After independent processing, a cross-attention bottleneck fuses the two
bands: the high band *queries* information from the low band so that
its lost harmonic overtones can be reconstructed using the strong
fundamental (F0) cues present in the low frequencies.

The ODE samplers (direct_solve, ode_solve) integrate the learned velocity
field from t=0 (noisy) to t=1 (clean) at inference time.

Tensor convention:
    B = batch size
    C = channel dimension (2 for real/imag STFT, or hidden_dim internally)
    F = total frequency bins (e.g. 257 for n_fft=512)
    T = STFT time frames
    H = hidden_dim (internal channel width of the network)
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint

from src.models.utils.basic_modules import (
    CrossAttentionBlock,
    FiLM,
    ResidualBlock,
    SelfAttentionBlock,
    SinusoidalTimeEmbedding,
)

# ── Low-Band Network (Self-Attention) ────────────────────────────────────────


class LowBandNet(nn.Module):
    """Process the low-frequency STFT band (0-4 kHz) with self-attention.

    Speech fundamentals live in the low band: the fundamental frequency (F0),
    harmonic overtones, formant transitions, and voiced/unvoiced boundaries.
    These patterns span many consecutive STFT frames, so we use Transformer
    blocks with self-attention along the *temporal* axis to capture them.

    Each layer consists of:
        1. SelfAttentionBlock — attends over T (time frames) independently
           per frequency bin, with RoPE for relative-position awareness.
        2. FiLM modulation — injects the flow-time embedding τ so the
           network adapts its behaviour depending on how noisy x_t is.

    Input / output shape: (B, C, F_low, T)
        B      = batch size
        C      = hidden_dim (channel width)
        F_low  = number of freq bins in the low band (0 … f_split-1)
        T      = number of STFT time frames
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        time_emb_dim: int = 128,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.films = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                SelfAttentionBlock(
                    dim,
                    num_heads=num_heads,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    gradient_checkpointing=gradient_checkpointing,
                )
            )
            self.films.append(FiLM(time_emb_dim, dim))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, C, F_low, T) — low-band STFT features after input projection.
            t_emb: (B, D_time)      — time embedding vector (from sinusoidal + MLP).

        Returns:
            (B, C, F_low, T) — temporally refined low-band features.
        """
        for attn, film in zip(self.layers, self.films, strict=True):
            x = attn(x)  # self-attention along T (temporal axis)
            x = film(x, t_emb)  # scale & shift features based on flow time
        return x


# ── High-Band Network (CNN Residual) ─────────────────────────────────────────


class HighBandNet(nn.Module):
    """Process the high-frequency STFT band (4-8 kHz) with dilated CNNs.

    The high band is dominated by noise textures, sibilants, plosive
    transients, and aperiodic energy — patterns that are spatially local
    in time-frequency.  Dilated 2-D convolutions are a natural fit here:
    stacking dilations [1, 2, 4, 8] grows the receptive field
    exponentially without increasing parameters, capturing context from
    fine detail up to ~100 ms without the cost of full attention.

    Each layer consists of:
        1. ResidualBlock — dilated Conv2d with skip connection.
        2. FiLM modulation — time-dependent scale & shift (same as LowBandNet).

    Input / output shape: (B, C, F_high, T)
        B      = batch size
        C      = hidden_dim (channel width)
        F_high = number of freq bins in the high band (f_split … F-1)
        T      = number of STFT time frames
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 4,
        dilations: list[int] | None = None,
        dropout: float = 0.1,
        time_emb_dim: int = 128,
    ) -> None:
        super().__init__()
        dilations = dilations or [1, 2, 4, 8]
        if len(dilations) != num_layers:
            # Cycle dilations to match num_layers
            dilations = [dilations[i % len(dilations)] for i in range(num_layers)]

        self.layers = nn.ModuleList()
        self.films = nn.ModuleList()
        for dil in dilations:
            self.layers.append(ResidualBlock(dim, dilation=dil, dropout=dropout))
            self.films.append(FiLM(time_emb_dim, dim))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, C, F_high, T) — high-band STFT features after input projection.
            t_emb: (B, D_time)       — time embedding vector.

        Returns:
            (B, C, F_high, T) — locally refined high-band features.
        """
        for res, film in zip(self.layers, self.films, strict=True):
            x = res(x)  # dilated residual convolution
            x = film(x, t_emb)  # time-conditioned modulation
        return x


# ── Cross-Attention Bottleneck ────────────────────────────────────────────────


class CrossAttentionBottleneck(nn.Module):
    """Bidirectional cross-attention bottleneck fusing the two frequency bands.

    After LowBandNet and HighBandNet process their bands independently,
    this module lets them exchange information in *both* directions:

        High → Low  (Q=high, K/V=low):
            The high-frequency band queries the low band for F0 / harmonic
            cues to reconstruct its lost overtone structure.

        Low → High  (Q=low, K/V=high):
            The low-frequency band queries the high band for textural
            context and ensures spectral continuity at f_split.

    Each direction has its own CrossAttentionBlock, GroupNorm, and FiLM.

    Input / output shapes:
        x_high: (B, C, F_high, T) — high-band features (needs harmonic cues)
        x_low:  (B, C, F_low,  T) — low-band features (has strong F0)
        output: tuple (x_high_enriched, x_low_enriched)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        time_emb_dim: int = 128,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        # High band queries low band for harmonic reconstruction (Q=high, K/V=low)
        self.cross_attn_hl = CrossAttentionBlock(
            dim,
            num_heads=num_heads,
            ff_mult=ff_mult,
            dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.norm_high = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.film_high = FiLM(time_emb_dim, dim)
        # Low band queries high band for spectral continuity at the boundary (Q=low, K/V=high)
        self.cross_attn_lh = CrossAttentionBlock(
            dim,
            num_heads=num_heads,
            ff_mult=ff_mult,
            dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.norm_low = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.film_low = FiLM(time_emb_dim, dim)

    def forward(
        self,
        x_high: torch.Tensor,
        x_low: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_high: (B, C, F_high, T) — high-band features (needs harmonic reconstruction)
            x_low:  (B, C, F_low, T)  — low-band features (has strong F0 / harmonic cues)
            t_emb:  (B, D_time)       — time embedding vector

        Returns:
            x_high_enriched: (B, C, F_high, T) — high-band enriched with low-band cues
            x_low_enriched:  (B, C, F_low,  T) — low-band enriched with high-band detail
        """
        # High band queries low band for harmonic reconstruction
        out_high = self.cross_attn_hl(x_high, x_low)  # (B, C, F_high, T)
        out_high = self.norm_high(out_high)
        out_high = self.film_high(out_high, t_emb)
        # Low band queries high band for spectral continuity at the boundary
        out_low = self.cross_attn_lh(x_low, x_high)  # (B, C, F_low, T)
        out_low = self.norm_low(out_low)
        out_low = self.film_low(out_low, t_emb)
        return out_high, out_low


# ── Velocity Network (Split-Band, STFT domain) ───────────────────────────────


class VelocityNet(nn.Module):
    """Split-band velocity predictor — the core of the CFM speech enhancer.

    This network takes the current noisy-clean interpolation x_t, the
    original noisy STFT, and the flow time t, and predicts the velocity
    field v_θ that tells the ODE solver which direction to move in
    STFT space to get closer to clean speech.

    The full forward pipeline:

      1. **Input projection** — Concatenate x_t and x_noisy along the
         channel axis → (B, 4, F, T) and project to hidden_dim H
         with a Conv2d → (B, H, F, T).  The 4 input channels are
         [real_xt, imag_xt, real_noisy, imag_noisy].

      2. **Time embedding** — Encode the scalar t ∈ [0,1] with
         sinusoidal features + a 2-layer MLP → τ ∈ R^{D_time}.

      3. **Band split** — Slice the frequency axis F at f_split:
            low  = bins [ 0 … f_split-1 ]   (0-4 kHz, fundamentals)
            high = bins [ f_split … F-1  ]   (4-8 kHz, textures)

      4. **Parallel branches** — LowBandNet (self-attention) processes
         the low band; HighBandNet (dilated CNN) processes the high band.
         Both are conditioned on τ via FiLM.

      5. **Cross-attention bottleneck** — The high band queries the low
         band (Q=high, K/V=low) so the high band can reconstruct its
         harmonic overtones using the fundamental cues from the low band.

      6. **Reconstruct** — Concatenate low + enriched high along F.

      7. **U-Net skip** — Concatenate the input-projection features
         (saved from step 1) with the reconstructed features along the
         channel axis, preserving fine phase/magnitude information that
         would otherwise be lost through the attention bottleneck.

      8. **Decoder** — A shallow CNN maps 2*H channels back to 2 (real +
         imaginary parts) → v_θ of shape (B, 2, F, T).
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        f_split: int = 128,
        time_emb_dim: int = 128,
        # LowBand config
        low_num_layers: int = 4,
        low_num_heads: int = 4,
        low_ff_mult: int = 4,
        low_dropout: float = 0.1,
        # HighBand config
        high_num_layers: int = 4,
        high_dilations: list[int] | None = None,
        high_dropout: float = 0.1,
        # Bottleneck config
        bottleneck_num_heads: int = 4,
        bottleneck_dropout: float = 0.1,
        # Decoder config
        decoder_num_layers: int = 2,
        # Memory
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.f_split = f_split

        # ── Time embedding: scalar t → sinusoidal features → 2-layer MLP → τ ──
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.GELU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        # ── Input projection ──
        # x_t and x_noisy each have 2 channels (real + imag), so we
        # concatenate them → 4 channels, then project to hidden_dim H.
        self.input_proj = nn.Sequential(
            nn.Conv2d(4, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=hidden_dim),
            nn.GELU(),
        )

        # ── Parallel processing branches (one per band) ──
        self.low_band = LowBandNet(
            dim=hidden_dim,
            num_layers=low_num_layers,
            num_heads=low_num_heads,
            ff_mult=low_ff_mult,
            dropout=low_dropout,
            time_emb_dim=time_emb_dim,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.high_band = HighBandNet(
            dim=hidden_dim,
            num_layers=high_num_layers,
            dilations=high_dilations,
            dropout=high_dropout,
            time_emb_dim=time_emb_dim,
        )

        # ── Cross-attention bottleneck (high queries low for harmonic reconstruction) ──
        self.bottleneck = CrossAttentionBottleneck(
            dim=hidden_dim,
            num_heads=bottleneck_num_heads,
            dropout=bottleneck_dropout,
            time_emb_dim=time_emb_dim,
            gradient_checkpointing=gradient_checkpointing,
        )

        # ── Shallow CNN decoder → map H hidden channels back to 2 (real + imag) ──
        # First layer accepts hidden_dim*2 due to U-Net skip connection from input_proj
        dec_layers: list[nn.Module] = [
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=hidden_dim),
            nn.GELU(),
        ]
        for _ in range(decoder_num_layers - 1):
            dec_layers.extend(
                [
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups=1, num_channels=hidden_dim),
                    nn.GELU(),
                ]
            )
        dec_layers.append(nn.Conv2d(hidden_dim, 2, kernel_size=1))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(
        self,
        x_t: torch.Tensor,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_t:     (B, 2, F, T) — STFT interpolation between noisy (t=0) and
                     clean (t=1).  The 2 channels are [real, imaginary].
            x_noisy: (B, 2, F, T) — original noisy STFT (conditioning signal).
            t:       (B,)         — flow time ∈ [0,1] for each sample in the batch.

        Returns:
            (B, 2, F, T) — predicted velocity v_θ in STFT space.
                           During inference the ODE solver integrates this
                           velocity from t=0 → t=1 to denoise the signal.
        """
        # 1) Time embedding: scalar t → rich vector τ
        t_emb = self.time_mlp(self.time_emb(t))  # (B, time_emb_dim)

        # 2) Input projection: stack real/imag of x_t and x_noisy → 4ch → H ch
        x = torch.cat([x_t, x_noisy], dim=1)  # (B, 4, F, T)
        x = self.input_proj(x)  # (B, H, F, T)

        # Save for U-Net skip connection (bypass around attention bottleneck)
        x_skip = x  # (B, H, F, T)

        # 3) Split along frequency into low band (fundamentals) and high band (textures)
        x_low = x[:, :, : self.f_split, :]  # (B, H, F_low, T)
        x_high = x[:, :, self.f_split :, :]  # (B, H, F_high, T)

        # 4) Process each band with its specialised sub-network
        x_low_out = self.low_band(x_low, t_emb)  # (B, H, F_low, T) — attention-refined
        x_high_out = self.high_band(x_high, t_emb)  # (B, H, F_high, T) — CNN-refined

        # 5) Cross-attention bottleneck: bidirectional band exchange
        x_high_enriched, x_low_enriched = self.bottleneck(x_high_out, x_low_out, t_emb)

        # 6) Reconstruct full-bandwidth representation
        x_full = torch.cat([x_low_enriched, x_high_enriched], dim=2)  # (B, H, F, T)

        # 7) U-Net skip connection: concat input_proj features with processed features
        x_full = torch.cat([x_full, x_skip], dim=1)  # (B, 2*H, F, T)

        # 8) Decode back to 2-channel velocity field [real, imag]
        v = self.decoder(x_full)  # (B, 2, F, T)
        return v


# ── OT-CFM Interpolant ────────────────────────────────────────────────────────


def sample_ot_interpolant(
    z_clean: torch.Tensor,
    z_noisy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a random flow-time t and build the OT-CFM training pair.

    Optimal-Transport Conditional Flow Matching defines a *linear*
    interpolation between noisy (t=0) and clean (t=1):

        x_t = (1 - t) · x_noisy  +  t · x_clean

    The ground-truth velocity for this linear path is simply:

        v_target = x_clean - x_noisy      (constant along the path)

    The network is trained to predict v_target from (x_t, x_noisy, t).

    Args:
        z_clean: (B, 2, F, T) — clean STFT target.
        z_noisy: (B, 2, F, T) — noisy STFT input.

    Returns:
        z_t:      (B, 2, F, T) — interpolated STFT at random time t.
        t:        (B,)         — sampled flow times ∈ [0,1].
        v_target: (B, 2, F, T) — ground-truth OT velocity.
        z_noisy:  (B, 2, F, T) — passthrough for convenience.
    """
    B = z_clean.shape[0]
    # Logit-normal sampling: concentrates t near 0.5 where prediction is hardest,
    # improving training efficiency vs. uniform sampling (cf. Stable Diffusion 3).
    t = torch.sigmoid(torch.randn(B, device=z_clean.device, dtype=z_clean.dtype))
    t_view = t[:, None, None, None]
    z_t = (1.0 - t_view) * z_noisy + t_view * z_clean
    v_target = z_clean - z_noisy
    return z_t, t, v_target, z_noisy


# ── ODE Samplers ─────────────────────────────────────────────────────────────


def direct_solve(
    velocity_fn,
    z_noisy: torch.Tensor,
) -> torch.Tensor:
    """One-shot denoising: predict velocity at t=0 and jump to t=1.

    Since the OT flow is (ideally) linear, a single Euler step from t=0
    to t=1 gives:

        x̂_clean = x_noisy + 1 · v_θ(x_noisy, t=0)

    This is the fastest possible inference (one network evaluation) but
    lower quality than multi-step ODE integration.  Useful for quick
    previews or as a warm-start.
    """
    t = torch.zeros(z_noisy.shape[0], device=z_noisy.device, dtype=z_noisy.dtype)
    v = velocity_fn(z_noisy, t)
    return z_noisy + v


def ode_solve(
    velocity_fn,
    z_noisy: torch.Tensor,
    method: str = "midpoint",
    num_steps: int = 25,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> torch.Tensor:
    """Integrate the velocity ODE from t=0 (noisy) to t=1 (clean).

    Solves  dx/dt = v_θ(x, t)  starting at x(0) = z_noisy.
    The solution x(1) is the denoised estimate.

    Fixed-step solvers (euler, midpoint) divide [0,1] into num_steps
    equally-spaced intervals.  The adaptive solver (dopri5) chooses
    its own step size targeting the given atol/rtol tolerances.

    More steps → higher quality but slower inference.  Typical:
        - Training / fast val:  euler, 10 steps
        - Validation:           midpoint, 20 steps
        - Test / production:    midpoint 50 steps or dopri5

    Args:
        velocity_fn: callable (z, t_batch) → velocity, produced by
                     SEModel._velocity_fn() which closes over x_noisy.
        z_noisy:  (B, 2, F, T) — initial state (noisy STFT).
        method:   ODE solver name — 'euler', 'midpoint', 'dopri5', etc.
        num_steps: number of fixed steps (euler/midpoint only).
        atol:     absolute tolerance (dopri5 / adaptive solvers).
        rtol:     relative tolerance (dopri5 / adaptive solvers).

    Returns:
        (B, 2, F, T) — estimated clean STFT at t=1.
    """

    class ODEFunc(torch.nn.Module):
        def __init__(self, vfn):
            super().__init__()
            self.vfn = vfn

        def forward(self, t, z):
            t_batch = t.expand(z.shape[0])
            return self.vfn(z, t_batch)

    func = ODEFunc(velocity_fn)
    t_span = torch.linspace(0.0, 1.0, num_steps + 1, device=z_noisy.device, dtype=z_noisy.dtype)

    options = {"step_size": 1.0 / num_steps} if method in ("euler", "midpoint") else {}
    sol = odeint(func, z_noisy, t_span, method=method, atol=atol, rtol=rtol, options=options)
    return sol[-1]
