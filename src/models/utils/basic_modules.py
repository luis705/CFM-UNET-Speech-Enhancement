"""Building blocks for the split-band VelocityNet.

Tensor convention used throughout this module:
    B = batch size
    C = channel / feature dimension (hidden_dim of the network)
    F = frequency bins (from STFT — 257 bins for n_fft=512)
    T = time frames  (number of STFT windows along the temporal axis)

Modules defined here:
    SinusoidalTimeEmbedding — encodes the ODE flow time t ∈ [0,1] into a
        high-dimensional vector so the network knows "how noisy" x_t is.
    ResidualBlock — dilated 2-D convolution with skip connection (for HighBandNet).
    FiLM — Feature-wise Linear Modulation, injects time conditioning into
        feature maps via learned per-channel scale (y) and shift (β).
    RotaryPositionEmbedding (RoPE) — rotation-based positional encoding that
        makes attention translation-invariant (works at any audio length).
    SelfAttentionBlock — temporal Transformer block that attends along T
        (captures long-range dependencies like F0 and harmonics in the low band).
    CrossAttentionBlock — cross-attention along the frequency axis F,
        letting one band query information from the other band.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as nnF
from torch.utils.checkpoint import checkpoint as grad_checkpoint

# ── Sinusoidal time embedding ────────────────────────────────────────────────


class SinusoidalTimeEmbedding(nn.Module):
    """Encode the continuous flow time t ∈ [0,1] as a fixed-frequency vector.

    In CFM, t=0 means "fully noisy" and t=1 means "fully clean".  A single
    scalar carries almost no information for the network, so we project it
    onto sinusoidal basis functions of increasing frequency (like a Fourier
    feature), giving the network a rich representation of the current
    denoising progress.

    Output shape: (B, dim).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) scalar flow times → (B, dim) embedding vectors."""
        half = self.dim // 2
        freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = (10000.0 * t[:, None].float()) * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


# ── Residual block ───────────────────────────────────────────────────────────


class ResidualBlock(nn.Module):
    """Dilated 2-D residual convolution block.

    The dilation controls the receptive field: dilation=1 sees a 3x3 window,
    dilation=4 sees a 9x9 window without any extra parameters.  Stacking
    blocks with increasing dilation (e.g. [1,2,4,8]) lets the HighBandNet
    capture both fine detail and broader context in the high-frequency band.

    The skip connection (x + block(x)) helps gradients flow and keeps the
    block's job focused on predicting the *residual* correction.

    Input / output: (B, C, F, T) — shape is preserved.
    """

    def __init__(self, channels: int, dilation: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.Dropout(dropout),
            nn.GroupNorm(num_groups=1, num_channels=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ── FiLM conditioning (used to inject time embedding into conv features) ─────


class FiLM(nn.Module):
    """Feature-wise Linear Modulation — injects time conditioning into features.

    Given a conditioning vector τ (the time embedding), FiLM predicts
    per-channel scale (y) and shift (β) and modulates the feature map:

        out = x * (1 + y) + β

    This is the main mechanism by which the network knows "how far along"
    the denoising ODE we are.  Weights are initialised to zero so FiLM
    starts as an identity (y=0, β=0) and learns to steer features gently.
    """

    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, channels * 2),
            nn.GELU(),
        )

        self.output = nn.Linear(channels * 2, channels * 2)
        self.output._is_film_output = True  # flag for zero-init in SEModel._init_weights

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (B, C, F, T) feature map, cond: (B, cond_dim) time embedding."""
        features = self.net(cond)
        gamma_beta = self.output(features)  # (B, 2*C)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * (1.0 + gamma) + beta


# ── Rotary Position Embedding (RoPE) ─────────────────────────────────────────


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) — relative position via Q/K rotation.

    Unlike learnable positional embeddings that are fixed to a maximum
    sequence length, RoPE encodes *relative* distances between frames by
    rotating Q and K vectors.  This means:

    - The model generalises naturally from 2 s training chunks to 10 s+
      test audio without positional aliasing or extrapolation artefacts.
    - Attention scores depend only on the distance between two frames,
      not their absolute position — this is the "translation invariance"
      property that matters for variable-length speech.

    The rotation frequencies are pre-computed (no learnable parameters).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (10_000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (cos, sin) caches of shape (seq_len, dim)."""
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (T, dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, dim)
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap and negate halves of the last dim: [x1, x2] → [-x2, x1].

    This is the core rotation primitive used by RoPE — geometrically it
    rotates each pair of dimensions in the embedding space.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE rotation to query and key tensors.

    q, k shape: (…, T, head_dim) — the last two dims are sequence length
    and per-head feature dimension.  cos/sin come from
    RotaryPositionEmbedding.forward() and broadcast over batch & heads.
    """
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


# ── Sinusoidal Positional Encoding (frequency axis) ──────────────────────────


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding for variable-length sequences.

    Used in CrossAttentionBlock to make frequency-axis attention position-aware:
    each frequency bin knows its location in the band, enabling the model to
    learn harmonic patterns that depend on relative bin positions.

    Output: x + PE[:seq_len], shape preserved.
    """

    def __init__(self, dim: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10_000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)  # (max_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., seq_len, dim) → x + PE[:seq_len]."""
        return x + self.pe[: x.shape[-2]]


# ── Self-Attention Block (temporal, with RoPE) ───────────────────────────────


class SelfAttentionBlock(nn.Module):
    """Pre-norm Transformer block — self-attention along the temporal axis.

    Used by LowBandNet to capture long-range temporal patterns in the
    low-frequency STFT band (0-4 kHz): fundamental frequency contour,
    harmonic structure, and voiced/unvoiced transitions.

    Shape flow:
        Input:  (B, C, F, T)            ← B=batch, C=hidden channels,
                                            F=freq bins in this band,
                                            T=STFT time frames
        Reshape → (B*F, T, C)           ← treat each freq bin independently;
                                            attention operates over T (time)
        Self-attention + FFN             ← Q, K, V all come from the same
                                            input; RoPE is applied to Q and K
                                            so the model learns *relative*
                                            temporal relationships
        Reshape → (B, C, F, T)          ← restore original layout

    Uses PyTorch's F.scaled_dot_product_attention which automatically
    selects the fastest available kernel (Flash Attention 2, memory-
    efficient attention, or math fallback).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.gradient_checkpointing = gradient_checkpointing

        # Attention projections
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = dropout

        # FFN
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

        # RoPE (per head_dim)
        self.rope = RotaryPositionEmbedding(self.head_dim)

    def _attn_block(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-head self-attention with RoPE.

        x: (N, T, C) where N = B*F (each frequency bin is an independent
        sequence of T time frames with C features).  Returns same shape.
        """
        N, T, C = x.shape

        # Pre-norm → QKV
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(N, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, N, heads, T, head_dim)
        q, k, v = qkv.unbind(0)  # each (N, heads, T, head_dim)

        # Apply RoPE
        cos, sin = self.rope(T, x.device)  # (T, head_dim)
        q, k = apply_rope(q, k, cos, sin)

        # Flash / memory-efficient attention
        out = nnF.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
        )  # (N, heads, T, head_dim)
        out = out.transpose(1, 2).reshape(N, T, C)
        return self.out_proj(out)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Full block: attention + FFN with residuals. x: (N, T, C)."""
        x = x + self._attn_block(x)
        x = x + self.ffn(self.norm2(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, F, T) → (B, C, F, T)"""
        B, C, F, T = x.shape
        # (B, C, F, T) → (B*F, T, C)
        x = x.permute(0, 2, 3, 1).reshape(B * F, T, C)

        if self.gradient_checkpointing and self.training:
            x = grad_checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            x = self._forward_impl(x)

        # (B*F, T, C) → (B, C, F, T)
        return x.reshape(B, F, T, C).permute(0, 3, 1, 2)


# ── Cross-Attention Block (frequency axis) ───────────────────────────────────


class CrossAttentionBlock(nn.Module):
    """Cross-attention between two sequences — operates along the frequency axis.

    Used in CrossAttentionBottleneck in both directions:
        H → L:  Q=high-band, K/V=low-band  (harmonic reconstruction)
        L → H:  Q=low-band,  K/V=high-band (spectral continuity at boundary)

    Sinusoidal positional encoding (freq_pe) is applied to Q and KV inputs
    so the model knows each bin's position within its band.

    Shape flow:
        Q input:  (B, C, F_q,  T)
        KV input: (B, C, F_kv, T)
        Reshape → (B*T, F_q, C) / (B*T, F_kv, C)
                                    ← time frames treated independently;
                                       attention operates over F (frequency)
        Cross-attention + FFN
        Reshape → (B, C, F_q, T)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.gradient_checkpointing = gradient_checkpointing

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, 2 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = dropout

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )
        self.freq_pe = SinusoidalPositionalEncoding(dim)

    def _cross_attn(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        """Multi-head cross-attention.

        x_q:  (N, F_q, C)  — queries (N = B*T)
        x_kv: (N, F_kv, C) — keys & values
        Returns: (N, F_q, C) — attended output per query freq bin.
        """
        N, Fq, C = x_q.shape
        Fkv = x_kv.shape[1]

        q = self.q_proj(self.norm_q(self.freq_pe(x_q)))
        kv = self.kv_proj(self.norm_kv(self.freq_pe(x_kv)))
        k, v = kv.split(C, dim=-1)

        q = q.reshape(N, Fq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(N, Fkv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(N, Fkv, self.num_heads, self.head_dim).transpose(1, 2)

        out = nnF.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
        )  # (N, heads, Fq, head_dim)
        out = out.transpose(1, 2).reshape(N, Fq, C)
        return self.out_proj(out)

    def _forward_impl(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        x_q = x_q + self._cross_attn(x_q, x_kv)
        x_q = x_q + self.ffn(self.norm_ffn(x_q))
        return x_q

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
    ) -> torch.Tensor:
        """
        x_q:  (B, C, F_q, T)  — query source
        x_kv: (B, C, F_kv, T) — key/value source
        Returns: (B, C, F_q, T)
        """
        B, C, Fq, T = x_q.shape
        Fkv = x_kv.shape[2]

        # (B, C, F, T) → (B*T, F, C)
        q = x_q.permute(0, 3, 2, 1).reshape(B * T, Fq, C)
        kv = x_kv.permute(0, 3, 2, 1).reshape(B * T, Fkv, C)

        if self.gradient_checkpointing and self.training:
            q = grad_checkpoint(self._forward_impl, q, kv, use_reentrant=False)
        else:
            q = self._forward_impl(q, kv)

        # (B*T, Fq, C) → (B, C, Fq, T)
        return q.reshape(B, T, Fq, C).permute(0, 3, 2, 1)
