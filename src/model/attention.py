from __future__ import annotations

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------


def _build_rope_cache(seq_len: int, dim: int, device: torch.device) -> torch.Tensor:
    """Return (seq_len, dim) cos/sin interleaved rotation matrix."""
    assert dim % 2 == 0
    half = dim // 2
    theta = 1.0 / (10000.0 ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, theta)  # (seq_len, half)
    return torch.stack([freqs.cos(), freqs.sin()], dim=-1).reshape(seq_len, dim)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, rope: torch.Tensor):
    """Apply RoPE to query and key tensors.

    Args:
        q, k: (B, heads, seq_len, head_dim)
        rope:  (seq_len, head_dim)
    """

    def rotate(x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        d2 = x.shape[-1] // 2
        x1, x2 = x[..., :d2], x[..., d2:]
        cos = r[..., 0::2]  # (seq, d2)
        sin = r[..., 1::2]
        cos = cos.unsqueeze(0).unsqueeze(0).to(x.device)  # (1, 1, seq, d2)
        sin = sin.unsqueeze(0).unsqueeze(0).to(x.device)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    return rotate(q, rope), rotate(k, rope)


# ---------------------------------------------------------------------------
# Single-axis multi-head self-attention with RoPE
# ---------------------------------------------------------------------------


class _AxisAttention(nn.Module):
    """Multi-head self-attention along one spatial axis with RoPE."""

    def __init__(self, channels: int, num_heads: int, max_seq_len: int = 8192) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)

        rope_cache = _build_rope_cache(max_seq_len, self.head_dim, torch.device("cpu"))
        self.register_buffer("rope_cache", rope_cache, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, C)
        Returns:
            (B, seq_len, C)
        """
        B, S, _C = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, S, head_dim)
        q, k, v = qkv.unbind(0)

        # Apply RoPE
        rope = self.rope_cache[:S]
        q, k = _apply_rope(q, k, rope)

        # Scaled dot-product attention
        out = F.scaled_dot_product_attention(q, k, v)  # (B, heads, S, head_dim)
        out = rearrange(out, "b h s d -> b s (h d)")
        return self.proj(out)


# ---------------------------------------------------------------------------
# Feed-forward block
# ---------------------------------------------------------------------------


class _FeedForward(nn.Module):
    def __init__(self, channels: int, mult: int = 4) -> None:
        super().__init__()
        hidden = channels * mult
        self.net = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Axial Attention: attention along freq then time, with LayerNorm + residual
# ---------------------------------------------------------------------------


class AxialAttention(nn.Module):
    """Axial self-attention block applied sequentially over the
    frequency and time axes of a 2-D feature map, with RoPE on each axis.
    """

    def __init__(self, channels: int, num_heads: int = 4, ff_mult: int = 4) -> None:
        super().__init__()
        # Freq-axis
        self.norm_freq = nn.LayerNorm(channels)
        self.attn_freq = _AxisAttention(channels, num_heads)
        # Time-axis
        self.norm_time = nn.LayerNorm(channels)
        self.attn_time = _AxisAttention(channels, num_heads)
        # Feed-forward
        self.norm_ff = nn.LayerNorm(channels)
        self.ff = _FeedForward(channels, ff_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, F, T)
        Returns:
            (B, C, F, T)
        """
        B, _C, Fr, T = x.shape

        # --- Attention along frequency axis ---
        # reshape to (B*T, F, C)
        h = rearrange(x, "b c f t -> (b t) f c")
        h = h + self.attn_freq(self.norm_freq(h))
        x = rearrange(h, "(b t) f c -> b c f t", b=B, t=T)

        # --- Attention along time axis ---
        # reshape to (B*F, T, C)
        h = rearrange(x, "b c f t -> (b f) t c")
        h = h + self.attn_time(self.norm_time(h))
        x = rearrange(h, "(b f) t c -> b c f t", b=B, f=Fr)

        # --- Feed-forward ---
        h = rearrange(x, "b c f t -> (b f t) c").unsqueeze(1)  # (N, 1, C)
        h_flat = h.squeeze(1)  # (N, C)
        h_flat = h_flat + self.ff(self.norm_ff(h_flat))
        x = rearrange(h_flat, "(b f t) c -> b c f t", b=B, f=Fr, t=T)

        return x


if __name__ == "__main__":
    # Configurações de teste simulando um espectrograma
    # B: Batch, C: Canais, F: Frequência, T: Tempo
    B, C, F_dim, T_dim = 2, 64, 128, 256

    print(f"Configuração do teste: B={B}, C={C}, F={F_dim}, T={T_dim}")

    # 1. Testando AxialAttention (O módulo principal)
    print("\n--- Testando AxialAttention ---")
    axial_model = AxialAttention(channels=C, num_heads=4, ff_mult=4)
    x = torch.randn(B, C, F_dim, T_dim)
    output = axial_model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    assert x.shape == output.shape, "Erro: Shape de saída do AxialAttention está incorreto"

    # 2. Testando _AxisAttention individualmente (Simulando um eixo)
    print("\n--- Testando _AxisAttention (Eixo Único) ---")
    axis_model = _AxisAttention(channels=C, num_heads=4)
    # Simula entrada formatada para um eixo: (Batch * OutroEixo, SeqLen, Channels)
    x_axis = torch.randn(B * T_dim, F_dim, C)
    out_axis = axis_model(x_axis)
    print(f"Input axis shape:  {x_axis.shape}")
    print(f"Output axis shape: {out_axis.shape}")
    assert x_axis.shape == out_axis.shape, "Erro: Shape de saída do _AxisAttention está incorreto"

    # 3. Testando _FeedForward
    print("\n--- Testando _FeedForward ---")
    ff_model = _FeedForward(channels=C, mult=4)
    x_ff = torch.randn(B * F_dim * T_dim, C)
    out_ff = ff_model(x_ff)
    print(f"Input FF shape:  {x_ff.shape}")
    print(f"Output FF shape: {out_ff.shape}")
    assert x_ff.shape == out_ff.shape, "Erro: Shape de saída do _FeedForward está incorreto"

    print("\n✅ Todos os módulos de atenção foram validados com sucesso!")
