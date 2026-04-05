from __future__ import annotations

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.attention import _apply_rope, _build_rope_cache


class AttentionGate(nn.Module):
    """Additive Attention Gate for skip connections.

    Takes the gating signal from the decoder and the skip feature map
    from the encoder, computes an attention coefficient, and returns
    the element-wise product with the skip feature.
    """

    def __init__(self, gate_channels: int, skip_channels: int) -> None:
        super().__init__()
        inter = skip_channels // 2 or 1
        self.w_gate = nn.Conv2d(gate_channels, inter, kernel_size=1, bias=False)
        self.w_skip = nn.Conv2d(skip_channels, inter, kernel_size=1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv2d(inter, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

        # Projeção para compatibilizar o skip com os canais do decoder (gate_channels)
        self.skip_proj = nn.Conv2d(skip_channels, gate_channels, kernel_size=1)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gate: (B, gate_ch, H, W) - upsampled decoder features.
            skip: (B, skip_ch, H, W) - encoder features.
        Returns:
            (B, skip_ch, H, W) - gated skip features.
        """
        g = self.w_gate(gate)
        s = self.w_skip(skip)
        alpha = self.psi(self.relu(g + s))  # (B, 1, H, W)
        return self.skip_proj(skip) * alpha


# Se quiser usar cross attention na porta:
class _AxisCrossAttention(nn.Module):
    """Motor 1D de Atenção Cruzada com RoPE."""

    def __init__(self, q_channels: int, kv_channels: int, num_heads: int, max_seq_len: int = 8192) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = q_channels // num_heads
        assert q_channels % num_heads == 0, "Canais da Query devem ser divisíveis pelo número de cabeças"

        # Projeções separadas para Query (Decoder) e Key/Value (Encoder)
        self.proj_q = nn.Linear(q_channels, q_channels)
        self.proj_kv = nn.Linear(kv_channels, 2 * q_channels)
        self.proj_out = nn.Linear(q_channels, q_channels)

        # Cache do RoPE pré-calculado na CPU, movido automaticamente via register_buffer
        # (Substitua _build_rope_cache pela sua função real)
        rope_cache = _build_rope_cache(max_seq_len, self.head_dim, torch.device("cpu"))
        self.register_buffer("rope_cache", rope_cache, persistent=False)

    def forward(self, q_input: torch.Tensor, kv_input: torch.Tensor) -> torch.Tensor:
        _B, S, _C = q_input.shape

        # 1. Prepara a Query
        q = self.proj_q(q_input).unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)

        # 2. Prepara Key e Value
        k, v = self.proj_kv(kv_input).chunk(2, dim=-1)
        k = k.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        v = v.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)

        # 3. Fatiamento do RoPE e Rotação Espacial
        # Garante que as perguntas e respostas estejam alinhadas no mesmo sistema de coordenadas
        rope = self.rope_cache[:S]
        q, k = _apply_rope(q, k, rope)

        # 4. SDPA Mágico
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h s d -> b s (h d)")
        return self.proj_out(out)


class AxialCrossAttentionGate(nn.Module):
    """Porta de Atenção Cruzada Axial para Skip Connections."""

    def __init__(self, gate_channels: int, skip_channels: int, num_heads: int = 4) -> None:
        super().__init__()

        # Eixo da Frequência
        self.norm_q_f = nn.LayerNorm(gate_channels)
        self.norm_kv_f = nn.LayerNorm(skip_channels)
        self.attn_freq = _AxisCrossAttention(gate_channels, skip_channels, num_heads)

        # Eixo do Tempo
        self.norm_q_t = nn.LayerNorm(gate_channels)
        self.norm_kv_t = nn.LayerNorm(skip_channels)
        self.attn_time = _AxisCrossAttention(gate_channels, skip_channels, num_heads)

        # Projeção para compatibilizar o skip (C_skip) com o output (C_gate) na soma residual
        self.skip_proj = nn.Conv2d(skip_channels, gate_channels, kernel_size=1)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        gate: (B, C, F, T) -> Sinal do Decoder
        skip: (B, C, F, T) -> Mapa de características do Encoder
        """
        B, _C, F_dim, T_dim = gate.shape

        # --- 1. Atenção Cruzada no Eixo da Frequência ---
        q_f = rearrange(gate, "b c f t -> (b t) f c")
        kv_f = rearrange(skip, "b c f t -> (b t) f c")

        h_f = self.attn_freq(self.norm_q_f(q_f), self.norm_kv_f(kv_f))
        h_f = rearrange(h_f, "(b t) f c -> b c f t", b=B, t=T_dim)

        # --- 2. Atenção Cruzada no Eixo do Tempo ---
        # A Query agora é o mapa já refinado pela frequência
        q_t = rearrange(h_f, "b c f t -> (b f) t c")
        # Key/Value continuam sendo a "verdade" nua e crua do Encoder
        kv_t = rearrange(skip, "b c f t -> (b f) t c")

        h_t = self.attn_time(self.norm_q_t(q_t), self.norm_kv_t(kv_t))
        out = rearrange(h_t, "(b f) t c -> b c f t", b=B, f=F_dim)

        # Conexão residual clássica da porta de atenção
        return self.skip_proj(skip) + out


if __name__ == "__main__":
    # Configurações de teste simulando tensores de uma UNet (Ex: espectrogramas)
    B, C, F_dim, T_dim = 2, 64, 128, 256

    print(f"Configuração do teste: B={B}, C={C}, F={F_dim}, T={T_dim}")

    # Sinais de teste
    gate_signal = torch.randn(B, C, F_dim, T_dim)
    skip_signal = torch.randn(B, C, F_dim, T_dim)

    # 1. Testando AttentionGate (Porta aditiva baseada em CNN)
    print("\n--- Testando AttentionGate ---")
    gate_model = AttentionGate(gate_channels=C, skip_channels=C)
    output = gate_model(gate_signal, skip_signal)
    print(f"Input shape (Gate/Skip): {gate_signal.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == skip_signal.shape, "Erro: Shape de saída do AttentionGate está incorreto"

    # 2. Testando AxialCrossAttentionGate (Porta baseada em Cross-Attention)
    print("\n--- Testando AxialCrossAttentionGate ---")
    axial_gate_model = AxialCrossAttentionGate(gate_channels=C, skip_channels=C, num_heads=4)
    output_axial = axial_gate_model(gate_signal, skip_signal)
    print(f"Input shape (Gate/Skip): {gate_signal.shape}")
    print(f"Output shape: {output_axial.shape}")
    assert output_axial.shape == skip_signal.shape, "Erro: Shape de saída do AxialCrossAttentionGate está incorreto"

    # 3. Testando compatibilidade de canais diferentes (Cenário do erro no Decoder)
    print("\n--- Testando com canais diferentes (Gate=32, Skip=64) ---")
    C_gate, C_skip = 32, 64
    g_low = torch.randn(B, C_gate, F_dim, T_dim)
    s_high = torch.randn(B, C_skip, F_dim, T_dim)

    ag_diff = AttentionGate(gate_channels=C_gate, skip_channels=C_skip)
    out_diff = ag_diff(g_low, s_high)
    print(f"AttentionGate (32/64) Output shape: {out_diff.shape}")
    assert out_diff.shape == (B, C_gate, F_dim, T_dim)

    print("\n✅ Todos os módulos de gates foram validados com sucesso!")
