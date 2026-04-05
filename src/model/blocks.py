# Encoder e decoder
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.attention import AxialAttention
from src.model.film import FiLM
from src.model.gates import AttentionGate


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels_factor: int,
        stride: int,
        num_heads: int,
        ff_mult: int,
        time_embedding_dim: int,
        n_groups: int = 32,
    ):
        super().__init__()
        out_channels = in_channels * channels_factor

        # 1. Projeção inicial para igualar os canais (Necessário para a soma residual)
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 2. Primeira extração de características
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(n_groups, out_channels),
            nn.GELU(),
        )

        # 3. Condicionamento temporal
        self.film = FiLM(out_channels, time_embedding_dim)

        # 4. Segunda extração de características
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(n_groups, out_channels),
            nn.GELU(),
        )

        # 5. Atenção axial
        self.attn = AxialAttention(out_channels, num_heads=num_heads, ff_mult=ff_mult)

        # 6. Downsample aprendível
        self.downsample = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(n_groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Iguala os canais
        x = self.proj(x)

        # Caminho residual
        h = self.conv1(x)
        h = self.film(h, t_emb)
        h = self.conv2(h)  # Refina sinal modulado pelo tempo
        x = x + h  # Conexão residual

        x = self.attn(x)

        skip = x  # Guarda para a U-Net antes do downsample
        x = self.downsample(x)
        return x, skip


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels_factor: int,
        upsampling_factor: int,
        num_heads: int,
        ff_mult: int,
        time_embedding_dim: int,
        n_groups: int = 32,
    ):
        super().__init__()
        out_channels = in_channels // channels_factor

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=upsampling_factor, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.GroupNorm(n_groups, out_channels),
            nn.GELU(),
        )

        self.gate = AttentionGate(gate_channels=out_channels, skip_channels=in_channels)

        # NOVA CAMADA: Projeção 1x1 idêntica a que existe no EncoderBlock
        # Serve para reduzir a concatenação (2 * out_channels) para out_channels de forma linear
        self.proj = nn.Conv2d(2 * out_channels, out_channels, kernel_size=1)

        # As convoluções agora operam estritamente em out_channels -> out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(n_groups, out_channels),
            nn.GELU(),
        )

        self.film = FiLM(out_channels, time_embedding_dim)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(n_groups, out_channels),
            nn.GELU(),
        )

        self.attn = AxialAttention(out_channels, num_heads=num_heads, ff_mult=ff_mult)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # 1. Sobe a resolução
        x = self.upsample(x)

        # Alinha as dimensões espaciais (H, W) caso haja descasamento (ex: 34 vs 33)
        # O F.pad com valores negativos faz crop automático se o upsample ficou maior que o skip
        x = F.pad(
            x,
            (
                0,
                skip.shape[3] - x.shape[3],  # Diferença na Largura (Tempo)
                0,
                skip.shape[2] - x.shape[2],  # Diferença na Altura (Frequência)
            ),
        )

        # 2. Pede para a porta filtrar o skip baseado no x atual
        gated_skip = self.gate(x, skip)

        # 3. CONCATENAÇÃO (x e gated_skip têm out_channels, o resultado tem in_channels)
        x_concat = torch.cat([x, gated_skip], dim=1)

        # 4. Projeção (Gera a base com 'out_channels' para podermos usar o fluxo residual)
        x_res = self.proj(x_concat)

        # 5. Caminho convolucional com soma residual protegida (O bug 2 resolvido)
        h = self.conv1(x_res)
        h = self.film(h, t_emb)
        h = self.conv2(h)
        x = x_res + h

        # 6. Atenção (Sem somar h novamente, o bug 1 resolvido!)
        x = self.attn(x)

        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels: int, num_heads: int, ff_mult: int, time_embedding_dim: int, n_groups: int = 32):
        super().__init__()

        # --- Primeiro Caminho Residual ---
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(n_groups, in_channels),
            nn.GELU(),
        )
        self.film1 = FiLM(in_channels, time_embedding_dim)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(n_groups, in_channels),
            nn.GELU(),
        )

        # --- O Miolo de Atenção ---
        self.attn = AxialAttention(in_channels, num_heads=num_heads, ff_mult=ff_mult)

        # --- Segundo Caminho Residual---
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(n_groups, in_channels),
            nn.GELU(),
        )
        self.film2 = FiLM(in_channels, time_embedding_dim)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(n_groups, in_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # Passagem 1: Residual Local
        h = self.conv1(x)
        h = self.film1(h, t_emb)
        h = self.conv2(h)
        x = x + h

        # Passagem 2: Atenção Global
        x = self.attn(x)

        # Passagem 3: Refinamento Local (Nova)
        h = self.conv3(x)
        h = self.film2(h, t_emb)
        h = self.conv4(h)
        x = x + h

        return x


if __name__ == "__main__":
    # Configurações experimentais
    B, C_in, F, T = 2, 32, 64, 128
    time_dim = 128

    print(f"Configuração Inicial: B={B}, C={C_in}, F={F}, T={T}, Time_dim={time_dim}")

    # 1. Testando EncoderBlock
    # O Encoder dobra os canais e reduz a resolução espacial pela metade
    print("\n--- Testando EncoderBlock ---")
    encoder = EncoderBlock(
        in_channels=C_in, channels_factor=2, stride=2, num_heads=4, ff_mult=4, time_embedding_dim=time_dim
    )

    x_in = torch.randn(B, C_in, F, T)
    t_emb = torch.randn(B, time_dim)

    x_out, enc_skip = encoder(x_in, t_emb)

    print(f"Input shape:  {x_in.shape}")
    print(f"Skip shape:   {enc_skip.shape}  (Canais dobrados)")
    print(f"Output shape: {x_out.shape} (Resolução reduzida)")

    assert enc_skip.shape == (B, C_in * 2, F, T)
    assert x_out.shape == (B, C_in * 2, F // 2, T // 2)

    # 2. Testando Bottleneck
    print("\n--- Testando Bottleneck ---")
    bot = Bottleneck(in_channels=C_in * 2, num_heads=4, ff_mult=4, time_embedding_dim=time_dim)

    x_out = bot(x_out, t_emb)

    print(f"Input shape:  {x_out.shape}")
    print(f"Output shape: {x_out.shape} (Resolução reduzida)")

    assert x_out.shape == (B, C_in * 2, F // 2, T // 2)

    # 2. Testando DecoderBlock
    # O Decoder recebe a saída do nível inferior, faz upsample e processa o skip
    print("\n--- Testando DecoderBlock ---")
    # Para o decoder, 'in_channels' é a entrada que vem de baixo (C_in * 2)
    # Ele deve reduzir os canais para casar com o skip do nível atual (que aqui simulamos como C_in)
    decoder = DecoderBlock(
        in_channels=C_in * 2,
        channels_factor=2,
        upsampling_factor=2,
        num_heads=4,
        ff_mult=4,
        time_embedding_dim=time_dim,
    )

    # O skip deve ter o mesmo número de canais que o out_channels do decoder (in_channels // 2)
    dec_out = decoder(x_out, enc_skip, t_emb)

    print(f"Decoder output shape: {dec_out.shape}")
    assert dec_out.shape == (B, C_in, F, T)

    print("\n✅ Blocos de Encoder e Decoder validados com sucesso!")
