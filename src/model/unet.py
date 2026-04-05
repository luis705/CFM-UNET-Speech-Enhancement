import torch
import torch.nn as nn

from src.model.blocks import Bottleneck, DecoderBlock, EncoderBlock
from src.model.time_emb import SinusoidalTimeEmbedding


class CFMUnet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        projection_dim: int,
        n_blocks: int,
        compression_factor: int,
        channels_factor: int,
        num_heads: int,
        ff_mult: int,
        n_groups: int,
        gradient_checkpointing: bool,
    ):
        super().__init__()

        assert projection_dim % compression_factor == 0, "projection_dim must be divisible by compression_factor"

        # Realiza checkpoints
        self.gradient_checkpointing = gradient_checkpointing

        # Time embeddings
        self.time_embedder = SinusoidalTimeEmbedding(time_emb_dim)

        # Input projection
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, projection_dim, kernel_size=3, padding=1),
            nn.GroupNorm(n_groups, projection_dim),
            nn.GELU(),
        )

        # Encoder
        encoder_in_channels = projection_dim
        self.encoders = nn.ModuleList()
        for _ in range(n_blocks):
            self.encoders.append(
                EncoderBlock(
                    in_channels=encoder_in_channels,
                    stride=compression_factor,
                    channels_factor=channels_factor,
                    num_heads=num_heads,
                    ff_mult=ff_mult,
                    time_embedding_dim=time_emb_dim,
                    n_groups=n_groups,
                )
            )

            encoder_in_channels *= channels_factor

        # Bottleneck
        self.bottleneck = Bottleneck(
            in_channels=encoder_in_channels,
            num_heads=num_heads,
            ff_mult=ff_mult,
            time_embedding_dim=time_emb_dim,
            n_groups=n_groups,
        )

        # Decoder
        decoder_in_channels = encoder_in_channels
        self.decoders = nn.ModuleList()
        for _ in range(n_blocks):
            self.decoders.append(
                DecoderBlock(
                    in_channels=decoder_in_channels,
                    upsampling_factor=compression_factor,
                    channels_factor=channels_factor,
                    num_heads=num_heads,
                    ff_mult=ff_mult,
                    time_embedding_dim=time_emb_dim,
                    n_groups=n_groups,
                )
            )
            decoder_in_channels //= channels_factor

        # Output projection para mapear de volta para in_channels
        self.output_projection = nn.Sequential(
            nn.GroupNorm(n_groups, decoder_in_channels),
            nn.GELU(),
            nn.Conv2d(projection_dim, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.training and self.gradient_checkpointing and not x.requires_grad:
            x.requires_grad_(True)
        t_emb = self.time_embedder(t)
        x = self.stem(x)

        skips = []
        # O loop do encoder deve ir até n_blocks - 1 para o último skip ser usado no bottleneck
        # E o primeiro skip é o stem
        for encoder_block in self.encoders:
            if self.gradient_checkpointing and self.training:
                x, skip = torch.utils.checkpoint.checkpoint(encoder_block, x, t_emb, use_reentrant=True)
            else:
                x, skip = encoder_block(x, t_emb)
            skips.append(skip)

        if self.gradient_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(self.bottleneck, x, t_emb, use_reentrant=True)
        else:
            x = self.bottleneck(x, t_emb)

        for decoder_block in self.decoders:
            skip = skips.pop()
            if self.gradient_checkpointing and self.training:
                # O decoder recebe x, skip e t_emb
                x = torch.utils.checkpoint.checkpoint(decoder_block, x, skip, t_emb, use_reentrant=True)
            else:
                x = decoder_block(x, skip, t_emb)

        x = self.output_projection(x)
        return x


if __name__ == "__main__":
    # Test CFMUnet
    print("\n--- Testando CFMUnet ---")

    # Configurações de teste para a UNet completa
    B = 2  # Batch size
    C_in, C_out = 4, 2  # Canais de entrada e saída
    F_dim, T_dim = 256, 128  # Dimensões espaciais (Frequência e Tempo)
    time_emb_dim = 128
    projection_dim = 64  # Canais iniciais após a projeção de entrada
    n_blocks = 2  # Número de blocos encoder/decoder
    compression_factor = 4  # Fator de downsample/upsample
    channels_factor = 4  # Fator de multiplicação/divisão de canais
    num_heads = 4
    ff_mult = 2
    n_groups = 32  # Ou 32, dependendo da configuração

    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet_model = CFMUnet(
        in_channels=C_in,
        out_channels=C_out,
        time_emb_dim=time_emb_dim,
        projection_dim=projection_dim,
        n_blocks=n_blocks,
        compression_factor=compression_factor,
        channels_factor=channels_factor,
        num_heads=num_heads,
        ff_mult=ff_mult,
        n_groups=n_groups,
        gradient_checkpointing=False,
    ).to(device)

    x_input = torch.randn(B, C_in, F_dim, T_dim).to(device)
    t_input = torch.rand(B).to(device)  # Tempo entre 0 e 1

    print(f"Input UNet shape: {x_input.shape}")
    output_unet = unet_model(x_input, t_input)
    print(f"Output UNet shape: {output_unet.shape}")

    # A saída da UNet deve ter o mesmo número de canais de entrada (C_in)
    # e as mesmas dimensões espaciais (F_dim, T_dim)
    assert output_unet.shape == (B, C_out, F_dim, T_dim), (
        f"Erro: Shape de saída da CFMUnet incorreto."
        f"Esperado: ({B}, {C_out}, {F_dim}, {T_dim}), "
        f"Recebido: {output_unet.shape}"
    )

    print("\n✅ CFMUnet validada com sucesso!")
