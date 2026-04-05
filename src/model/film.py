import torch
import torch.nn as nn


class FiLM(nn.Module):
    def __init__(self, num_channels: int, time_embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(time_embedding_dim, num_channels * 2),
            nn.GELU(),
        )

        self.out = nn.Linear(num_channels * 2, num_channels * 2)
        self.out.final_film = True

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        non_linear = self.net(t)
        out = self.out(non_linear)
        gamma, beta = out.chunk(2, dim=1)

        missing_dims = max(x.dim() - gamma.dim(), 0)

        target_shape = gamma.shape + (1,) * missing_dims
        gamma = gamma.view(target_shape)
        beta = beta.view(target_shape)

        return x * (1 + gamma) + beta


if __name__ == "__main__":
    batch_size = 4
    num_channels = 64
    time_embedding_dim = 128

    film_layer = FiLM(num_channels=num_channels, time_embedding_dim=time_embedding_dim)

    # Simulando o tempo já codificado (t) que sai do SinusoidalTimeEmbedding
    t_emb = torch.randn(batch_size, time_embedding_dim)

    print("--- Testando FiLM ---")

    # 1. Features 2D (Batch, Channels) - Ex: um vetor latente achatado no bottleneck
    x_2d = torch.randn(batch_size, num_channels)
    out_2d = film_layer(x_2d, t_emb)
    print(f"Input 2D: {x_2d.shape} -> Output 2D: {out_2d.shape}")

    # 2. Features 3D (Batch, Channels, Length) - Ex: áudio raw (1D convolution)
    x_3d = torch.randn(batch_size, num_channels, 16000)
    out_3d = film_layer(x_3d, t_emb)
    print(f"Input 3D: {x_3d.shape} -> Output 3D: {out_3d.shape}")

    # 3. Features 4D (Batch, Channels, Freq, Time) - Ex: espectrograma (2D convolution)
    x_4d = torch.randn(batch_size, num_channels, 128, 256)
    out_4d = film_layer(x_4d, t_emb)
    print(f"Input 4D: {x_4d.shape} -> Output 4D: {out_4d.shape}")
