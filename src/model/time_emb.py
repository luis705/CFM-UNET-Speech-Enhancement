import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    r"""Calculates the sin/cos time embedding.

    Given a time step t as input we generate a vector of size dim in which the elements follow:

    PE(t, 2i) = \sin \left(  \dfrac{t}{10000^{\frac{2i}{dim}}} \right)
    PE(t, 2i + 1) = \cos \left(  \dfrac{t}{10000^{\frac{2i}{dim}}} \right)

    The geometric progression of frequencies emulates, in a certain way, the human sound perception
    Args:
        dim (int): number of elements on the embedding vector
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, f"A dimensão do embedding temporal deve ser par. Recebido: {dim}"
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1)

        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


if __name__ == "__main__":
    embedder = SinusoidalTimeEmbedding(128)

    # 1. Tensor 1D padrão (Batch,)
    inp_1d = torch.tensor([0.1, 0.5, 0.9])
    print(f"1D Input {inp_1d.shape} -> Output shape: {embedder(inp_1d).shape}")  # Esperado: [3, 128]

    # 2. Escalar (0D) - Simula a inferência gerando áudio passo a passo
    inp_scalar = torch.tensor(0.5)
    print(f"Scalar Input {inp_scalar.shape} -> Output shape: {embedder(inp_scalar).shape}")  # Esperado: [1, 128]

    # 3. Tensor 2D (Batch, 1) - Formato chato comum que sai de alguns dataloaders
    inp_2d = torch.tensor([[0.2], [0.4], [0.8]])
    print(f"2D Input {inp_2d.shape} -> Output shape: {embedder(inp_2d).shape}")  # Esperado: [3, 128]
