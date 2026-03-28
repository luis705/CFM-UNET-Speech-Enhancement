"""Loss functions for CFM-SE."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio

from src.data.transforms import get_window

# ── Sigmoid warmup schedule ──────────────────────────────────────────────────


def sigmoid_warmup(epoch: int | float, lambda_start: float, lambda_end: float, e0: float, k: float) -> float:
    """lambda(e) = lambda_start + (lambda_end - lambda_start) * sigmoid(k*(e - e0))"""
    return lambda_start + (lambda_end - lambda_start) / (1.0 + math.exp(-k * (epoch - e0)))


# ── CFM loss (MSE on velocity) ───────────────────────────────────────────────


def cfm_loss(v_pred: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
    """MSE loss on predicted velocity (mean over all dims including batch)."""
    return F.mse_loss(v_pred, v_target)


# ── Multi-Resolution STFT Loss ──────────────────────────────────────────────


class MultiResolutionSTFTLoss(nn.Module):
    """Spectral convergence + log-magnitude STFT loss at multiple resolutions."""

    def __init__(
        self,
        fft_sizes: list[int] | None = None,
        hop_sizes: list[int] | None = None,
        win_sizes: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.fft_sizes = fft_sizes or [512, 1024, 2048]
        self.hop_sizes = hop_sizes or [128, 256, 512]
        self.win_sizes = win_sizes or [512, 1024, 2048]

    @staticmethod
    def _stft_mag(x: torch.Tensor, fft_size: int, hop_size: int, win_size: int) -> torch.Tensor:
        window = get_window("hann", win_size, x.device)
        spec = torch.stft(x, fft_size, hop_size, win_size, window=window, return_complex=True)
        return spec.abs()

    @staticmethod
    def _spectral_convergence(pred_mag: torch.Tensor, target_mag: torch.Tensor) -> torch.Tensor:
        # Per-sample Frobenius norms over (F, T) → (B,)
        diff_norm = torch.norm(target_mag - pred_mag, p="fro", dim=(-2, -1))
        target_norm = torch.norm(target_mag, p="fro", dim=(-2, -1)).clamp(min=1e-8)
        return diff_norm / target_norm

    @staticmethod
    def _log_stft_magnitude(pred_mag: torch.Tensor, target_mag: torch.Tensor) -> torch.Tensor:
        # Per-sample L1 on log-magnitude, mean over (F, T) → (B,)
        diff = torch.abs(torch.log(pred_mag.clamp(min=1e-7)) - torch.log(target_mag.clamp(min=1e-7)))
        return diff.mean(dim=(-2, -1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred, target: (B, T) waveforms → (B,) per-sample loss."""
        loss = torch.zeros(pred.shape[0], device=pred.device)
        for fft_s, hop_s, win_s in zip(self.fft_sizes, self.hop_sizes, self.win_sizes, strict=True):
            pred_mag = self._stft_mag(pred, fft_s, hop_s, win_s)
            target_mag = self._stft_mag(target, fft_s, hop_s, win_s)
            loss = loss + self._spectral_convergence(pred_mag, target_mag)
            loss = loss + self._log_stft_magnitude(pred_mag, target_mag)
        return loss / len(self.fft_sizes)


# ── Compressed Complex Loss ──────────────────────────────────────────────────
def compressed_complex_loss(
    pred_spec: torch.Tensor,
    target_spec: torch.Tensor,
    compression: float = 0.3,
) -> torch.Tensor:
    """Loss on compressed complex STFT with numerical stability."""

    # Sai do autocast para garantir precisão nos cálculos complexos
    with torch.amp.autocast(device_type="cuda", enabled=False):
        # Força float32 para evitar o erro de 'ComplexHalf' e instabilidades
        pred_complex = torch.complex(pred_spec[:, 0].float(), pred_spec[:, 1].float())
        target_complex = torch.complex(target_spec[:, 0].float(), target_spec[:, 1].float())

        # Adiciona um epsilon pequeno para estabilidade no gradiente da potência
        eps = 1e-12

        # Compressed magnitude: (abs + eps)^c
        pred_mag = (pred_complex.abs() + eps).pow(compression)
        target_mag = (target_complex.abs() + eps).pow(compression)

        # Compressed complex: mag^c * exp(j*phase)
        # O angle() e o exp(1j * ...) são muito sensíveis, float32 é obrigatório aqui
        pred_compressed = pred_mag * torch.exp(1j * pred_complex.angle())
        target_compressed = target_mag * torch.exp(1j * target_complex.angle())

        # L1 per sample on real/imaginary components, mean over (F, T) → (B,)
        loss_ri = (
            F.l1_loss(pred_compressed.real, target_compressed.real, reduction="none")
            + F.l1_loss(pred_compressed.imag, target_compressed.imag, reduction="none")
        ).mean(dim=(-2, -1))

        # L1 per sample on magnitude → (B,)
        loss_mag = F.l1_loss(pred_mag, target_mag, reduction="none").mean(dim=(-2, -1))

        return loss_ri + loss_mag