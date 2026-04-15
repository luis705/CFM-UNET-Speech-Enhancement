import torch
import torch.nn.functional as F


def cfm_loss(v_pred: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Calcula a perda do Conditional Flow Matching.

    Args:
        v_pred: O campo vetorial predito pela U-Net (B, C, F, T)
        x_0: Condição inicial (Áudio ruidoso) (B, C, F, T)
        x_1: Destino (Áudio limpo) (B, C, F, T)
    """
    # O alvo real é a derivada da trajetória linear no tempo
    v_target = x_1 - x_0

    # MSE Loss é o padrão para treinar o campo vetorial
    loss_unreduced = F.mse_loss(v_pred, v_target, reduction="none")
    weighted_loss = loss_unreduced * weight

    return weighted_loss.mean()


class STFTLoss(torch.nn.Module):
    def __init__(self, fft_size: int, hop_size: int, win_size: int):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x, y: (B, T)
        # Garantir float32 para estabilidade da STFT
        x = x.to(torch.float32)
        y = y.to(torch.float32)

        # Squeeze channel dimension if present (B, 1, T) -> (B, T)
        if x.ndim == 3:
            x = x.squeeze(1)
        if y.ndim == 3:
            y = y.squeeze(1)

        window = torch.hann_window(self.win_size).to(x.device)

        stft_x = torch.stft(x, self.fft_size, self.hop_size, self.win_size, window=window, return_complex=True)
        stft_y = torch.stft(y, self.fft_size, self.hop_size, self.win_size, window=window, return_complex=True)

        mag_x = torch.abs(stft_x) + 1e-7
        mag_y = torch.abs(stft_y) + 1e-7

        sc_loss = torch.norm(mag_y - mag_x, p="fro") / torch.norm(mag_y, p="fro")
        mag_loss = F.l1_loss(torch.log(mag_y), torch.log(mag_x))

        return sc_loss, mag_loss


class MRSTFTLoss(torch.nn.Module):
    def __init__(self, fft_sizes: list[int], hop_sizes: list[int], win_sizes: list[int]):
        super().__init__()
        self.stft_losses = torch.nn.ModuleList(
            [STFTLoss(f, h, w) for f, h, w in zip(fft_sizes, hop_sizes, win_sizes, strict=True)]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sc_loss_total = 0.0
        mag_loss_total = 0.0
        for loss_func in self.stft_losses:
            sc_loss, mag_loss = loss_func(x, y)
            sc_loss_total += sc_loss
            mag_loss_total += mag_loss

        return (sc_loss_total + mag_loss_total) / len(self.stft_losses)


class MelSpectralLoss(torch.nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, est_mel: torch.Tensor, target_mel: torch.Tensor) -> torch.Tensor:
        # est_mel, target_mel: (B, 1, Mels, T)
        # 1. Perda de Magnitude Logarítmica (L1 direta pois já estamos em log-mel)
        l_mag = F.l1_loss(est_mel, target_mel)

        # 2. Convergência Espectral (exige escala linear)
        est_mel_lin = torch.exp(est_mel)
        target_mel_lin = torch.exp(target_mel)

        sc_num = torch.linalg.norm(target_mel_lin - est_mel_lin, ord="fro", dim=(-2, -1))
        sc_den = torch.linalg.norm(target_mel_lin, ord="fro", dim=(-2, -1))
        l_sc = (sc_num / (sc_den + self.eps)).mean()

        return l_mag + l_sc
