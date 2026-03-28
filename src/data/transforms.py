"""STFT / ISTFT transforms and audio normalization utilities."""

import torch
import torch.nn.functional as F


def get_window(window_type: str, win_length: int, device: torch.device) -> torch.Tensor:
    if window_type == "hann":
        return torch.hann_window(win_length, device=device)
    if window_type == "hamming":
        return torch.hamming_window(win_length, device=device)
    raise ValueError(f"Unknown window type: {window_type}")


def stft(
    waveform: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int = 512,
    window_type: str = "hann",
) -> torch.Tensor:
    """Compute STFT and return (B, 2, F, T) with real/imag channels."""
    window = get_window(window_type, win_length, waveform.device)
    # (B, F, T) complex
    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    # Stack real and imaginary → (B, 2, F, T)
    return torch.stack([spec.real, spec.imag], dim=1)


def istft(
    spec_2ch: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int = 512,
    window_type: str = "hann",
    length: int | None = None,
) -> torch.Tensor:
    """Inverse STFT from (B, 2, F, T) → (B, T_samples)."""
    with torch.amp.autocast(device_type="cuda", enabled=False):
        window = get_window(window_type, win_length, spec_2ch.device)

        # Força float32 para evitar o erro de ComplexHalf experimental
        real = spec_2ch[:, 0].float()
        imag = spec_2ch[:, 1].float()
        complex_spec = torch.complex(real, imag)

        return torch.istft(
            complex_spec,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            length=length,
        )


def normalize_waveform(waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero-mean, unit-variance normalization per sample. Returns (normed, mean, std)."""
    mean = waveform.mean(dim=-1, keepdim=True)
    std = waveform.std(dim=-1, keepdim=True).clamp(min=1e-8)
    return (waveform - mean) / std, mean, std


def denormalize_waveform(waveform: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return waveform * std + mean


def pad_to_segment(waveform: torch.Tensor, segment_length: int) -> torch.Tensor:
    """Right-pad waveform so its length is a multiple of segment_length."""
    length = waveform.shape[-1]
    if length >= segment_length:
        return waveform
    return F.pad(waveform, (0, segment_length - length))
