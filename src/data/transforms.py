from einops import rearrange
import torch


def complex_to_channels(spec: torch.Tensor) -> torch.Tensor:
    """Converts complex STFT to 2-channel real.


    Args:
        spec (torch.Tensor): complex STFT of shape (2, F, T)

    Returns:
        torch.Tensor: 2-channel real STFT of shape (2, F, T)
    """
    return torch.cat((spec.real, spec.imag), dim=0)


def channels_to_complex(spec: torch.Tensor) -> torch.Tensor:
    """
    Converte STFT de 2 canais reais (saída da rede) de volta para complexo.
    Aceita formato com batch (B, 2, F, T) ou sem batch (2, F, T).
    """
    # Formato (B, 2, F, T)
    return torch.complex(spec[:, 0, ...], spec[:, 1, ...])


def pad_or_crop(length: int, *signals: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Pad or crop a given signal to the given number of samples

    Args:
        signals: Tensors to be padded or cropped of shape (..., T)
        length (_type_): Target length of the signal

    Returns:
        tuple[torch.Tensor, ...]: Padded or cropped signals of shape (..., length)
    """
    # Calcula o total necessário. Se negativo, o F.pad fará o crop automaticamente.
    # Usamos s.shape[-1] para ser compatível com (C, T) ou (B, C, T)
    return tuple(torch.nn.functional.pad(s, (0, length - s.shape[-1])) for s in signals)


def z_score_normalize(noisy: torch.Tensor, clean: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize both noisy and clean audios using Z-score (mean and std) of the noisy signal.

    Args:
        noisy (torch.Tensor): Noisy signal of shape (1, T)
        clean (torch.Tensor): Clean signal of shape (1, T)
        eps (float): Small value to avoid division by zero.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Normalized noisy, clean, mean, and std.
    """
    mean = noisy.mean()
    std = noisy.std()
    return (noisy - mean) / (std + eps), (clean - mean) / (std + eps), mean, std


def power_law_compress(spec: torch.Tensor, alpha: float) -> torch.Tensor:
    """Apply power law compression to the given spectrogram.

    Args:
        spec (torch.Tensor): Input spectrogram of shape (2, F, T)
        alpha (float): Compression exponent

    Returns:
        torch.Tensor: Compressed spectrogram of shape (2, F, T)
    """
    return torch.pow(spec, alpha)


def power_law_decompress(compressed_spec: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Reverte a compressão power-law mantendo a fase intacta.
    Z_uncompressed = |Z_compressed|^(1/alpha) * e^(j * phase)
    """
    mag = torch.abs(compressed_spec)
    phase = torch.angle(compressed_spec)

    # Descomprime a magnitude
    mag_decompressed = mag ** (1.0 / alpha)

    # Reconstrói o tensor complexo
    return torch.polar(mag_decompressed, phase)


def signal_to_stft(
    signal: torch.Tensor,
    fft_size: int,
    hop_length: int,
    window_size: int,
    alpha: float,
) -> torch.Tensor:
    # STFT → complex (F, T)
    window = torch.hann_window(window_size).to(signal.device)
    spec = torch.stft(
        signal, n_fft=fft_size, hop_length=hop_length, win_length=window_size, return_complex=True, window=window
    )

    # Power-law compression
    spec = power_law_compress(spec, alpha)

    # Complex → 2-channel real
    spec = complex_to_channels(spec)  # (2, F, T)

    return spec


def preprocess_item(
    noisy_audio: torch.Tensor,
    clean_audio: torch.Tensor,
    pad_crop: bool,
    length: int,
    fft_size: int,
    hop_length: int,
    window_size: int,
    alpha: float,
) -> torch.Tensor:
    # Pad / crop to fixed length
    if pad_crop:
        noisy_audio, clean_audio = pad_or_crop(length, noisy_audio, clean_audio)

    # Z-score normalize (based on noisy to preserve SNR)
    noisy, clean, mean, std = z_score_normalize(noisy_audio, clean_audio)

    noisy_stft = signal_to_stft(noisy, fft_size, hop_length, window_size, alpha)
    clean_stft = signal_to_stft(clean, fft_size, hop_length, window_size, alpha)

    return noisy_audio, clean_audio, noisy_stft, clean_stft, mean, std


def stft_to_waveform(
    spec: torch.Tensor,
    length: int,
    fft_size: int,
    hop_length: int,
    window_size: int,
    alpha: float,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """
    Converte a STFT comprimida de volta para o áudio no domínio do tempo.

    Args:
        spec: Tensor da STFT. Pode ser 2 canais reais (saída da U-Net) ou já complexo.
        length: Tamanho original do áudio em samples (importante para iSTFT cortar as bordas).
        fft_size: n_fft usado.
        hop_length: hop_length usado.
        window_size: win_length usado.
        alpha: fator da compressão power-law usado no pré-processamento.
        mean: A média usada na normalização para restaurar o offset original.
        std: O desvio padrão usado na normalização para restaurar a escala original.
    """
    spec = channels_to_complex(spec)

    # 2. Desfaz a compressão
    spec = power_law_decompress(spec, alpha)

    # 3. iSTFT
    window = torch.hann_window(window_size).to(spec.device)
    waveform = torch.istft(
        spec,
        n_fft=fft_size,
        hop_length=hop_length,
        win_length=window_size,
        length=length,  # O 'length' garante que o vetor saia exatamente com o tamanho original
        return_complex=False,
        window=window,
    )

    # 4. Desfaz o z-score normalize: x = (z * std) + mean
    waveform = waveform * rearrange(std, "B -> B 1") + rearrange(mean, "B -> B 1")

    return waveform
