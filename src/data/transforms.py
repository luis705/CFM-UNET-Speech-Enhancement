import torch
import torchaudio

def complex_to_channels(spec: torch.Tensor) -> torch.Tensor:
    """Converts complex STFT to 2-channel real.


    Args:
        spec (torch.Tensor): complex STFT of shape (2, F, T)

    Returns:
        torch.Tensor: 2-channel real STFT of shape (2, F, T)
    """
    if spec.ndim == 3:  # (B, F, T)
        return torch.stack([spec.real, spec.imag], dim=1)  # (B, 2, F, T)
    # Fallback para (F, T)
    return torch.stack([spec.real, spec.imag], dim=0)


def channels_to_complex(spec: torch.Tensor) -> torch.Tensor:
    """
    Converte STFT de 2 canais reais (saída da rede) de volta para complexo.
    Aceita formato com batch (B, 2, F, T) ou sem batch (2, F, T).
    """
    # Formato (B, 2, F, T)
    if spec.ndim == 4:
        return torch.complex(spec[:, 0, ...], spec[:, 1, ...])
    # Formato (2, F, T)
    return torch.complex(spec[0, ...], spec[1, ...])


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


def z_score_normalize(
    noisy: torch.Tensor, clean: torch.Tensor, eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize both noisy and clean audios using Z-score (mean and std) of the noisy signal.

    Args:
        noisy (torch.Tensor): Noisy signal of shape (1, T)
        clean (torch.Tensor): Clean signal of shape (1, T)
        eps (float): Small value to avoid division by zero.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Normalized noisy, clean, mean, and std.
    """
    # Calcula média e std por amostra no batch (dimensões de tempo)
    dims = tuple(range(1, noisy.ndim))
    mean = noisy.mean(dim=dims, keepdim=True)
    std = noisy.std(dim=dims, keepdim=True)

    noisy_norm = (noisy - mean) / (std + eps)
    clean_norm = (clean - mean) / (std + eps)
    return noisy_norm, clean_norm, mean.squeeze(), std.squeeze()


def power_law_compress(spec: torch.Tensor, alpha: float) -> torch.Tensor:
    """Apply power law compression to the given spectrogram.

    Args:
        spec (torch.Tensor): Input spectrogram of shape (2, F, T)
        alpha (float): Compression exponent

    Returns:
    torch.Tensor: Compressed complex spectrogram.
    """
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    return torch.polar(torch.pow(mag, alpha), phase)


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
    # torch.stft expects 1D or 2D. We squeeze the channel dimension if present.
    if signal.ndim == 3:
        signal = signal.squeeze(1)

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
    fft_size: int = 512,
    hop_length: int = 128,
    window_size: int = 256,
    alpha: float = 0.5,
    sample_rate: int = 16000,
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

    # 2. Desfaz a compressão (Polar não aceita batch nativamente em versões antigas, mas torch.complex sim)
    mag = torch.abs(spec) ** (1.0 / alpha)
    phase = torch.angle(spec)
    spec = torch.polar(mag, phase)

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
    waveform = waveform * std.view(-1, 1) + mean.view(-1, 1) if std.ndim > 0 else waveform * std + mean

    # Restore Channel dimension: (B, T) -> (B, 1, T) for OLA and consistency
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(1)

    return waveform


def add_stochastic_noise(signal: torch.Tensor, min_snr_db: float = 10.0, max_snr_db: float = 25.0) -> torch.Tensor:
    """Adiciona ruído estocástico (branco, rosa ou castanho) ao sinal com um SNR aleatório."""
    
    # Sorteia a cor do ruído: 0 (branco), 1 (rosa), 2 (castanho)
    # Ruído branco: gaussiana normal
    # Ruído rosa: 1/f -> flicker noise: transistores e semicondutores
    # Ruído castanho: f -> Vibrações mecânicas: vento batendo, microfone balançando, rumble de sala

    beta = torch.randint(0, 3, (1,)).item()
    
    N = signal.shape[-1]
    noise = torch.randn_like(signal)
    
    # Se beta > 0, moldamos o espetro para criar ruído rosa (1) ou castanho (2)
    if beta > 0:
        # Passa para o domínio da frequência
        X = torch.fft.rfft(noise)
        freqs = torch.fft.rfftfreq(N, device=signal.device)
        
        # Evita a divisão por zero na componente DC (frequência 0)
        freqs[0] = 1.0 
        
        # Calcula a amplitude: decai com a frequência com base no beta
        amplitude = 1.0 / (freqs ** (beta / 2.0))
        amplitude[0] = 0.0  # Remove o offset DC
        
        # Aplica o filtro e regressa ao domínio do tempo
        Y = X * amplitude
        noise = torch.fft.irfft(Y, n=N)
        
        # Normaliza o ruído gerado para garantir estabilidade no cálculo da potência
        noise = noise / (torch.std(noise) + 1e-8)
    
    # Calcula a potência do sinal e do ruído
    signal_power = torch.mean(signal ** 2)
    noise_power = torch.mean(noise ** 2)
    
    # Evita divisão por zero se o sinal for silêncio absoluto
    if signal_power == 0:
        return signal
        
    # Sorteia um SNR alvo para esta amostra
    target_snr = torch.empty(1).uniform_(min_snr_db, max_snr_db).item()
    
    # Calcula o fator de escala para atingir o SNR desejado
    scale = torch.sqrt(signal_power / noise_power) * (10.0 ** (-target_snr / 20.0))
    
    return signal + (noise * scale)