"""Evaluation metrics: full (test) and fast (validation during training)."""

import contextlib
import csv
from pathlib import Path

from joblib import Parallel, delayed
import numpy as np
import torch
from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility

from src.config import MetricsConfig

# ── Helpers ──────────────────────────────────────────────────────────────────


def _flatten_wav_list(wav_list: list[torch.Tensor]) -> list[torch.Tensor]:
    flat: list[torch.Tensor] = []
    for x in wav_list:
        if x.ndim == 1:
            flat.append(x.cpu())
        elif x.ndim == 2:
            flat.extend([s.cpu() for s in x])
        else:
            raise ValueError(f"Expected 1D or 2D tensor, got {x.ndim}D")
    return flat


def _compute_si_sdr(pred: torch.Tensor, clean: torch.Tensor) -> float:
    try:
        return float(scale_invariant_signal_distortion_ratio(pred, clean))
    except Exception:
        return 0.0


def _compute_si_sar(pred: torch.Tensor, clean: torch.Tensor) -> float:
    """Scale-Invariant Signal-to-Artifacts Ratio."""
    try:
        clean_zm = clean - clean.mean()
        pred_zm = pred - pred.mean()
        s_target = (torch.dot(pred_zm, clean_zm) / (torch.dot(clean_zm, clean_zm) + 1e-8)) * clean_zm
        e_artif = pred_zm - s_target
        return float(10 * torch.log10(torch.dot(s_target, s_target) / (torch.dot(e_artif, e_artif) + 1e-8)))
    except Exception:
        return 0.0


def _compute_lsd(pred: torch.Tensor, clean: torch.Tensor, n_fft: int = 512) -> float:
    """Log Spectral Distance (dB)."""
    try:
        window = torch.hann_window(n_fft, device=pred.device)
        pred_spec = torch.stft(pred, n_fft=n_fft, hop_length=n_fft // 4, window=window, return_complex=True)
        clean_spec = torch.stft(clean, n_fft=n_fft, hop_length=n_fft // 4, window=window, return_complex=True)
        pred_pow = pred_spec.abs().pow(2).clamp(min=1e-10)
        clean_pow = clean_spec.abs().pow(2).clamp(min=1e-10)
        # LSD = mean over time of sqrt(mean over freq of (log power diff)^2)
        log_diff_sq = (10 * torch.log10(pred_pow) - 10 * torch.log10(clean_pow)).pow(2)
        return float(log_diff_sq.mean(dim=0).sqrt().mean())
    except Exception:
        return 0.0


def _compute_log_mag(pred: torch.Tensor, clean: torch.Tensor, n_fft: int = 512) -> float:
    """Log-Magnitude Distance (dB)."""
    try:
        window = torch.hann_window(n_fft, device=pred.device)
        pred_spec = torch.stft(pred, n_fft=n_fft, hop_length=n_fft // 4, window=window, return_complex=True)
        clean_spec = torch.stft(clean, n_fft=n_fft, hop_length=n_fft // 4, window=window, return_complex=True)
        pred_mag = pred_spec.abs().clamp(min=1e-10)
        clean_mag = clean_spec.abs().clamp(min=1e-10)
        return float((20 * torch.log10(pred_mag) - 20 * torch.log10(clean_mag)).abs().mean())
    except Exception:
        return 0.0


def _compute_seg_snr(
    pred: torch.Tensor,
    clean: torch.Tensor,
    frame_len: int = 512,
    hop_len: int = 256,
) -> float:
    """Segmental SNR (dB) — SNR per frame, clipped to [-10, 35] dB, then averaged."""
    try:
        min_len = min(len(pred), len(clean))
        pred, clean = pred[:min_len], clean[:min_len]
        n_frames = (min_len - frame_len) // hop_len + 1
        if n_frames <= 0:
            return 0.0
        snrs: list[float] = []
        for i in range(n_frames):
            s = i * hop_len
            c = clean[s : s + frame_len]
            n = pred[s : s + frame_len] - c
            c_energy = torch.dot(c, c)
            n_energy = torch.dot(n, n)
            if c_energy > 1e-8 and n_energy > 1e-8:
                snrs.append(float(10 * torch.log10(c_energy / n_energy)).clip(-10.0, 35.0))
        return float(np.mean(snrs)) if snrs else 0.0
    except Exception:
        return 0.0


def _compute_all_fast(pred: torch.Tensor, clean: torch.Tensor) -> tuple[float, ...]:
    """Compute all fast validation metrics for a single pair."""
    return (
        _compute_si_sdr(pred, clean),
        _compute_si_sar(pred, clean),
        _compute_lsd(pred, clean),
        _compute_log_mag(pred, clean),
        _compute_seg_snr(pred, clean),
    )


# ── Fast metrics (validation during training) ────────────────────────────────


def compute_fast_metrics(
    pred_wavs: list[torch.Tensor],
    clean_wavs: list[torch.Tensor],
) -> dict[str, float]:
    """Compute fast validation metrics: SI-SDR, SI-SAR, LSD, Log-Mag, SegSNR.

    Operates on CPU tensors already in the audio buffer (no extra forward passes).
    """
    pred_flat = _flatten_wav_list(pred_wavs)
    clean_flat = _flatten_wav_list(clean_wavs)

    if len(pred_flat) != len(clean_flat) or not pred_flat:
        return {}

    results = Parallel(n_jobs=4, prefer="threads")(
        delayed(_compute_all_fast)(p, c) for p, c in zip(pred_flat, clean_flat, strict=True)
    )

    si_sdr_vals, si_sar_vals, lsd_vals, log_mag_vals, seg_snr_vals = zip(*results, strict=True)

    return {
        "SI-SDR": float(np.mean(si_sdr_vals)),
        "SI-SAR": float(np.mean(si_sar_vals)),
        "LSD": float(np.mean(lsd_vals)),
        "Log-Mag": float(np.mean(log_mag_vals)),
        "SegSNR": float(np.mean(seg_snr_vals)),
    }


# ── Full metrics (test script) ───────────────────────────────────────────────


def compute_slow_metrics(
    cfg: MetricsConfig,
    pred: torch.Tensor,
    clean: torch.Tensor,
    sample_rate: int,
) -> tuple[float, float, float]:
    """Compute per-sample CPU metrics (PESQ, ESTOI, SI-SDR)."""
    pesq_val, estoi_val = 0.0, 0.0

    if cfg.PESQ:
        with contextlib.suppress(Exception):
            pesq_val = float(perceptual_evaluation_speech_quality(pred, clean, sample_rate, "wb"))
    if cfg.ESTOI:
        with contextlib.suppress(Exception):
            estoi_val = float(short_time_objective_intelligibility(pred, clean, sample_rate, extended=True))

    return pesq_val, estoi_val


def compute_metrics(
    metrics_cfg: MetricsConfig,
    pred_wavs: list[torch.Tensor],
    clean_wavs: list[torch.Tensor],
    sample_rate: int = 16_000,
    device: torch.device | str = "cuda",
) -> dict[str, float]:
    """Compute all evaluation metrics. Test script only."""
    pred_flat = _flatten_wav_list(pred_wavs)
    clean_flat = _flatten_wav_list(clean_wavs)

    if len(pred_flat) != len(clean_flat):
        raise ValueError("pred_wavs and clean_wavs must have the same number of examples")

    if not pred_flat:
        return {}

    resultados = Parallel(n_jobs=8, prefer="threads")(
        delayed(compute_slow_metrics)(metrics_cfg, p, c, sample_rate)
        for p, c in zip(pred_flat, clean_flat, strict=True)
    )

    pesq_vals, estoi_vals = zip(*resultados, strict=True)

    results: dict[str, float] = {}

    if metrics_cfg.ESTOI:
        results["ESTOI"] = float(np.mean(estoi_vals))
    if metrics_cfg.PESQ:
        results["WB-PESQ"] = float(np.mean(pesq_vals))

    # ── Fast spectral / time-domain metrics (SI-SAR, LSD, Log-Mag, SegSNR) ──
    fast_results = Parallel(n_jobs=8, prefer="threads")(
        delayed(_compute_all_fast)(p, c) for p, c in zip(pred_flat, clean_flat, strict=True)
    )
    si_sdr_fast, si_sar_vals, lsd_vals, log_mag_vals, seg_snr_vals = zip(*fast_results, strict=True)

    if metrics_cfg.SI_SDR:
        results["SI-SDR"] = float(np.mean(si_sdr_fast))
    results["SI-SAR"] = float(np.mean(si_sar_vals))
    results["LSD"] = float(np.mean(lsd_vals))
    results["Log-Mag"] = float(np.mean(log_mag_vals))
    results["SegSNR"] = float(np.mean(seg_snr_vals))

    if metrics_cfg.DNSMOS:
        dnsmos_sig, dnsmos_bak, dnsmos_ovrl = [], [], []
        for pred in pred_flat:
            with contextlib.suppress(Exception):
                dns_scores = deep_noise_suppression_mean_opinion_score(
                    preds=pred,
                    personalized=False,
                    fs=sample_rate,
                    device=device,
                )
                dnsmos_sig.append(dns_scores[..., 1])
                dnsmos_bak.append(dns_scores[..., 2])
                dnsmos_ovrl.append(dns_scores[..., 3])
        if dnsmos_sig:
            results["DNSMOS-SIG"] = float(np.mean(dnsmos_sig))
            results["DNSMOS-BAK"] = float(np.mean(dnsmos_bak))
            results["DNSMOS-OVRL"] = float(np.mean(dnsmos_ovrl))

    return results


# ── CSV logger ───────────────────────────────────────────────────────────────


def append_metrics_csv(csv_path: str | Path, step: int, epoch: int, metrics: dict[str, float]) -> None:
    csv_path = Path(csv_path)
    file_exists = csv_path.exists()
    fieldnames = ["step", "epoch", *sorted(metrics.keys())]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = {"step": step, "epoch": epoch, **metrics}
        writer.writerow(row)
