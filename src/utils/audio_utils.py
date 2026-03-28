"""STFT/iSTFT wrappers and OLA inference utilities for the SE pipeline."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from src.data.transforms import istft as _istft_core, stft as _stft_core

if TYPE_CHECKING:
    from src.models.se_model import SEModel


def stft_transform(
    wav: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window_type: str,
) -> torch.Tensor:
    """Compute STFT → (B, 2, F, T)."""
    return _stft_core(wav, n_fft, hop_length, win_length, window_type)


def istft_transform(
    spec: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window_type: str,
    length: int | None = None,
) -> torch.Tensor:
    """Inverse STFT from (B, 2, F, T) → (B, T_samples)."""
    return _istft_core(spec, n_fft, hop_length, win_length, window_type, length=length)


# ──────────────────────────────────────────────────────────────────────────────
# Weighted Overlap-Add (WOLA) inference for variable-length audio
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def enhance_audio(
    noisy_wav: torch.Tensor,
    model: SEModel,
    chunk_size: int,
    overlap: float,
    solver: str,
    num_steps: int,
    max_batch_chunks: int,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> torch.Tensor:
    """Enhance variable-length audio via time-domain overlap-add.

    Uses a Hann analysis/synthesis window with 50% overlap (COLA-compliant).

    Args:
        noisy_wav: 1-D tensor (T,) on model device.
        model: SEModel instance (must expose .enhance() for short segments).
        chunk_size: Number of samples per chunk.
        overlap: Overlap ratio in [0, 1). Typically 0.5 for Hann COLA.
        solver: ODE solver name passed to ode_solve.
        num_steps: Number of ODE integration steps.
        max_batch_chunks: Maximum number of chunks to process in one GPU batch.

    Returns:
        1-D tensor (T,) enhanced waveform on CPU.
    """
    from src.models.aggregates.cfm import ode_solve

    device = noisy_wav.device
    orig_len = noisy_wav.shape[0]

    hop = int(chunk_size * (1.0 - overlap))
    if hop <= 0:
        raise ValueError(f"Invalid overlap={overlap} with chunk_size={chunk_size}")

    # Pad waveform so all chunks are full-length
    n_chunks = max(1, math.ceil((orig_len - chunk_size) / hop) + 1)
    padded_len = (n_chunks - 1) * hop + chunk_size
    if padded_len > orig_len:
        noisy_wav = F.pad(noisy_wav, (0, padded_len - orig_len))

    # Hann synthesis window (COLA-compliant at 50% overlap)
    window = torch.hann_window(chunk_size, device=device, dtype=noisy_wav.dtype)

    # Extract overlapping chunks: (n_chunks, chunk_size)
    chunks = noisy_wav.unfold(0, chunk_size, hop)  # (n_chunks, chunk_size)

    # Output accumulator and window envelope
    output = torch.zeros(padded_len, device=device, dtype=noisy_wav.dtype)
    envelope = torch.zeros(padded_len, device=device, dtype=noisy_wav.dtype)

    # Process in sub-batches to limit GPU memory
    for start in range(0, n_chunks, max_batch_chunks):
        end = min(start + max_batch_chunks, n_chunks)
        batch_chunks = chunks[start:end]  # (B_sub, chunk_size)

        # STFT → run CFM ODE → iSTFT
        noisy_spec = model._stft(batch_chunks)  # (B_sub, 2, F, T_stft)
        vfn = model._velocity_fn(noisy_spec)
        pred_spec = ode_solve(
            vfn,
            noisy_spec,
            method=solver,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )
        pred_chunks = model._istft(pred_spec, length=chunk_size)  # (B_sub, chunk_size)

        # Apply Hann window and accumulate via OLA
        for j in range(end - start):
            idx = start + j
            offset = idx * hop
            output[offset : offset + chunk_size] += pred_chunks[j] * window
            envelope[offset : offset + chunk_size] += window * window

    # Normalize by window envelope (avoid division by zero in zero-padded tail)
    envelope = envelope.clamp(min=1e-8)
    output = output / envelope

    return output[:orig_len].cpu()
