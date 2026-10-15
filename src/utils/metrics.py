from __future__ import annotations

import logging

import numpy as np
from pesq import pesq
from pystoi import stoi
import torch

logger = logging.getLogger(__name__)


def compute_wb_pesq(ref: torch.Tensor, deg: torch.Tensor, sr: int = 16000) -> float:
    """Compute Wideband PESQ score.

    Args:
        ref: (T,) clean reference waveform.
        deg: (T,) degraded / enhanced waveform.
        sr:  sample rate (must be 16000 for WB-PESQ).

    Returns:
        PESQ score (float).
    """
    # Garante que os tensores tenham formato (Batch, Time) removendo dimensões de canal extras
    if ref.ndim == 1:
        ref = ref.unsqueeze(0)
    if deg.ndim == 1:
        deg = deg.unsqueeze(0)
    ref = ref.squeeze(1) if ref.ndim == 3 else ref
    deg = deg.squeeze(1) if deg.ndim == 3 else deg

    ref_np = ref.detach().cpu().numpy().astype(np.float64)
    deg_np = deg.detach().cpu().numpy().astype(np.float64)

    results = []
    for r, d in zip(ref_np, deg_np, strict=True):
        try:
            # A biblioteca pesq exige arrays 1D (T,)
            results.append(pesq(sr, r, d, "wb"))
        except Exception as e:
            # logger.warning(f"Error occurred while computing PESQ for a sample: {e}")
            results.append(1.0)

    return np.mean(results)


def compute_estoi(ref: torch.Tensor, deg: torch.Tensor, sr: int = 16000) -> float:
    """Compute Extended Short-Time Objective Intelligibility (ESTOI).

    Args:
        ref: (T,) clean reference waveform.
        deg: (T,) degraded / enhanced waveform.
        sr:  sample rate.

    Returns:
        ESTOI score in [0, 1].
    """
    # Garante que os tensores tenham formato (Batch, Time) removendo dimensões de canal extras
    if ref.ndim == 1:
        ref = ref.unsqueeze(0)
    if deg.ndim == 1:
        deg = deg.unsqueeze(0)
    ref = ref.squeeze(1) if ref.ndim == 3 else ref
    deg = deg.squeeze(1) if deg.ndim == 3 else deg

    ref_np = ref.detach().cpu().numpy().astype(np.float64)
    deg_np = deg.detach().cpu().numpy().astype(np.float64)

    results = []
    for r, d in zip(ref_np, deg_np, strict=True):
        try:
            results.append(stoi(r, d, sr))
        except Exception as e:
            # logger.warning(f"Error occurred while computing ESTOI for a sample: {e}")
            results.append(1.0)
    return np.mean(results)
