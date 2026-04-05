"""Audio logging callback — logs waveforms, spectrograms, and fast metrics to TensorBoard."""

import pytorch_lightning as pl
import torch
import torchaudio

import gc

from src.utils.metrics import compute_estoi, compute_wb_pesq


class AudioLoggingCallback(pl.Callback):
    """Log audio samples and fast SI-SDR metrics at the end of each validation epoch.

    Reads ``pl_module.val_audio_buffer`` — a list of ``(clean, noisy, pred)`` CPU tensors
    that ``validation_step`` populates (capped at ``audio_log_samples``).
    """

    def __init__(self, n_samples: int = 4):
        super().__init__()
        self.n_samples = n_samples

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Garante que o buffer de áudio comece vazio em cada validação."""
        buf = getattr(pl_module, "val_audio_buffer", None)
        if buf is not None:
            buf.clear()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        buf = getattr(pl_module, "val_audio_buffer", [])
        if not buf:
            return

        tb = pl_module.logger.experiment if pl_module.logger else None
        sr = pl_module.audio_sample_rate
        step = pl_module.global_step

        # ── Log first sample's audio + spectrogram ───────────────────────
        samples_to_log = buf[: self.n_samples]

        for i, (clean, noisy, pred) in enumerate(samples_to_log):
            if tb is not None:
                # Usamos um sufixo /sample_0, /sample_1 para organizar no TB
                self._log_audio_triplet(tb, clean, noisy, pred, step, sr, suffix=f"sample_{i}")

        # ── Compute fast metrics on buffered samples ─────────────────────
        # Empilha os tensores e garante o formato (N, T) para as métricas
        pred_wavs = torch.stack([t[2].flatten() for t in samples_to_log])
        clean_wavs = torch.stack([t[0].flatten() for t in samples_to_log])

        pesq_value = compute_wb_pesq(clean_wavs, pred_wavs)
        pl_module.log("val/PESQ", pesq_value, sync_dist=True)

        estoi_value = compute_estoi(clean_wavs, pred_wavs)
        pl_module.log("val/STOI", estoi_value, sync_dist=True)

        # ── Clear buffer ─────────────────────────────────────────────────
        buf.clear()
        gc.collect()

    @staticmethod
    def _log_audio_triplet(
        tb, clean: torch.Tensor, noisy: torch.Tensor, pred: torch.Tensor, step: int, sr: int, suffix: str
    ) -> None:

        # Garante formato (1, T) para add_audio
        tb.add_audio(f"val/{suffix}/clean", clean.flatten().unsqueeze(0), step, sample_rate=sr)
        tb.add_audio(f"val/{suffix}/noisy", noisy.flatten().unsqueeze(0), step, sample_rate=sr)
        tb.add_audio(f"val/{suffix}/enhanced", pred.flatten().unsqueeze(0), step, sample_rate=sr)

        mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=512, hop_length=128, n_mels=80)
        amp_to_db = torchaudio.transforms.AmplitudeToDB()

        for tag, wav in [
            (f"val/{suffix}/spec_clean", clean),
            (f"val/{suffix}/spec_noisy", noisy),
            (f"val/{suffix}/spec_enhanced", pred),
        ]:
            # MelSpectrogram espera (..., Time). flatten + unsqueeze garante (1, T)
            spec = mel(wav.flatten().unsqueeze(0))
            spec_db = amp_to_db(spec)
            spec_img = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + 1e-8)
            tb.add_image(tag, spec_img, step)
