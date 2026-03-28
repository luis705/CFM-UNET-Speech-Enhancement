"""Audio logging callback — logs waveforms, spectrograms, and fast metrics to TensorBoard."""

import pytorch_lightning as pl
import torch
import torchaudio

from src.utils.metrics import compute_fast_metrics


class AudioLoggingCallback(pl.Callback):
    """Log audio samples and fast SI-SDR metrics at the end of each validation epoch.

    Reads ``pl_module.val_audio_buffer`` — a list of ``(clean, noisy, pred)`` CPU tensors
    that ``validation_step`` populates (capped at ``audio_log_samples``).
    """

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
        sr = pl_module.cfg.audio.sample_rate
        step = pl_module.global_step

        # ── Log first sample's audio + spectrogram ───────────────────────
        for i, (clean, noisy, pred) in enumerate(buf):
            if tb is not None:
                # Usamos um sufixo /sample_0, /sample_1 para organizar no TB
                self._log_audio_triplet(tb, clean, noisy, pred, step, sr, suffix=f"sample_{i}")

        # ── Compute fast metrics on buffered samples ─────────────────────
        pred_wavs = [t[2] for t in buf]
        clean_wavs = [t[0] for t in buf]
        fast = compute_fast_metrics(pred_wavs, clean_wavs)
        for k, v in fast.items():
            pl_module.log(f"val/{k}", v, sync_dist=True)

        # ── Clear buffer ─────────────────────────────────────────────────
        buf.clear()

    @staticmethod
    def _log_audio_triplet(
        tb,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        pred: torch.Tensor,
        step: int,
        sr: int,
        suffix: str
    ) -> None:
        def normalize(w: torch.Tensor) -> torch.Tensor:
            return w / (torch.max(torch.abs(w)) + 1e-8)

        tb.add_audio(f"val/{suffix}/clean", normalize(clean).unsqueeze(0), step, sample_rate=sr)
        tb.add_audio(f"val/{suffix}/noisy", normalize(noisy).unsqueeze(0), step, sample_rate=sr)
        tb.add_audio(f"val/{suffix}/enhanced", normalize(pred).unsqueeze(0), step, sample_rate=sr)

        mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=512, hop_length=128, n_mels=80)
        amp_to_db = torchaudio.transforms.AmplitudeToDB()

        for tag, wav in [(f"val/{suffix}/spec_clean", clean), (f"val/{suffix}/spec_noisy", noisy), (f"val/{suffix}/spec_enhanced", pred)]:
            spec = mel(wav.unsqueeze(0))
            spec_db = amp_to_db(spec)
            spec_img = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + 1e-8)
            tb.add_image(tag, spec_img.squeeze(0).unsqueeze(0), step)
