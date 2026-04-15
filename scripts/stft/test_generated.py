#!/usr/bin/env python3
"""Test entrypoint for evaluating generated noisy WAV files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


import pytorch_lightning as L
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from src.cfm.lightning_module import CFMModule
from src.utils.callbacks import LoggingCallback, SystemStatsLoggerCallback
from src.utils.settings import Settings, get_settings


def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim != 2:
        raise ValueError(f"Waveform com shape invalido: {tuple(waveform.shape)}")
    if waveform.shape[0] == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def _resample_if_needed(waveform: torch.Tensor, source_sr: int, target_sr: int) -> torch.Tensor:
    if source_sr == target_sr:
        return waveform
    resampler = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=target_sr)
    return resampler(waveform)


def _match_length(clean_waveform: torch.Tensor, target_len: int) -> torch.Tensor:
    clean_len = clean_waveform.shape[-1]
    if clean_len == target_len:
        return clean_waveform
    if clean_len > target_len:
        return clean_waveform[:, :target_len]

    repeats = (target_len + clean_len - 1) // clean_len
    tiled = clean_waveform.repeat(1, repeats)
    return tiled[:, :target_len]


class GeneratedNoisyDataset(Dataset):
    """Pairs each generated noisy WAV with the provided clean reference WAV."""

    def __init__(self, clean_audio: Path, noisy_dir: Path, target_sr: int) -> None:
        super().__init__()

        if not clean_audio.exists() or not clean_audio.is_file():
            raise FileNotFoundError(f"Arquivo de audio limpo nao encontrado: {clean_audio}")
        if not noisy_dir.exists() or not noisy_dir.is_dir():
            raise FileNotFoundError(f"Diretorio de audios ruidosos nao encontrado: {noisy_dir}")

        noisy_files = sorted(p for p in noisy_dir.glob("*.wav") if p.is_file())
        if not noisy_files:
            raise ValueError(f"Nenhum arquivo .wav encontrado em: {noisy_dir}")

        clean_wave, clean_sr = torchaudio.load(clean_audio)
        clean_wave = _to_mono(clean_wave)
        clean_wave = _resample_if_needed(clean_wave, source_sr=clean_sr, target_sr=target_sr)

        self.clean_waveform = clean_wave.contiguous()
        self.noisy_files = noisy_files
        self.target_sr = target_sr

    def __len__(self) -> int:
        return len(self.noisy_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        noisy_path = self.noisy_files[idx]
        noisy_wave, noisy_sr = torchaudio.load(noisy_path)
        noisy_wave = _to_mono(noisy_wave)
        noisy_wave = _resample_if_needed(noisy_wave, source_sr=noisy_sr, target_sr=self.target_sr)

        clean_wave = _match_length(self.clean_waveform, noisy_wave.shape[-1])
        return noisy_wave.contiguous(), clean_wave.contiguous()


class GeneratedNoisyDataModule(L.LightningDataModule):
    def __init__(self, settings: Settings, clean_audio: Path, noisy_dir: Path) -> None:
        super().__init__()
        self.settings = settings
        self.clean_audio = clean_audio
        self.noisy_dir = noisy_dir
        self._test_ds: GeneratedNoisyDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in ("test", None):
            self._test_ds = GeneratedNoisyDataset(
                clean_audio=self.clean_audio,
                noisy_dir=self.noisy_dir,
                target_sr=self.settings.audio.sample_rate,
            )

    def test_dataloader(self) -> DataLoader:
        if self._test_ds is None:
            raise RuntimeError("DataModule nao inicializado. Chame setup('test') antes.")

        num_workers = int(self.settings.data.num_workers)
        return DataLoader(
            self._test_ds,
            batch_size=1,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )


def _resolve_precision(precision: str | int, accelerator: str) -> str | int:
    if accelerator == "cpu" and isinstance(precision, str):
        if "16" in precision or "bf16" in precision:
            return "32-true"
    return precision


def main() -> None:
    parser = argparse.ArgumentParser(description="Test CFM on generated noisy WAV files")
    parser.add_argument("clean_audio", type=str, help="Path to original clean WAV")
    parser.add_argument("noisy_dir", type=str, help="Directory containing generated noisy WAV files")
    parser.add_argument("--config", type=str, default="settings.yaml", help="Path to settings YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.checkpoint is None:
        raise ValueError("Checkpoint path must be provided")

    clean_audio = Path(args.clean_audio)
    noisy_dir = Path(args.noisy_dir)

    L.seed_everything(args.seed)
    settings = get_settings(args.config)

    # Data
    datamodule = GeneratedNoisyDataModule(
        settings=settings,
        clean_audio=clean_audio,
        noisy_dir=noisy_dir,
    )

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    with torch.device(device):
        model = CFMModule.load_from_checkpoint(args.checkpoint, cfg=settings)

    callbacks = [
        LoggingCallback(
            n_samples=settings.logging.n_audios,
            config_path=args.config,
            histogram_logging_frequency=settings.logging.histogram_logging_frequency,
        ),
        SystemStatsLoggerCallback(),
    ]

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    precision = _resolve_precision(settings.training.precision, accelerator)

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=1,
        precision=precision,
        callbacks=callbacks,
        logger=False,
    )

    results = trainer.test(model, datamodule=datamodule)
    print(results)


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("agg")
    torch.set_float32_matmul_precision("high")
    main()
