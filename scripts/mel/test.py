"""Training entrypoint for CFM speech enhancement."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytorch_lightning as L
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
)
from pytorch_lightning.loggers import TensorBoardLogger

from src.cfm.lightning_module import CFMModule
from src.data.datamodule import VBDMDDataModule
from src.utils.callbacks import (
    EMACallback,
    LoggingCallback,
    MinEpochEarlyStopping,
    SystemStatsLoggerCallback,
)
from src.utils.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CFM Speech Enhancement")
    parser.add_argument("--config", type=str, default="settings.yaml", help="Path to settings YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.checkpoint is None:
        raise ValueError("Checkpoint path must be provided")



    L.seed_everything(args.seed)
    settings = get_settings(args.config)

    # Data
    datamodule = VBDMDDataModule(settings)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    with torch.device(device):
        model = CFMModule.load_from_checkpoint(args.checkpoint, cfg=settings, input_mel=True)


    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=Path(settings.logging.log_dir) / "checkpoints",
            filename="epoch={epoch:03d}-sisdr={val/SAMPLE_SI_SDR:.3f}",
            monitor="val/SAMPLE_SI_SDR",
            mode="max",
            save_top_k=3,
            auto_insert_metric_name=False,
        ),
        MinEpochEarlyStopping(
            monitor="val/SAMPLE_SI_SDR",
            mode="max",
            patience=settings.training.early_stop_patience,
            min_epochs=settings.training.early_stop_min_epochs,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        LoggingCallback(
            n_samples=settings.logging.n_audios,
            config_path=args.config,
            histogram_logging_frequency=settings.logging.histogram_logging_frequency,
        ),
        RichModelSummary(max_depth=2),
        EMACallback(0.99),
        SystemStatsLoggerCallback(),
    ]

    # Trainer
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision=settings.training.precision,
        callbacks=callbacks,
        logger=False,  # Desativa o TensorBoard para salvar APENAS localmente via Callback
    )

    results = trainer.test(model, datamodule=datamodule)

    print(results)


if __name__ == "__main__":
    import matplotlib
    import torch

    matplotlib.use("agg")
    torch.set_float32_matmul_precision("high")
    main()
