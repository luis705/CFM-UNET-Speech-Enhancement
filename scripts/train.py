"""Training entrypoint for CFM speech enhancement."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import matplotlib
import torch

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytorch_lightning as pL
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
)
from pytorch_lightning.loggers import MLFlowLogger

from src.cfm.lightning_module import CFMModule
from src.data.datamodule import VBDMDDataModule
from src.utils.callbacks import (
    EMACallback,
    LoggingCallback,
    MinEpochEarlyStopping,
    SlowMetricsCallback,
    SystemStatsLoggerCallback,
)
from src.utils.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CFM Speech Enhancement")
    parser.add_argument("--config", type=str, default="settings.yaml", help="Path to settings YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tuner", action="store_true", help="Use learning rate tuner")
    args = parser.parse_args()

    pL.seed_everything(args.seed)
    settings = get_settings(args.config)

    # Data
    datamodule = VBDMDDataModule(settings)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    with torch.device(device):
        model = CFMModule(settings)

    # Logger
    mlflow_logger = MLFlowLogger(
        experiment_name="cfm-unet",
        tracking_uri="http://192.168.0.100:10002",
        run_name="cfm-unet-run",
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=Path(settings.logging.log_dir) / "checkpoints",
            filename="epoch={epoch:03d}-loss={val/loss_cfm:.3f}",
            monitor="val/loss_cfm",
            mode="min",
            save_top_k=3,
            auto_insert_metric_name=False,
            save_last=True
        ),
        MinEpochEarlyStopping(
            monitor="val/loss_cfm",
            mode="min",
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
        SlowMetricsCallback(settings),
        RichModelSummary(max_depth=1),
        EMACallback(0.99),
        SystemStatsLoggerCallback(),
    ]

    # Trainer
    trainer = pL.Trainer(
        accelerator="gpu",
        devices=1,
        precision=settings.training.precision,
        max_epochs=settings.training.max_epochs,
        accumulate_grad_batches=settings.training.accum_batches,
        gradient_clip_val=settings.training.grad_clip,
        callbacks=callbacks,
        logger=mlflow_logger,
        log_every_n_steps=10,
        val_check_interval=settings.training.val_interval,
    )

    if args.tuner:
        tuner = pL.tuner.Tuner(trainer)
        lr_finder = tuner.lr_find(model, datamodule=datamodule)
        print(lr_finder.suggestion())
        sys.exit()

    # Training
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)

if __name__ == "__main__":
    matplotlib.use("agg")
    torch.set_float32_matmul_precision("high")
    main()
