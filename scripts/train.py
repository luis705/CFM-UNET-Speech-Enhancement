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
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--mel", action="store_true", help="Use mel-spectrograms")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    pL.seed_everything(args.seed)
    settings = get_settings(args.config)

    # Data
    datamodule = VBDMDDataModule(settings)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    with torch.device(device):
        model = CFMModule(settings, input_mel=args.mel)

    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=settings.logging.log_dir,
        name="cfm-unet-se" if not args.mel else "cfm-unet-se-mel",
        # log_graph=True,
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=Path(settings.logging.log_dir) / "checkpoints",
            filename="epoch={epoch:03d}-sisdr={val_sample/SI_SDR:.3f}",
            monitor="val_sample/SI_SDR",
            mode="max",
            save_top_k=3,
            auto_insert_metric_name=False,
        ),
        MinEpochEarlyStopping(
            monitor="val_sample/SI_SDR",
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
    trainer = pL.Trainer(
        accelerator="gpu",
        devices=1,
        precision=settings.training.precision,
        max_epochs=settings.training.max_epochs,
        accumulate_grad_batches=settings.training.accum_batches,
        gradient_clip_val=settings.training.grad_clip,
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=10,
        val_check_interval=settings.training.val_interval,
    )

    # Training
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)

    # Test - Carrega automaticamente o melhor checkpoint monitorado pelo ModelCheckpoint
    # Isso garante que as métricas de teste sejam extraídas do melhor modelo, não do último.
    print("\n[Train Script] Iniciando avaliação no conjunto de teste com o melhor checkpoint...")
    results = trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

    print(results)


if __name__ == "__main__":
    matplotlib.use("agg")
    torch.set_float32_matmul_precision("high")
    main()
