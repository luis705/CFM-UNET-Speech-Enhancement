"""Trainer setup: callbacks, loggers."""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from src.callbacks import (
    AudioLoggingCallback,
    CheckpointLoggerCallback,
    ConfigLoggerCallback,
    EMACallback,
    SystemStatsLoggerCallback
)
from src.config import Settings

# ── Build Trainer ────────────────────────────────────────────────────────────


def build_trainer(cfg: Settings, settings_path: str, version: str | None = None) -> pl.Trainer:
    tc = cfg.training
    cc = cfg.checkpoint
    lc = cfg.logging

    logger = TensorBoardLogger(save_dir=lc.log_dir, name=lc.name, version=version)

    callbacks = [
        ModelCheckpoint(
            dirpath=cc.dirpath,
            monitor=cc.monitor,
            mode=cc.mode,
            save_top_k=cc.save_top_k,
            save_last=cc.save_last,
            filename="cfm-se-{epoch:03d}-{val_loss:.4f}",
        ),
        EarlyStopping(
            monitor=tc.early_stopping_metric,
            patience=tc.early_stopping_patience,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="step"),
        EMACallback(ema_decay=tc.ema_decay),
        AudioLoggingCallback(),
        SystemStatsLoggerCallback(),
        CheckpointLoggerCallback(),
        ConfigLoggerCallback(settings_path),
    ]

    trainer = pl.Trainer(
        max_epochs=tc.max_epochs,
        precision=tc.precision,
        gradient_clip_val=tc.gradient_clip_val,
        accumulate_grad_batches=tc.accumulate_grad_batches,
        logger=logger,
        callbacks=callbacks,
        val_check_interval=cfg.validation.every_n_steps,
        log_every_n_steps=50,
        deterministic=False,
    )
    return trainer
