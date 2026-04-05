"""Training entrypoint for CFM speech enhancement."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytorch_lightning as L
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
)
from pytorch_lightning.loggers import TensorBoardLogger

from src.cfm.lightning_module import CFMModule
from src.data.datamodule import VBDMDDataModule
from src.utils.callbacks import AudioLoggingCallback, ConfigLoggerCallback, EMACallback, SystemStatsLoggerCallback
from src.utils.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CFM Speech Enhancement")
    parser.add_argument("--config", type=str, default="settings.yaml", help="Path to settings YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    L.seed_everything(args.seed)
    settings = get_settings(args.config)

    # Data
    datamodule = VBDMDDataModule(settings)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    with torch.device(device):
        model = CFMModule(
            in_channels=settings.data.input_channels,
            out_channels=settings.data.output_channels,
            time_emb_dim=settings.model.time_emb_dim,
            projection_dim=settings.model.projection_dim,
            n_blocks=settings.model.n_blocks,
            compression_factor=settings.model.compression_factor,
            channels_factor=settings.model.channels_factor,
            num_heads=settings.model.num_heads,
            ff_mult=settings.model.ff_mult,
            n_groups=settings.model.n_groups,
            gradient_checkpointing=settings.model.gradient_checkpointing,
            ode_val_steps=settings.ode.val_steps,
            ode_test_steps=settings.ode.test_steps,
            audio_length_seconds=settings.audio.audio_length_seconds,
            audio_sample_rate=settings.audio.sample_rate,
            fft_size=settings.audio.fft_size,
            hop_length=settings.audio.hop_length,
            window_size=settings.audio.window_size,
            alpha=settings.audio.alpha,
            lr=settings.training.optim.lr,
            weight_decay=settings.training.optim.weight_decay,
            pct_start=settings.training.optim.pct_start,
            validation_slow_metrics_pct=settings.training.validation_slow_metrics_pct,
            histogram_logging_frequency=settings.training.histogram_logging_frequency,
            n_audios_to_log=settings.logging.n_audios,
        )

    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=settings.logging.log_dir,
        name="cfm-unet-se",
        # log_graph=True,
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=Path(settings.logging.log_dir) / "checkpoints",
            filename="epoch{epoch:03d}-pesq{val/pesq:.3f}",
            monitor="val/pesq",
            mode="max",
            save_top_k=3,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(
            monitor="val/pesq",
            mode="max",
            patience=settings.training.early_stop_patience,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        AudioLoggingCallback(n_samples=settings.logging.n_audios),
        RichModelSummary(max_depth=2),
        ConfigLoggerCallback(args.config),
        EMACallback(0.99),
        SystemStatsLoggerCallback(),
    ]

    # Trainer
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision=settings.training.precision,
        max_epochs=settings.training.max_epochs,
        accumulate_grad_batches=settings.training.accum_batches,
        gradient_clip_val=settings.training.grad_clip,
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=10,
        profiler="simple",
        val_check_interval=settings.training.val_interval,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)


if __name__ == "__main__":
    import torch
    import matplotlib
    matplotlib.use('agg')
    torch.set_float32_matmul_precision('high')
    main()
