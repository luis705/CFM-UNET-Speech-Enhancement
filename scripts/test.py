"""Evaluate a trained checkpoint on the VoiceBank-DEMAND test set."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytorch_lightning as L
from pytorch_lightning.loggers import TensorBoardLogger

from src.utils.settings import get_settings
from src.cfm.lightning_module import CFMModule
from src.data.datamodule import VBDMDDataModule


def main() -> None:
    parser = argparse.ArgumentParser(description="Test CFM Speech Enhancement")
    parser.add_argument("--config", type=str, default="settings.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--output", type=str, default="test_results.csv", help="CSV output path")
    args = parser.parse_args()

    settings = get_settings(args.config)

    datamodule = VBDMDDataModule(settings)
    model = CFMModule.load_from_checkpoint(args.checkpoint, settings=settings)

    tb_logger = TensorBoardLogger(
        save_dir=settings.logging.log_dir,
        name="cfm-speech-enhancement-test",
    )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision=settings.training.precision,
        logger=tb_logger,
    )

    results = trainer.test(model, datamodule=datamodule)

    # Write results to CSV
    if results:
        metrics = results[0]
        output_path = Path(args.output)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in metrics.items():
                writer.writerow([k, v])
        print(f"Results saved to {output_path}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
