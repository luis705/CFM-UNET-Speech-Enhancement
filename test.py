"""Test script — run full evaluation on the test set."""

import argparse
from pathlib import Path
import random

import torch
import torchaudio
from tqdm import tqdm

from src.config import Settings
from src.data.dataset import build_dataloaders
from src.models.se_model import SEModel
from src.utils.metrics import append_metrics_csv, compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Test CFM-SE model (full metrics)")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output-csv", type=str, default="test_metrics.csv")
    parser.add_argument("--num-samples", type=int, default=0, help="0 = all samples")
    parser.add_argument("--solver", type=str, default=None, help="ODE solver (default: from config ode.test.solver)")
    parser.add_argument(
        "--num-steps", type=int, default=None, help="ODE steps (default: from config ode.test.num_steps)"
    )
    parser.add_argument("--atol", type=float, default=None, help="Absolute tolerance (dopri5)")
    parser.add_argument("--rtol", type=float, default=None, help="Relative tolerance (dopri5)")
    args = parser.parse_args()

    cfg = Settings.from_yaml(args.config)

    # Override ODE settings from CLI if provided
    if args.solver is not None:
        cfg.ode.test.solver = args.solver
    if args.num_steps is not None:
        cfg.ode.test.num_steps = args.num_steps
    if args.atol is not None:
        cfg.ode.test.atol = args.atol
    if args.rtol is not None:
        cfg.ode.test.rtol = args.rtol

    solver = cfg.ode.test.solver
    num_steps = cfg.ode.test.num_steps
    atol = cfg.ode.test.atol
    rtol = cfg.ode.test.rtol

    model = SEModel.load_from_checkpoint(args.checkpoint, cfg=cfg)
    model.eval()
    model.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Override data config for test: full-length clips, no splitting
    cfg.data.segment_length = 0
    cfg.data.batch_size = 1
    cfg.data.val_split_from_test = None  # Use the full test set for evaluation
    _, _, test_loader = build_dataloaders(cfg.data, cfg.audio)

    num_samples = min(args.num_samples, len(test_loader)) if args.num_samples > 0 else len(test_loader)

    pred_wavs: list[torch.Tensor] = []
    clean_wavs: list[torch.Tensor] = []

    save_dir = Path(cfg.test.save_dir)
    if cfg.test.save_audio_ratio > 0:
        save_dir.mkdir(parents=True, exist_ok=True)
        random.seed(42)

    print(f"Testing on {num_samples} samples with {solver} solver ({num_steps} steps)...")

    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        if i >= num_samples:
            break
        noisy_wav = batch["noisy"].to(device).squeeze(0)  # (T,)
        clean = batch["clean"].squeeze(0)

        with torch.no_grad():
            pred = model.enhance_long(
                noisy_wav,
                solver=solver,
                num_steps=num_steps,
                atol=atol,
                rtol=rtol,
            ).cpu()

        pred_wavs.append(pred)
        clean_wavs.append(clean)

        if cfg.test.save_audio_ratio > 0 and random.random() < cfg.test.save_audio_ratio:
            filename = batch["filename"][0]
            torchaudio.save(save_dir / f"{filename}_enhanced.wav", pred.unsqueeze(0), cfg.audio.sample_rate)
            torchaudio.save(save_dir / f"{filename}_noisy.wav", noisy_wav.cpu().unsqueeze(0), cfg.audio.sample_rate)
            if "clean" in batch:
                torchaudio.save(save_dir / f"{filename}_clean.wav", clean.unsqueeze(0), cfg.audio.sample_rate)

    print("Calculating metrics...")
    metrics = compute_metrics(
        cfg.metrics,
        pred_wavs,
        clean_wavs,
        cfg.audio.sample_rate,
        device=device,
    )

    print("\n── Test Results ───────────────────────────")
    for k, v in sorted(metrics.items()):
        print(f"  {k:>12s}: {v:.4f}")
    print()

    append_metrics_csv(args.output_csv, step=0, epoch=0, metrics=metrics)
    print(f"Metrics saved to {args.output_csv}")


if __name__ == "__main__":
    main()
