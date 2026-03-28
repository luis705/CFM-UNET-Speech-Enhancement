"""Training entry-point."""

import argparse
from pathlib import Path
import warnings

import matplotlib
import torch
from torch.jit import TracerWarning
import torch.multiprocessing as mp

from src.config import Settings
from src.data.dataset import build_dataloaders
from src.models.se_model import SEModel
from src.trainer import build_trainer


def _find_latest_checkpoint(dirpath: str) -> Path | None:
    """Return the most recent checkpoint in *dirpath*, preferring ``last.ckpt``."""
    ckpt_dir = Path(dirpath)
    if not ckpt_dir.is_dir():
        return None

    last = ckpt_dir / "last.ckpt"
    if last.is_file():
        return last

    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0] if ckpts else None


def _find_latest_session(dirpath: str) -> str | None:
    """Return the most recent session in *dirpath*."""
    session_dir = Path(dirpath)
    if not session_dir.is_dir():
        return None

    sessions = sorted(session_dir.glob("session-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(sessions[0]) if sessions else None


def _prompt_resume(ckpt_path: Path) -> bool:
    """Ask the user whether to resume from *ckpt_path*."""
    print(f"\n[checkpoint] Found existing checkpoint: {ckpt_path}")
    while True:
        answer = input("Resume from this checkpoint? [y/n]: ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please answer 'y' or 'n'.")


def _try_resume(ckpt_path: Path, model: torch.nn.Module) -> Path | None:
    """Validate that *ckpt_path* is compatible with *model*.

    Returns the path if compatible, ``None`` otherwise (with a warning).
    """
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", {})
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())

        if model_keys != ckpt_keys:
            missing = model_keys - ckpt_keys
            unexpected = ckpt_keys - model_keys
            parts = ["[checkpoint] WARNING: Model architecture differs from checkpoint."]
            if missing:
                parts.append(
                    f"  Missing keys  ({len(missing)}): {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
                )
            if unexpected:
                parts.append(
                    f"  Unexpected keys ({len(unexpected)}): {sorted(unexpected)[:5]}"
                    f"{'...' if len(unexpected) > 5 else ''}"
                )
            parts.append("[checkpoint] Starting training from scratch.")
            print("\n".join(parts))
            return None

        print(f"[checkpoint] Checkpoint is compatible. Resuming from {ckpt_path}")
        return ckpt_path
    except Exception as exc:
        print(f"[checkpoint] WARNING: Failed to load checkpoint ({exc}). Starting from scratch.")
        return None


def _resolve_checkpoint(cfg: Settings) -> Path | None:
    """Detect checkpoint, prompt the user, and validate compatibility."""
    ckpt_path = _find_latest_checkpoint(cfg.checkpoint.dirpath)
    if ckpt_path is None:
        return None
    if not _prompt_resume(ckpt_path):
        return None
    return ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CFM-SE model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = Settings.from_yaml(args.config)
    train_loader, val_loader, _ = build_dataloaders(cfg.data, cfg.audio)
    model = SEModel(cfg)

    ckpt_path = _resolve_checkpoint(cfg)
    if ckpt_path is not None:
        ckpt_path = _try_resume(ckpt_path, model)

    version = None
    if ckpt_path is None:
        version = _find_latest_session(cfg.checkpoint.dirpath)

    trainer = build_trainer(cfg, args.config, version)
    trainer.fit(model, train_loader, val_loader, ckpt_path=str(ckpt_path) if ckpt_path else None)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=TracerWarning)
    matplotlib.use("Agg")  # Backend não-interativo para evitar problemas em ambientes headless

    mp.set_start_method("spawn", force=True)
    torch.set_float32_matmul_precision("high")

    main()
