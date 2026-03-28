"""Inference entry-point - enhance a single audio file or folder."""

import argparse
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from src.config import Settings
from src.models.se_model import SEModel


def enhance_file(
    model: SEModel,
    input_path: Path,
    output_path: Path,
    cfg: Settings,
    solver: str | None = None,
    num_steps: int | None = None,
    atol: float | None = None,
    rtol: float | None = None,
) -> None:
    wav, sr = torchaudio.load(str(input_path))
    if sr != cfg.audio.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, cfg.audio.sample_rate)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)  # (T,)

    device = next(model.parameters()).device
    with torch.no_grad():
        pred = model.enhance_long(
            wav.to(device),
            solver=solver,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        ).cpu()

    sf.write(str(output_path), pred.numpy(), cfg.audio.sample_rate)


def main() -> None:
    parser = argparse.ArgumentParser(description="CFM-SE Inference")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Input wav file or directory")
    parser.add_argument("--output", type=str, required=True, help="Output wav file or directory")
    parser.add_argument("--solver", type=str, default=None, help="ODE solver override")
    parser.add_argument("--num-steps", type=int, default=None, help="ODE steps override")
    parser.add_argument("--chunk-size", type=int, default=None, help="OLA chunk size in samples")
    parser.add_argument("--overlap", type=float, default=None, help="OLA overlap ratio (0..1)")
    parser.add_argument("--atol", type=float, default=None, help="ODE absolute tolerance (adaptive solvers)")
    parser.add_argument("--rtol", type=float, default=None, help="ODE relative tolerance (adaptive solvers)")
    args = parser.parse_args()

    cfg = Settings.from_yaml(args.config)

    # Apply CLI overrides to inference config
    if args.chunk_size is not None:
        cfg.inference.chunk_size = args.chunk_size
    if args.overlap is not None:
        cfg.inference.overlap = args.overlap

    model = SEModel.load_from_checkpoint(args.checkpoint, cfg=cfg)
    model.eval()
    model.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Resolve ODE solver defaults (use test-quality settings for inference)
    solver = args.solver if args.solver is not None else cfg.ode.test.solver
    num_steps = args.num_steps if args.num_steps is not None else cfg.ode.test.num_steps
    atol = args.atol if args.atol is not None else cfg.ode.test.atol
    rtol = args.rtol if args.rtol is not None else cfg.ode.test.rtol

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enhance_file(model, input_path, output_path, cfg, solver, num_steps, atol, rtol)
        print(f"Enhanced: {input_path} \u2192 {output_path}")
    elif input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        wav_files = sorted(input_path.glob("*.wav"))
        for wf in wav_files:
            out_f = output_path / wf.name
            enhance_file(model, wf, out_f, cfg, solver, num_steps, atol, rtol)
            print(f"Enhanced: {wf.name}")
        print(f"\nDone. {len(wav_files)} files written to {output_path}")
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")


if __name__ == "__main__":
    main()
