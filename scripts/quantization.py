"""Very small example for loading trained weights into the U-Net and benchmarking quantization.

This script is intentionally simple:
- loads the model weights saved by scripts/test.py
- builds a plain CFMUnet
- compares fp32 vs quantized inference on a random batch

It does NOT depend on CFMModule. For quantization, the Lightning wrapper is unnecessary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.unet import CFMUnet
from src.utils.settings import get_settings


def build_unet(cfg) -> CFMUnet:
    return CFMUnet(
        in_channels=4,
        out_channels=2,
        time_emb_dim=cfg.model.time_emb_dim,
        projection_dim=cfg.model.projection_dim,
        n_blocks=cfg.model.n_blocks,
        compression_factor=cfg.model.compression_factor,
        channels_factor=cfg.model.channels_factor,
        num_heads=cfg.model.num_heads,
        ff_mult=cfg.model.ff_mult,
        n_groups=cfg.model.n_groups,
        gradient_checkpointing=False,
    )


def load_from_pt(model: CFMUnet, weights_path: str | Path) -> CFMUnet:
    payload = torch.load(str(weights_path), map_location="cpu")

    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print("Missing keys:", missing[:10])
        print("Unexpected keys:", unexpected[:10])

    return model


def forward_once(model: nn.Module, device: torch.device) -> torch.Tensor:
    model.eval()
    x = torch.randn(1, 4, 256, 128, device=device)
    t = torch.rand(1, device=device)

    with torch.no_grad():
        y = model(x, t)

    print(f"\tOutput shape: {tuple(y.shape)}")
    # Iterando sobre os módulos quantizados para ver o tipo real dos pesos
    for name, module in model.named_modules():
        if hasattr(module, "_packed_params"):
            # Usa a op nativa do PyTorch para desempacotar pesos e bias
            weight, bias = torch.ops.quantized.linear_unpack(module._packed_params)
            print(f"Camada {name} - Weight dtype: {weight.dtype}")
    return y


def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    model.eval()
    qmodel = torch.quantization.quantize_dynamic(
        model,
        {nn.Conv2d, nn.Linear},
        dtype=torch.qint8,
    )
    return qmodel


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal quantization smoke test for CFMUnet")
    parser.add_argument("--weights", type=str, required=True, help="Path to the .pt file with the trained weights")
    parser.add_argument("--config", type=str, default="settings.yaml", help="Path to the project settings YAML")
    args = parser.parse_args()

    cfg = get_settings(args.config)

    device = torch.device("cpu")

    fp32_model = build_unet(cfg).to(device)
    fp32_model = load_from_pt(fp32_model, args.weights)
    print("Original model ok:")
    forward_once(fp32_model, device)

    quantized_model = apply_dynamic_quantization(fp32_model.cpu())
    print("Quantized model ok:")
    forward_once(quantized_model, device)

    out_path = Path(args.weights).with_name("cfm_unet_quantized.pt")
    torch.save(quantized_model.state_dict(), out_path)
    print(f"Saved quantized state dict to: {out_path}")


if __name__ == "__main__":
    main()
