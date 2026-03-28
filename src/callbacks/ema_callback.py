"""EMA (Exponential Moving Average) callback for PyTorch Lightning."""

import pytorch_lightning as pl
import torch


class EMACallback(pl.Callback):
    """Maintain EMA shadow parameters and swap them in during validation."""

    def __init__(self, ema_decay: float = 0.999) -> None:
        super().__init__()
        self.ema_decay = ema_decay
        self._ema_params: dict[str, torch.Tensor] = {}
        self._originals: dict[str, torch.Tensor] = {}

    # ── Initialise shadow params ─────────────────────────────────────────

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                self._ema_params[name] = param.data.clone()

    # ── Update EMA after each training batch ─────────────────────────────

    @torch.no_grad()
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx) -> None:
        d = self.ema_decay
        for name, param in pl_module.named_parameters():
            if name in self._ema_params:
                self._ema_params[name].mul_(d).add_(param.data, alpha=1.0 - d)

    # ── Swap to EMA for validation ───────────────────────────────────────

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._originals.clear()
        for name, param in pl_module.named_parameters():
            if name in self._ema_params:
                self._originals[name] = param.data.clone()
                param.data.copy_(self._ema_params[name])

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        for name, param in pl_module.named_parameters():
            if name in self._originals:
                param.data.copy_(self._originals[name])
        self._originals.clear()
