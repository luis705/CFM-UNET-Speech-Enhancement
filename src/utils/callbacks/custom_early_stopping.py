from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping


class MinEpochEarlyStopping(EarlyStopping):
    """
    Early stopping callback que só começa a contar a paciência após um número mínimo de épocas.
    """

    def __init__(self, min_epochs: int = 0, **kwargs: Any):
        super().__init__(**kwargs)
        self.min_epochs = min_epochs

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.current_epoch < self.min_epochs:
            return
        super().on_validation_end(trainer, pl_module)
