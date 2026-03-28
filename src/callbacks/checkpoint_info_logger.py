from datetime import datetime

import pytorch_lightning as pl


class CheckpointLoggerCallback(pl.Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if not trainer.logger:
            return
        timestamp = datetime.now().isoformat()
        trainer.logger.experiment.add_text(
            "checkpoint_saved", f"Checkpoint salvo em {timestamp}", global_step=trainer.global_step
        )
