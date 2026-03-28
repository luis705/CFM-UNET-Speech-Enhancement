import os
import shutil

import pytorch_lightning as pl


class SystemStatsLoggerCallback(pl.Callback):
    def __init__(self, metrics_path: str):
        self.metrics_path = metrics_path

    def on_fit_end(self, trainer, pl_module):
        if not trainer.logger:
            return
        if os.path.exists(self.metrics_path):
            shutil.copy(self.metrics_path, os.path.join(trainer.logger.log_dir, "metrics.csv"))
