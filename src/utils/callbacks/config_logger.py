import os
import shutil

import pytorch_lightning as pl


class ConfigLoggerCallback(pl.Callback):
    def __init__(self, config_path: str):
        self.config_path = config_path

    def on_train_start(self, trainer, pl_module):
        if not trainer.logger:
            return
        log_dir = trainer.logger.log_dir  # dir do run atual
        if os.path.exists(self.config_path):
            shutil.copy(self.config_path, os.path.join(log_dir, "config.yaml"))
            # Opcional: logar conteúdo como texto também
            with open(self.config_path) as f:
                config_content = f.read()
            trainer.logger.experiment.add_text("config", config_content, global_step=0)

        trainer.logger.experiment.add_text("model_summary", str(pl_module), global_step=0)
