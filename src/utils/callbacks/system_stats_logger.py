import time

import GPUtil
import psutil
import pytorch_lightning as pl


class SystemStatsLoggerCallback(pl.Callback):
    def __init__(self):
        self.batch_timer = None
        self.total_batches = 0
        self.done_batches = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.batch_timer = time.time()
        self.done_batches = 0
        self.total_batches = len(trainer.train_dataloader) * trainer.max_epochs

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        t = time.time() - self.batch_timer
        self.done_batches += 1
        self.batch_timer = time.time()
        eta = (self.total_batches - self.done_batches) * (
            t if self.done_batches == 0 else (trainer.fit_loop.epoch_loop.total_batch_idx + 1 != 0 and t)
        )  # simplificado

        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        gpus = GPUtil.getGPUs()
        gpu_util, vram, temp = 0.0, 0.0, 0.0
        if gpus:
            gpu = gpus[0]
            gpu_util = gpu.load * 100
            vram = gpu.memoryUtil * 100
            temp = gpu.temperature

        pl_module.log("perf/batch_time", t, on_step=True, logger=True, prog_bar=False)
        pl_module.log("perf/eta", eta, on_step=True, logger=True, prog_bar=False)
        pl_module.log("perf/cpu", cpu, on_step=True, logger=True, prog_bar=False)
        pl_module.log("perf/ram", ram, on_step=True, logger=True, prog_bar=False)
        pl_module.log("perf/gpu", gpu_util, on_step=True, logger=True, prog_bar=False)
        pl_module.log("perf/vram", vram, on_step=True, logger=True, prog_bar=False)
        pl_module.log("perf/temp", temp, on_step=True, logger=True, prog_bar=False)
