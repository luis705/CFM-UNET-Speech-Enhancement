from __future__ import annotations

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from src.data.dataset import VBDMDDataset
from src.utils.settings import Settings


class VBDMDDataModule(pl.LightningDataModule):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self._train_ds = None
        self._val_ds = None
        self._test_ds = None

    def setup(self, stage: str | None = None) -> None:
        cfg = self.settings

        if stage in ("fit", None):
            full_train = VBDMDDataset(
                split="train",
                hf_repo=cfg.dataset.hf_repo,
                audio_cfg=cfg.audio,
                cache_dir=cfg.dataset.cache_dir,
                pad_crop=True
            )
            n_val = int(len(full_train) * cfg.data.val_split)
            n_train = len(full_train) - n_val
            self._train_ds, self._val_ds = random_split(full_train, [n_train, n_val])
            self._val_ds.pad_crop = False

        if stage in ("test", None):
            self._test_ds = VBDMDDataset(
                split="test",
                hf_repo=cfg.dataset.hf_repo,
                audio_cfg=cfg.audio,
                cache_dir=cfg.dataset.cache_dir,
                pad_crop=False
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self.settings.data.batch_size,
            num_workers=self.settings.data.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            batch_size=self.settings.data.batch_size * 2,
            num_workers=self.settings.data.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_ds,
            batch_size=self.settings.data.batch_size * 2,
            num_workers=self.settings.data.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
        )
