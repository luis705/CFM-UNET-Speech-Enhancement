from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from src.data.dataset import VBDMDDataset
from src.utils.settings import Settings


class VBDMDDataModule(pl.LightningDataModule):
    def __init__(self, settings: Settings, test_samples_limit: int = -1) -> None:
        super().__init__()
        self.settings = settings
        self.test_samples_limit = test_samples_limit
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
                pad_or_crop=True,
                length=cfg.audio.audio_length_seconds * cfg.audio.sample_rate,
                augment=cfg.data.augment,
                augment_pct=cfg.data.augment_pct,
            )
            n_val = int(len(full_train) * cfg.data.val_split)
            n_train = len(full_train) - n_val

            # Usamos um gerador com seed fixa para garantir que o split seja sempre o mesmo
            train_subset, val_subset = random_split(
                full_train, [n_train, n_val], generator=torch.Generator().manual_seed(42)
            )

            self._train_ds = train_subset

            # Criamos um novo dataset base para validação explicitamente sem pad_or_crop
            full_val_var_len = VBDMDDataset(
                split="train",
                hf_repo=cfg.dataset.hf_repo,
                audio_cfg=cfg.audio,
                cache_dir=cfg.dataset.cache_dir,
                pad_or_crop=False,
                augment=False,
                include_metadata=True,
            )
            # Aplicamos os índices do subset de validação ao dataset sem crop
            self._val_ds = torch.utils.data.Subset(full_val_var_len, val_subset.indices)

        if stage in ("test", None):
            self._test_ds = VBDMDDataset(
                split="test",
                hf_repo=cfg.dataset.hf_repo,
                audio_cfg=cfg.audio,
                cache_dir=cfg.dataset.cache_dir,
                pad_or_crop=False,
                augment=False,
                include_metadata=True,
            )
            if self.test_samples_limit > 0 and len(self._test_ds) > self.test_samples_limit:
                # Limita o dataset de teste ao número especificado de amostras
                self._test_ds = torch.utils.data.Subset(self._test_ds, range(self.test_samples_limit))

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
            batch_size=1,
            num_workers=self.settings.data.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_ds,
            batch_size=1,
            num_workers=self.settings.data.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
        )
