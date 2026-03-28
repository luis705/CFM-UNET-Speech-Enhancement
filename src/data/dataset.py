"""VoiceBank-DEMAND dataset loader backed by HuggingFace Datasets (Arrow format).

Audio is stored in Apache Arrow files (memory-mapped, columnar) and decoded on
the fly, which is significantly faster than reading individual WAV files.
The dataset is automatically downloaded and cached on first use.
"""

from datasets import Dataset as HFDataset, load_dataset
import torch
from torch.utils.data import DataLoader, Dataset
import torchaudio

from src.config import AudioConfig, DataConfig
from src.data.transforms import pad_to_segment


class VoiceBankDEMAND(Dataset):
    """Paired clean / noisy speech dataset backed by HuggingFace Datasets.

    Wraps a HuggingFace ``datasets.Dataset`` split.  Audio is stored in
    Apache Arrow format (memory-mapped, columnar) and decoded on the fly.
    """

    def __init__(
        self,
        hf_split: HFDataset,
        audio_cfg: AudioConfig,
        clean_column: str = "clean",
        noisy_column: str = "noisy",
        filename_column: str = "id",
        segment_length: int = 32_000,
        train: bool = True,
    ) -> None:
        self.hf_split = hf_split
        self.clean_column = clean_column
        self.noisy_column = noisy_column
        self.filename_column = filename_column
        self.segment_length = segment_length
        self.sample_rate = audio_cfg.sample_rate
        self.train = train

    def __len__(self) -> int:
        return len(self.hf_split)

    def _to_tensor(self, audio_entry: dict) -> torch.Tensor:
        """Convert a HuggingFace Audio dict ``{array, sampling_rate}`` to a 1-D tensor."""
        wav = torch.as_tensor(audio_entry["array"], dtype=torch.float32)
        sr = audio_entry["sampling_rate"]
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if wav.ndim > 1:
            wav = wav.mean(dim=0)
        return wav

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        example = self.hf_split[idx]
        clean = self._to_tensor(example[self.clean_column])
        noisy = self._to_tensor(example[self.noisy_column])

        # Ensure same length
        min_len = min(clean.shape[-1], noisy.shape[-1])
        clean = clean[:min_len]
        noisy = noisy[:min_len]

        if self.segment_length > 0 and min_len >= self.segment_length:
            # Para treino - random crop
            # Para validação - start crop
            start = torch.randint(0, min_len - self.segment_length + 1, (1,)).item() if self.train else 0
            clean = clean[start : start + self.segment_length]
            noisy = noisy[start : start + self.segment_length]
        else:
            clean = pad_to_segment(clean, self.segment_length)
            noisy = pad_to_segment(noisy, self.segment_length)

        filename = example.get(self.filename_column, str(idx))
        return {"clean": clean, "noisy": noisy, "filename": filename}


def load_hf_split(data_cfg: DataConfig, split: str) -> HFDataset:
    """Load a single split from the HuggingFace dataset (downloads on first use)."""
    return load_dataset(
        data_cfg.hf_dataset,
        split=split,
        cache_dir=data_cfg.hf_cache_dir,
        trust_remote_code=data_cfg.hf_trust_remote_code,
    )


def build_dataloaders(
    data_cfg: DataConfig,
    audio_cfg: AudioConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader)."""
    train_hf = load_hf_split(data_cfg, data_cfg.train_split)
    test_hf = load_hf_split(data_cfg, data_cfg.test_split)
    val_hf = None

    if data_cfg.val_split_from_test is not None:
        # Carve validation from test split
        parts = test_hf.train_test_split(test_size=data_cfg.val_split_from_test, seed=42)
        test_hf, val_hf = parts["train"], parts["test"]
    elif data_cfg.val_split:
        try:
            val_hf = load_hf_split(data_cfg, data_cfg.val_split)
        except ValueError:
            # Split doesn't exist in the dataset - carve one from train
            parts = train_hf.train_test_split(test_size=data_cfg.val_ratio, seed=42)
            train_hf, val_hf = parts["train"], parts["test"]
    else:
        parts = train_hf.train_test_split(test_size=data_cfg.val_ratio, seed=42)
        train_hf, val_hf = parts["train"], parts["test"]

    kw = {
        "clean_column": data_cfg.clean_column,
        "noisy_column": data_cfg.noisy_column,
        "filename_column": data_cfg.filename_column,
    }

    train_ds = VoiceBankDEMAND(train_hf, audio_cfg, **kw, segment_length=data_cfg.segment_length, train=True)
    val_ds = VoiceBankDEMAND(val_hf, audio_cfg, **kw, segment_length=data_cfg.segment_length, train=False)
    test_ds = VoiceBankDEMAND(test_hf, audio_cfg, **kw, segment_length=data_cfg.segment_length, train=False)

    loader_kwargs = {
        "batch_size": data_cfg.batch_size,
        "num_workers": data_cfg.num_workers,
        "pin_memory": data_cfg.pin_memory,
        "persistent_workers": data_cfg.persistent_workers,
        "prefetch_factor": data_cfg.prefetch_factor,
    }

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=True, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
