# from torch.utils.data import Datasetfrom __future__ import annotations

from datasets import load_dataset
import torch
from torch.utils.data import Dataset
import torchaudio

from src.data.transforms import pad_or_crop
from src.utils.settings import AudioConfig


class VBDMDDataset(Dataset):
    """VoiceBank-DEMAND dataset wrapper.

    Each sample returns:
        noisy  - (1, T) waveform tensor
        clean  - (1, T) waveform tensor
    """

    def __init__(
        self,
        split: str,
        hf_repo: str,
        audio_cfg: AudioConfig,
        cache_dir: str | None = None,
        length: int = 0,
        pad_or_crop: bool = False,
    ) -> None:
        super().__init__()
        self.audio_cfg = audio_cfg
        self.ds = load_dataset(hf_repo, split=split, cache_dir=cache_dir)
        self.length = length
        self.pad_or_crop = pad_or_crop

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.ds[idx]

        noisy_audio = item["noisy"]["array"]
        clean_audio = item["clean"]["array"]
        sr = item["noisy"]["sampling_rate"]

        noisy = torch.tensor(noisy_audio, dtype=torch.float32).unsqueeze(0)
        clean = torch.tensor(clean_audio, dtype=torch.float32).unsqueeze(0)

        # Resample if needed
        if sr != self.audio_cfg.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.audio_cfg.sample_rate)
            noisy = resampler(noisy)
            clean = resampler(clean)

        if self.pad_or_crop:
            noisy, clean = pad_or_crop(self.length, noisy, clean)

        return noisy, clean


if __name__ == "__main__":
    from src.utils.settings import get_settings

    settings = get_settings()
    dataset = VBDMDDataset(
        split="train",
        hf_repo=settings.dataset.hf_repo,
        audio_cfg=settings.audio,
        cache_dir=settings.dataset.cache_dir,
    )
    print(dataset[0])
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
