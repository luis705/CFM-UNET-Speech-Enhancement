# from torch.utils.data import Datasetfrom __future__ import annotations

from datasets import load_dataset
import torch
from torch.utils.data import Dataset
import torchaudio

from src.data.transforms import preprocess_item
from src.utils.settings import AudioConfig


class VBDMDDataset(Dataset):
    """VoiceBank-DEMAND dataset wrapper.

    Each sample returns:
        noisy_stft  - (2, F, T) real tensor  [real, imag] of compressed noisy STFT
        clean_stft  - (2, F, T) real tensor  [real, imag] of compressed clean STFT
        mean        - scalar tensor (z-score mean)
        std         - scalar tensor (z-score std)
    """

    def __init__(
        self,
        split: str,
        hf_repo: str,
        audio_cfg: AudioConfig,
        cache_dir: str | None = None,
        pad_crop: bool = True,
    ) -> None:
        super().__init__()
        self.audio_cfg = audio_cfg
        self.ds = load_dataset(hf_repo, split=split, cache_dir=cache_dir)
        self.pad_crop = pad_crop

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        return preprocess_item(
            noisy_audio=noisy,
            clean_audio=clean,
            pad_crop=self.pad_crop,
            length=self.audio_cfg.audio_length_seconds * self.audio_cfg.sample_rate,
            fft_size=self.audio_cfg.fft_size,
            hop_length=self.audio_cfg.hop_length,
            window_size=self.audio_cfg.window_size,
            alpha=self.audio_cfg.alpha,
        )


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
    print(dataset[0][2])
