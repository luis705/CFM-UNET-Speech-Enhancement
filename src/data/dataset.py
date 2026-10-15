
import math
from pathlib import Path
import re
from typing import Any

from datasets import load_dataset
import torch
from torch.utils.data import Dataset
import torchaudio

from src.data.transforms import pad_or_crop, add_stochastic_noise
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
        augment: bool = True,
        augment_pct: float = 0.25,
        include_metadata: bool = False,
    ) -> None:
        super().__init__()
        self.audio_cfg = audio_cfg
        self.ds = load_dataset(hf_repo, split=split, cache_dir=cache_dir)
        self.length = length
        self.pad_or_crop = pad_or_crop
        self.augment = augment
        self.augment_pct = augment_pct
        self.include_metadata = include_metadata


    def __len__(self) -> int:
        return len(self.ds)

    @staticmethod
    def _parse_snr_from_path(noisy_path: str | None) -> float | None:
        if not noisy_path:
            return None

        match = re.search(r"snr(?P<snr>-?\d+(?:[p\.]\d+)?)dB", noisy_path, flags=re.IGNORECASE)
        if not match:
            return None

        raw = match.group("snr").replace("p", ".")
        try:
            return float(raw)
        except ValueError:
            return None

    @staticmethod
    def _parse_noise_type_from_path(noisy_path: str | None) -> str | None:
        if not noisy_path:
            return None

        stem = Path(noisy_path).stem
        match = re.search(r"_(?P<noise>[^_]+)_snr", stem, flags=re.IGNORECASE)
        if match:
            return match.group("noise")

        return None

    def _extract_metadata(self, item: dict[str, Any], idx: int) -> dict[str, Any]:
        noisy_info = item.get("noisy") if isinstance(item.get("noisy"), dict) else {}
        noisy_path = noisy_info.get("path")

        # 1. Tratamento seguro para SNR (Float)
        snr_raw = item.get("snr_db")
        snr_db = float('nan')  # Usa NaN (Not a Number) em vez de None
        
        if snr_raw is None:
            parsed_snr = self._parse_snr_from_path(noisy_path)
            if parsed_snr is not None:
                snr_db = float(parsed_snr)
        else:
            try:
                snr_db = float(snr_raw)
            except (TypeError, ValueError):
                pass # Mantém como float('nan')

        # 2. Tratamento seguro para o tipo de ruído (String)
        noise_type = item.get("noise_type")
        if noise_type is None:
            noise_type = self._parse_noise_type_from_path(noisy_path)
            
        # Substitui None por "unknown" para o collate aceitar a string
        safe_noise_type = str(noise_type) if noise_type is not None else "unknown"

        sample_id = (
            item.get("sample_id")
            or item.get("id")
            or item.get("file")
            or noisy_path
            or str(idx)
        )

        return {
            "snr_db": snr_db, 
            "snr_db_estimated": False,
            "noise_type": safe_noise_type,
            "sample_id": str(sample_id),
        }

    @staticmethod
    def _estimate_snr_db(noisy: torch.Tensor, clean: torch.Tensor) -> float:
        """Estimate SNR in dB from aligned noisy and clean signals."""
        noisy_1d = noisy.flatten().to(torch.float32)
        clean_1d = clean.flatten().to(torch.float32)
        noise_1d = noisy_1d - clean_1d

        signal_power = torch.mean(clean_1d * clean_1d)
        noise_power = torch.mean(noise_1d * noise_1d)

        # Clamp powers for numerical stability in near-silent regions.
        signal_power = torch.clamp(signal_power, min=1e-12)
        noise_power = torch.clamp(noise_power, min=1e-12)

        snr_db = 10.0 * torch.log10(signal_power / noise_power)
        return float(snr_db.item())

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | dict[str, Any]:
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

        if self.augment and torch.rand(1) < self.augment_pct:
            noisy = add_stochastic_noise(noisy)

        if not self.include_metadata:
            return noisy, clean

        metadata = self._extract_metadata(item, idx)

        # When source metadata has no SNR label, estimate it from the pair.
        snr_val = metadata.get("snr_db")
        if isinstance(snr_val, (float, int)) and math.isnan(float(snr_val)):
            metadata["snr_db"] = self._estimate_snr_db(noisy, clean)
            metadata["snr_db_estimated"] = True

        return {
            "noisy": noisy,
            "clean": clean,
            **metadata,
        }


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
