#!/usr/bin/env python3
"""CLI para limpar áudio usando um checkpoint treinado do CFM."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import pytorch_lightning as pl
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cfm.lightning_module import CFMModule
from src.utils.callbacks import LoggingCallback, SlowMetricsCallback
from src.utils.settings import get_settings

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".webm"}


class AudioInferenceDataset(Dataset):
    def __init__(self, paths: list[Path], sample_rate: int):
        self.paths = paths
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.paths[index]
        wav, sr = torchaudio.load(path)

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        wav = wav.to(torch.float32)
        return {
            "noisy": wav,
            "clean": wav,
            "sample_id": path.stem,
        }


def _collect_audio_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in AUDIO_EXTENSIONS else []

    if path.is_dir():
        files: list[Path] = []
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix.lower() in AUDIO_EXTENSIONS:
                files.append(child)
        return files

    return []


def _save_audio(output_path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), wav.detach().cpu().squeeze(0), sample_rate)

def pad_collate_fn(batch):
    # Encontra o tamanho máximo de áudio neste lote específico
    max_len = max(item["noisy"].shape[-1] for item in batch)
    
    batched_noisy = []
    batched_clean = []
    sample_ids = []
    
    for item in batch:
        noisy = item["noisy"]
        clean = item["clean"]
        
        # Calcula quanto falta para chegar no tamanho máximo
        pad_size = max_len - noisy.shape[-1]
        
        # F.pad adiciona zeros. O formato (0, pad_size) preenche apenas à direita (no final do áudio)
        noisy_padded = F.pad(noisy, (0, pad_size))
        clean_padded = F.pad(clean, (0, pad_size))
        
        batched_noisy.append(noisy_padded)
        batched_clean.append(clean_padded)
        sample_ids.append(item["sample_id"])
        
    return {
        "noisy": torch.stack(batched_noisy),
        "clean": torch.stack(batched_clean),
        "sample_id": sample_ids
    }


def _build_datamodule(input_path: Path, sample_rate: int, batch_size: int = 1) -> DataLoader:
    files = _collect_audio_files(input_path)
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo de áudio foi encontrado em: {input_path}")

    dataset = AudioInferenceDataset(files, sample_rate)
    
    # Adicionamos o batch_size e o collate_fn aqui
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=pad_collate_fn
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Limpa áudio(s) usando um checkpoint treinado do CFM.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Caminho do arquivo .ckpt do modelo treinado")
    parser.add_argument("--input", type=str, required=True, help="Arquivo de áudio ou diretório com áudios")
    parser.add_argument("--output-dir", type=str, default="./cleaned_audio", help="Diretório para salvar os áudios limpos")
    parser.add_argument("--config", type=str, default="settings.yaml", help="Arquivo de config do projeto")
    parser.add_argument("--n-steps", type=int, default=None, help="Número de passos do ODE (opcional; se omitido usa o do checkpoint/config)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Dispositivo para inferência")
    args = parser.parse_args()

    settings = get_settings(args.config)
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device if args.device != "auto" else "cpu"
    )

    input_path = Path(args.input).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo/diretório de entrada não encontrado: {input_path}")

    model = CFMModule.load_from_checkpoint(str(checkpoint_path), cfg=settings, strict=False)
    model.eval()
    model.to(device)
    if args.n_steps is not None:
        model.ode_test_steps = args.n_steps

    settings.test.save_dir = str(output_dir)
    settings.logging.log_dir = str(output_dir / "logs")

    dataloader = _build_datamodule(input_path, settings.audio.sample_rate)

    model.test_audio_buffer.clear()

    total_audio_duration = 0.0
    total_processing_time = 0.0

    for batch in tqdm(dataloader):
        noisy = batch["noisy"].to(device)
        
        # Calcula a duração do áudio em segundos
        audio_duration = noisy.shape[-1] / settings.audio.sample_rate
        total_audio_duration += audio_duration

        with torch.inference_mode():
            start = perf_counter()
            enhanced = model.process_ola(noisy, n_steps=model.ode_test_steps)
            
            # Sincroniza a GPU para garantir uma medição de tempo precisa
            if device.type == "cuda":
                torch.cuda.synchronize()
                
            end = perf_counter()
            
        proc_time = end - start
        total_processing_time += proc_time
        
        # Itera sobre cada áudio gerado dentro do lote
        for i in range(enhanced.shape[0]):
            sample_name = str(batch["sample_id"][i])
            if sample_name.endswith(".wav"):
                sample_name = sample_name[:-4]

            out_path = output_dir / f"{sample_name}_enhanced.wav"
            
            # enhanced[i] tem shape (Canal, Tempo). 
            # Usamos unsqueeze(0) pois o seu _save_audio atual aplica um squeeze(0) na entrada.
            _save_audio(out_path, enhanced[i].unsqueeze(0), settings.audio.sample_rate)

    audio_per_sec = total_audio_duration / total_processing_time

    print(f"Áudios limpos salvos em: {output_dir}")
    print(f"Total de áudio processado: {total_audio_duration:.2f} s")
    print(f"Tempo total de processamento: {total_processing_time:.2f} s")
    print(f"Velocidade média: {audio_per_sec:.2f} segundos de áudio por segundo de processamento (RTx)")

if __name__ == "__main__":
    main()
