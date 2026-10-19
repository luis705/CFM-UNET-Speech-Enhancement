#!/usr/bin/env python3
"""
Continuous Batching Router para limpeza de áudio.
Suporta ficheiros locais e datasets Hugging Face (.arrow).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
import torchaudio
from einops import rearrange

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.unet import CFMUnet
from src.cfm.solver import ode_solve
from src.data.transforms import preprocess_item, stft_to_waveform

BLOCK_SECONDS = 15


def load_pure_cfm_model(weights_path: Path, device: torch.device) -> tuple[CFMUnet, dict]:
    payload = torch.load(weights_path, map_location="cpu", weights_only=False)
    cfg = payload["cfg"]
    cfg_model = cfg["model"]
    
    unet_model = CFMUnet(
        in_channels=cfg_model.get("in_channels", 4),
        out_channels=cfg_model.get("out_channels", 2),
        time_emb_dim=cfg_model["time_emb_dim"],
        projection_dim=cfg_model["projection_dim"],
        n_blocks=cfg_model["n_blocks"],
        compression_factor=cfg_model["compression_factor"],
        channels_factor=cfg_model["channels_factor"],
        num_heads=cfg_model["num_heads"],
        ff_mult=cfg_model["ff_mult"],
        n_groups=cfg_model["n_groups"],
        gradient_checkpointing=False
    )
    
    unet_model.load_state_dict(payload["model_state_dict"])
    unet_model.eval()
    unet_model.to(device)
    
    return unet_model, cfg


class AudioEnhancer:
    def __init__(self, model: CFMUnet, cfg: dict, device: torch.device):
        self.model = model
        self.device = device
        self.cfg_audio = cfg["audio"]
        self.cfg_ode = cfg["ode"]
        
        self.audio_length_seconds = self.cfg_audio["audio_length_seconds"]
        self.sample_rate = self.cfg_audio["sample_rate"]
        self.fft_size = self.cfg_audio["fft_size"]
        self.hop_length = self.cfg_audio["hop_length"]
        self.window_size = self.cfg_audio["window_size"]
        self.alpha = self.cfg_audio["alpha"]

    @torch.no_grad()
    def process_ola(self, noisy_wav: torch.Tensor, n_steps: int = None, chunk_batch_size: int = 4) -> torch.Tensor:
        B, _C, T_orig = noisy_wav.shape
        win_len = int(self.audio_length_seconds * self.sample_rate)
        hop_len = win_len // 2
        n_steps = n_steps or self.cfg_ode.get("test_steps", 10)

        pad_total = hop_len - (T_orig % hop_len)
        noisy_padded = F.pad(noisy_wav, (0, pad_total + hop_len))
        T_pad = noisy_padded.shape[-1]

        chunks = noisy_padded.unfold(-1, win_len, hop_len)
        n_chunks = chunks.shape[-2]
        chunks = rearrange(chunks, "B C N L -> (B N) C L")

        _, _, chunks_stft, _, mean, std = preprocess_item(
            chunks, chunks,
            pad_crop=False, length=win_len, fft_size=self.fft_size,
            hop_length=self.hop_length, window_size=self.window_size,
            alpha=self.alpha, sample_rate=self.sample_rate,
        )

        n_total_chunks = chunks_stft.shape[0]
        final_stfts = []

        for i in range(0, n_total_chunks, chunk_batch_size):
            batch_stft = chunks_stft[i : i + chunk_batch_size]
            solve_out = ode_solve(self.model, batch_stft, n_steps=n_steps)
            final_stfts.append(solve_out.cpu())

        final_stft_tensor = torch.cat(final_stfts, dim=0).to(self.device)
        
        wav_chunks = stft_to_waveform(
            final_stft_tensor, length=win_len, fft_size=self.fft_size,
            hop_length=self.hop_length, window_size=self.window_size,
            alpha=self.alpha, mean=mean, std=std,
        )

        if wav_chunks.shape[-1] != win_len:
            wav_chunks = wav_chunks[..., :win_len]
        if wav_chunks.ndim == 2:
            wav_chunks = wav_chunks.unsqueeze(1)

        window = torch.hann_window(win_len, device=self.device).view(1, 1, win_len)
        wav_chunks = wav_chunks * window
        combined = rearrange(wav_chunks, "(B N) C L -> B (C L) N", B=B)

        output = F.fold(combined, output_size=(1, T_pad), kernel_size=(1, win_len), stride=(1, hop_len))
        window_ones = window.expand(B, n_chunks, -1).transpose(1, 2)
        weights = F.fold(window_ones, output_size=(1, T_pad), kernel_size=(1, win_len), stride=(1, hop_len))

        return (output / (weights + 1e-8)).squeeze(2)[..., :T_orig]


class AudioStreamSession:
    def __init__(self, sample_id: str, wav: torch.Tensor, original_sr: int, target_sr: int):
        self.sample_id = sample_id
        self.sample_rate = target_sr
        self.block_size = BLOCK_SECONDS * target_sr
        
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if original_sr != target_sr:
            wav = torchaudio.functional.resample(wav, original_sr, target_sr)
            
        self.full_audio = wav.squeeze(0)
        self.total_samples = self.full_audio.shape[-1]
        self.cursor = 0
        self.output_buffer = []

    def is_finished(self) -> bool:
        return self.cursor >= self.total_samples

    def read_next_block(self) -> tuple[torch.Tensor, bool, int]:
        end = min(self.cursor + self.block_size, self.total_samples)
        chunk = self.full_audio[self.cursor:end]
        valid_samples = chunk.shape[-1]
        
        self.cursor = end
        eof = self.is_finished()
        
        if valid_samples < self.block_size:
            chunk = F.pad(chunk, (0, self.block_size - valid_samples))
            
        return chunk.unsqueeze(0), eof, valid_samples

    def write_output(self, enhanced_chunk: torch.Tensor, valid_samples: int):
        valid_audio = enhanced_chunk.squeeze(0)[:valid_samples].cpu()
        self.output_buffer.append(valid_audio)

    def save_to_disk(self, output_dir: Path):
        if not self.output_buffer:
            return
        final_wav = torch.cat(self.output_buffer, dim=0).unsqueeze(0)
        out_path = output_dir / f"{self.sample_id}_stream_enhanced.wav"
        torchaudio.save(str(out_path), final_wav, self.sample_rate)


def main():
    parser = argparse.ArgumentParser(description="Continuous Batching Router CFM")
    parser.add_argument("--weights", type=str, required=True, help="Caminho do arquivo cfm_unet_weights.pt")
    parser.add_argument("--input", type=str, default=None, help="Diretório com áudios locais (opcional se usar --hf-dataset)")
    parser.add_argument("--hf-dataset", type=str, default=None, help="Nome do dataset Hugging Face (ex: JacobLinCool/voice_bank-demand-16k)")
    parser.add_argument("--split", type=str, default="test", help="Split do dataset (test, train, validation)")
    parser.add_argument("--max-files", type=int, default=None, help="Limita o número máximo de ficheiros a processar")
    parser.add_argument("--output-dir", type=str, default="./cleaned_audio")
    parser.add_argument("--batch-size", type=int, default=4, help="Número máximo de streams concorrentes na GPU")
    parser.add_argument("--n-steps", type=int, default=10, help="Passos do solver EDO")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--chunk-batch", type=int, default=16, help="Fatias simultâneas reais na GPU (limita a VRAM)")
    args = parser.parse_args()

    if not args.input and not args.hf_dataset:
        raise ValueError("É necessário providenciar um diretório local (--input) ou um dataset (--hf-dataset).")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{device}] Carregando modelo purista (.pt)...")
    model, cfg = load_pure_cfm_model(Path(args.weights).resolve(), device)
    enhancer = AudioEnhancer(model, cfg, device)
    target_sample_rate = cfg["audio"]["sample_rate"]

    pending_queue = []

    # Carregamento Condicional (Hugging Face vs Local)
    if args.hf_dataset:
        print(f"A descarregar/carregar dataset Hugging Face: {args.hf_dataset} (split: {args.split})...")
        from datasets import load_dataset
        ds = load_dataset(args.hf_dataset, split=args.split)
        
        if args.max_files is not None:
            ds = ds.select(range(min(args.max_files, len(ds))))
            
        for i, item in enumerate(ds):
            # Procura pela coluna que contém o áudio ruidoso ('noisy' ou 'audio')
            audio_data = item.get("noisy") or item.get("audio")
            if not audio_data:
                continue
                
            wav = torch.tensor(audio_data["array"], dtype=torch.float32).unsqueeze(0)
            original_sr = audio_data["sampling_rate"]
            sample_id = f"hf_audio_{i:04d}"
            
            pending_queue.append(AudioStreamSession(sample_id, wav, original_sr, target_sample_rate))
    else:
        input_dir = Path(args.input).resolve()
        audio_files = list(input_dir.glob("*.m4a")) + list(input_dir.glob("*.wav"))
        
        if args.max_files is not None:
            audio_files = audio_files[:args.max_files]
            
        for p in audio_files:
            wav, original_sr = torchaudio.load(p)
            pending_queue.append(AudioStreamSession(p.stem, wav, original_sr, target_sample_rate))

    active_streams: list[AudioStreamSession] = []

    print(f"\nA iniciar roteamento de {len(pending_queue)} streams. Batch: {args.batch_size}")

    total_audio_processed = 0.0
    total_gpu_time = 0.0

    while pending_queue or active_streams:
        while len(active_streams) < args.batch_size and pending_queue:
            active_streams.append(pending_queue.pop(0))
            
        if not active_streams:
            break
            
        batch_chunks = []
        batch_metadata = []
        
        for stream in active_streams:
            chunk, eof, valid_samples = stream.read_next_block()
            batch_chunks.append(chunk)
            batch_metadata.append({"stream": stream, "eof": eof, "valid_samples": valid_samples})
            
        noisy_batch = torch.stack(batch_chunks).to(device)
        
        with torch.inference_mode():
            start_t = perf_counter()
            
            enhanced_batch = enhancer.process_ola(noisy_batch, n_steps=args.n_steps, chunk_batch_size=args.chunk_batch)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            proc_time = perf_counter() - start_t
            total_gpu_time += proc_time
            
        print(f"Lote finalizado: {len(active_streams)} streams | Tempo GPU: {proc_time:.3f}s")
        
        streams_to_remove = []
        for i, meta in enumerate(batch_metadata):
            stream = meta["stream"]
            stream.write_output(enhanced_batch[i], meta["valid_samples"])
            total_audio_processed += meta["valid_samples"] / target_sample_rate
            
            if meta["eof"]:
                stream.save_to_disk(output_dir)
                streams_to_remove.append(stream)
                print(f"  [✔] Stream concluída: {stream.sample_id}")
                
        for stream in streams_to_remove:
            active_streams.remove(stream)

    rtx = total_audio_processed / total_gpu_time if total_gpu_time > 0 else 0
    print("\n" + "="*50)
    print("Processamento concluído.")
    print(f"Velocidade global do Router: {rtx:.2f}x (segundos de áudio / seg de processamento)")
    print("="*50)

if __name__ == "__main__":
    main()