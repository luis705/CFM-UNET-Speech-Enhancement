#!/usr/bin/env python3
"""Generate noisy WAV copies at multiple SNRs with white, colored and external noise."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from tqdm import tqdm

NOISE_BETA: Dict[str, float] = {
    "white": 0.0,
    "pink": 1.0,
    "brown": 2.0,
    "blue": -1.0,
    "violet": -2.0,
}

WORKER_SIGNAL: np.ndarray | None = None
WORKER_DTYPE: np.dtype[Any] | None = None
WORKER_SAMPLE_RATE: int | None = None
WORKER_ACTIVE_THRESHOLD_DB: float = 40.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recebe um WAV de entrada e gera copias com ruido branco/colorido/external "
            "em diferentes niveis de SNR."
        )
    )
    parser.add_argument("input_wav", type=Path, help="Caminho para o arquivo WAV de entrada.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("outputs_noisy"),
        help="Diretorio de saida (padrao: outputs_noisy).",
    )
    parser.add_argument(
        "--snr-db",
        type=float,
        nargs="+",
        default=[0.0, 5.0, 10.0, 20.0],
        help="Lista de valores de SNR em dB (padrao: 0 5 10 20).",
    )
    parser.add_argument(
        "--copies-per-snr",
        type=int,
        default=2,
        help="Quantidade de copias por combinacao (padrao: 2).",
    )
    parser.add_argument(
        "--noise-types",
        nargs="+",
        choices=sorted(NOISE_BETA.keys()),
        default=list(NOISE_BETA.keys()),
        help="Tipos de ruido colorido/aleatorio a gerar.",
    )
    parser.add_argument(
        "--active-threshold-db",
        type=float,
        default=40.0,
        help=(
            "Threshold em dB abaixo do pico de energia por frame para detectar voz ativa "
            "(padrao: 40)."
        ),
    )
    parser.add_argument(
        "--external-noise-path",
        type=Path,
        default=Path('noise'),
        help="Arquivo WAV ou diretorio com WAVs de ruido nao-estacionario.",
    )
    parser.add_argument(
        "--external-noise-copies",
        type=int,
        default=None,
        help=(
            "Quantidade de amostras de ruidos externos a selecionar. Cada amostra gera "
            "--copies-per-snr arquivos em modo aleatorio. Se omitido com --external-full-sweep, "
            "usa todos os WAVs externos."
        ),
    )
    parser.add_argument(
        "--external-mixed-copies",
        type=int,
        default=None,
        help=(
            "Quantidade de amostras de ruidos externos para modo externo+aleatorio. "
            "Cada amostra gera --copies-per-snr arquivos em modo aleatorio. Se omitido com "
            "--external-full-sweep, usa todos os WAVs externos e combina com noise-types."
        ),
    )
    parser.add_argument(
        "--external-full-sweep",
        action="store_true",
        help=(
            "Se habilitado, externo e externo+misto varrem toda a lista de SNR "
            "(e no misto tambem todos os noise-types)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Numero de processos paralelos (padrao: nucleos logicos).",
    )
    parser.add_argument(
        "--metadata-format",
        choices=["none", "csv", "json", "both"],
        default="csv",
        help="Formato de metadados de saida (padrao: csv).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente aleatoria para reproducibilidade (padrao: 42).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="aug",
        help="Prefixo no nome dos arquivos de saida (padrao: aug).",
    )
    return parser.parse_args()


def to_float_signal(audio: np.ndarray) -> np.ndarray:
    if audio.dtype.kind in ("i", "u"):
        info = np.iinfo(audio.dtype)
        if audio.dtype.kind == "u":
            midpoint = (info.max + 1) / 2.0
            scale = midpoint
            return (audio.astype(np.float64) - midpoint) / scale
        scale = max(abs(info.min), abs(info.max))
        return audio.astype(np.float64) / float(scale)

    return audio.astype(np.float64)


def from_float_signal(audio: np.ndarray, target_dtype: np.dtype[Any]) -> np.ndarray:
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        # Peak normalization para evitar hard clipping.
        audio = audio / peak

    audio = np.clip(audio, -1.0, 1.0)

    if np.issubdtype(target_dtype, np.floating):
        return audio.astype(target_dtype)

    info = np.iinfo(target_dtype)
    if target_dtype.kind == "u":
        midpoint = (info.max + 1) / 2.0
        restored = np.round(audio * midpoint + midpoint)
    else:
        scale = max(abs(info.min), abs(info.max))
        restored = np.round(audio * scale)

    return np.clip(restored, info.min, info.max).astype(target_dtype)


def colored_noise(num_samples: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    freqs = np.fft.rfftfreq(num_samples)
    scale = np.ones_like(freqs, dtype=np.float64)

    nonzero = freqs > 0
    scale[nonzero] = freqs[nonzero] ** (-beta / 2.0)

    real = rng.normal(size=freqs.shape[0])
    imag = rng.normal(size=freqs.shape[0])
    spectrum = (real + 1j * imag) * scale
    spectrum[0] = 0.0 + 0.0j

    if num_samples % 2 == 0:
        spectrum[-1] = spectrum[-1].real + 0.0j

    noise = np.fft.irfft(spectrum, n=num_samples)
    std = np.std(noise)
    if std <= 1e-12:
        return np.zeros(num_samples, dtype=np.float64)

    return noise / std


def generate_noise(shape: Iterable[int], noise_type: str, rng: np.random.Generator) -> np.ndarray:
    shape_tuple = tuple(shape)
    if len(shape_tuple) == 1:
        return colored_noise(shape_tuple[0], NOISE_BETA[noise_type], rng)

    samples, channels = shape_tuple
    out = np.zeros((samples, channels), dtype=np.float64)
    for ch in range(channels):
        out[:, ch] = colored_noise(samples, NOISE_BETA[noise_type], rng)
    return out


def frame_energy(signal: np.ndarray, frame_size: int = 1024, hop: int = 512) -> np.ndarray:
    mono = signal if signal.ndim == 1 else np.mean(signal, axis=1)

    if mono.size < frame_size:
        return np.array([np.mean(mono**2)], dtype=np.float64)

    energies: List[float] = []
    for start in range(0, mono.size - frame_size + 1, hop):
        frame = mono[start : start + frame_size]
        energies.append(float(np.mean(frame**2)))

    return np.array(energies, dtype=np.float64)


def active_speech_rms(signal: np.ndarray, active_threshold_db: float = 40.0) -> float:
    energies = frame_energy(signal)
    peak_energy = float(np.max(energies))
    if peak_energy <= 1e-14:
        return 0.0

    threshold = peak_energy * (10 ** (-active_threshold_db / 10.0))
    active = energies[energies >= threshold]
    if active.size == 0:
        return 0.0

    return float(np.sqrt(np.mean(active)))


def apply_snr(
    signal: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    active_threshold_db: float = 40.0,
) -> np.ndarray:
    signal_rms = active_speech_rms(signal, active_threshold_db=active_threshold_db)
    noise_rms = np.sqrt(np.mean(noise**2))

    if signal_rms <= 1e-12:
        raise ValueError("O audio de entrada parece silencioso; nao e possivel calcular SNR.")
    if noise_rms <= 1e-12:
        raise ValueError("Falha na geracao de ruido (RMS ~ 0).")

    target_noise_rms = signal_rms / (10 ** (snr_db / 20.0))
    scaled_noise = noise * (target_noise_rms / noise_rms)
    return signal + scaled_noise


def list_external_noise_files(path: Path) -> List[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Caminho de ruidos externos nao encontrado: {path}")

    if path.is_file():
        if path.suffix.lower() != ".wav":
            raise ValueError("Se --external-noise-path for arquivo, ele deve ser .wav")
        return [path]

    files = sorted(p for p in path.glob("*.wav") if p.is_file())
    if not files:
        raise ValueError(f"Nenhum .wav encontrado em: {path}")
    return files


def resample_if_needed(noise: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return noise

    g = math.gcd(source_sr, target_sr)
    up = target_sr // g
    down = source_sr // g
    return resample_poly(noise.astype(np.float64), up=up, down=down, axis=0)


def ensure_channel_count(noise: np.ndarray, channels: int) -> np.ndarray:
    if channels == 1:
        return noise if noise.ndim == 1 else np.mean(noise, axis=1)

    if noise.ndim == 1:
        return np.tile(noise[:, None], (1, channels))

    if noise.shape[1] == channels:
        return noise
    if noise.shape[1] > channels:
        return noise[:, :channels]

    reps = int(np.ceil(channels / noise.shape[1]))
    expanded = np.tile(noise, (1, reps))
    return expanded[:, :channels]


def fit_noise_length(
    noise: np.ndarray, target_samples: int, rng: np.random.Generator
) -> np.ndarray:
    if noise.shape[0] == target_samples:
        return noise
    if noise.shape[0] > target_samples:
        max_offset = noise.shape[0] - target_samples
        offset = int(rng.integers(0, max_offset + 1))
        return noise[offset : offset + target_samples]

    repeats = int(np.ceil(target_samples / noise.shape[0]))
    tiled = np.tile(noise, repeats) if noise.ndim == 1 else np.tile(noise, (repeats, 1))
    return tiled[:target_samples]


def load_external_noise(
    path: Path, target_shape: Sequence[int], target_sr: int, rng: np.random.Generator
) -> np.ndarray:
    noise_sr, noise_raw = wavfile.read(path)
    noise_float = to_float_signal(noise_raw)
    noise_float = resample_if_needed(noise_float, source_sr=noise_sr, target_sr=target_sr)

    channels = 1 if len(target_shape) == 1 else int(target_shape[1])
    target_samples = int(target_shape[0])

    noise_float = ensure_channel_count(noise_float, channels=channels)
    noise_float = fit_noise_length(noise_float, target_samples=target_samples, rng=rng)

    std = np.std(noise_float)
    if std <= 1e-12:
        raise ValueError(f"Ruido externo invalido (RMS ~ 0): {path}")

    return noise_float / std


def init_worker(
    signal: np.ndarray,
    dtype_str: str,
    sample_rate: int,
    active_threshold_db: float,
) -> None:
    global WORKER_SIGNAL
    global WORKER_DTYPE
    global WORKER_SAMPLE_RATE
    global WORKER_ACTIVE_THRESHOLD_DB

    WORKER_SIGNAL = signal
    WORKER_DTYPE = np.dtype(dtype_str)
    WORKER_SAMPLE_RATE = sample_rate
    WORKER_ACTIVE_THRESHOLD_DB = active_threshold_db


def render_job(job: Dict[str, Any]) -> Dict[str, Any]:
    if WORKER_SIGNAL is None or WORKER_DTYPE is None or WORKER_SAMPLE_RATE is None:
        raise RuntimeError("Worker nao inicializado corretamente.")

    rng = np.random.default_rng(job["seed"])
    signal = WORKER_SIGNAL

    mode = job["mode"]
    if mode == "synthetic":
        noise = generate_noise(signal.shape, noise_type=str(job["noise_type"]), rng=rng)
    elif mode == "external":
        ext_path = Path(str(job["external_noise_path"]))
        noise = load_external_noise(ext_path, signal.shape, WORKER_SAMPLE_RATE, rng)
    elif mode == "external_mixed":
        ext_path = Path(str(job["external_noise_path"]))
        ext_noise = load_external_noise(ext_path, signal.shape, WORKER_SAMPLE_RATE, rng)
        random_noise = generate_noise(signal.shape, noise_type=str(job["noise_type"]), rng=rng)
        noise = ext_noise + random_noise
    else:
        raise ValueError(f"Modo de job invalido: {mode}")

    noisy = apply_snr(
        signal,
        noise,
        float(job["snr_db"]),
        active_threshold_db=WORKER_ACTIVE_THRESHOLD_DB,
    )
    restored = from_float_signal(noisy, WORKER_DTYPE)

    out_path = Path(str(job["out_path"]))
    wavfile.write(out_path, WORKER_SAMPLE_RATE, restored)

    return {
        "file": out_path.name,
        "path": str(out_path),
        "mode": mode,
        "snr_db": float(job["snr_db"]),
        "noise_type": job.get("noise_type"),
        "external_noise": job.get("external_noise_name"),
    }


def snr_label(snr_db: float) -> str:
    return str(snr_db).replace(".", "p")


def write_metadata(rows: List[Dict[str, Any]], output_dir: Path, fmt: str) -> None:
    if fmt == "none":
        return

    rows_sorted = sorted(rows, key=lambda row: str(row["file"]))
    fieldnames = ["file", "path", "mode", "snr_db", "noise_type", "external_noise"]

    if fmt in ("csv", "both"):
        csv_path = output_dir / "metadata.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_sorted)

    if fmt in ("json", "both"):
        json_path = output_dir / "metadata.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(rows_sorted, f, indent=2, ensure_ascii=True)


def build_jobs(
    args: argparse.Namespace,
    external_files: List[Path],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    jobs: List[Dict[str, Any]] = []

    def next_seed() -> int:
        return int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))

    for noise_type in args.noise_types:
        for snr in args.snr_db:
            for i in range(args.copies_per_snr):
                name = f"{args.prefix}_{noise_type}_snr{snr_label(float(snr))}dB_{i + 1:03d}.wav"
                jobs.append(
                    {
                        "mode": "synthetic",
                        "noise_type": str(noise_type),
                        "snr_db": float(snr),
                        "out_path": str(output_dir / name),
                        "seed": next_seed(),
                    }
                )

    if args.external_noise_copies > 0:
        selected_external = list(
            rng.choice(external_files, size=args.external_noise_copies, replace=False)
        )
        for ext_path in selected_external:
            if args.external_full_sweep:
                for snr in args.snr_db:
                    for i in range(args.copies_per_snr):
                        name = (
                            f"{args.prefix}_external-{ext_path.stem}_"
                            f"snr{snr_label(float(snr))}dB_{i + 1:03d}.wav"
                        )
                        jobs.append(
                            {
                                "mode": "external",
                                "snr_db": float(snr),
                                "out_path": str(output_dir / name),
                                "seed": next_seed(),
                                "external_noise_path": str(ext_path),
                                "external_noise_name": ext_path.name,
                            }
                        )
            else:
                for i in range(args.copies_per_snr):
                    snr = float(rng.choice(args.snr_db))
                    name = (
                        f"{args.prefix}_external-{ext_path.stem}_"
                        f"snr{snr_label(snr)}dB_{i + 1:03d}.wav"
                    )
                    jobs.append(
                        {
                            "mode": "external",
                            "snr_db": snr,
                            "out_path": str(output_dir / name),
                            "seed": next_seed(),
                            "external_noise_path": str(ext_path),
                            "external_noise_name": ext_path.name,
                        }
                    )

    if args.external_mixed_copies > 0:
        selected_mixed = list(rng.choice(external_files, size=args.external_mixed_copies, replace=False))
        for ext_path in selected_mixed:
            if args.external_full_sweep:
                for snr in args.snr_db:
                    for noise_type in args.noise_types:
                        for i in range(args.copies_per_snr):
                            name = (
                                f"{args.prefix}_externalmix-{ext_path.stem}-{noise_type}_"
                                f"snr{snr_label(float(snr))}dB_{i + 1:03d}.wav"
                            )
                            jobs.append(
                                {
                                    "mode": "external_mixed",
                                    "noise_type": str(noise_type),
                                    "snr_db": float(snr),
                                    "out_path": str(output_dir / name),
                                    "seed": next_seed(),
                                    "external_noise_path": str(ext_path),
                                    "external_noise_name": ext_path.name,
                                }
                            )
            else:
                for i in range(args.copies_per_snr):
                    snr = float(rng.choice(args.snr_db))
                    noise_type = str(rng.choice(args.noise_types))
                    name = (
                        f"{args.prefix}_externalmix-{ext_path.stem}-{noise_type}_"
                        f"snr{snr_label(snr)}dB_{i + 1:03d}.wav"
                    )
                    jobs.append(
                        {
                            "mode": "external_mixed",
                            "noise_type": noise_type,
                            "snr_db": snr,
                            "out_path": str(output_dir / name),
                            "seed": next_seed(),
                            "external_noise_path": str(ext_path),
                            "external_noise_name": ext_path.name,
                        }
                    )

    return jobs


def resolve_external_copy_counts(
    args: argparse.Namespace,
    external_files: Sequence[Path],
) -> tuple[int, int]:
    total_external = len(external_files)

    if args.external_noise_copies is None:
        external_noise_copies = total_external if args.external_full_sweep else 0
    else:
        external_noise_copies = int(args.external_noise_copies)

    if args.external_mixed_copies is None:
        external_mixed_copies = total_external if args.external_full_sweep else 0
    else:
        external_mixed_copies = int(args.external_mixed_copies)

    return external_noise_copies, external_mixed_copies


def main() -> None:
    args = parse_args()

    if args.copies_per_snr < 1:
        raise ValueError("--copies-per-snr deve ser >= 1")
    if args.external_noise_copies is not None and args.external_noise_copies < 0:
        raise ValueError("--external-noise-copies deve ser >= 0")
    if args.external_mixed_copies is not None and args.external_mixed_copies < 0:
        raise ValueError("--external-mixed-copies deve ser >= 0")
    if args.active_threshold_db < 0:
        raise ValueError("--active-threshold-db deve ser >= 0")
    if args.workers < 1:
        raise ValueError("--workers deve ser >= 1")

    if not args.input_wav.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {args.input_wav}")

    sample_rate, original = wavfile.read(args.input_wav)
    float_signal = to_float_signal(original)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    external_files: List[Path] = []
    wants_external_noise = (
        args.external_full_sweep
        or (args.external_noise_copies is not None and args.external_noise_copies > 0)
        or (args.external_mixed_copies is not None and args.external_mixed_copies > 0)
    )
    if wants_external_noise:
        if args.external_noise_path is None:
            raise ValueError(
                "Para usar ruidos externos, informe --external-noise-path com arquivo ou diretorio."
            )
        external_files = list_external_noise_files(args.external_noise_path)

        external_noise_copies, external_mixed_copies = resolve_external_copy_counts(
            args,
            external_files,
        )
        args.external_noise_copies = external_noise_copies
        args.external_mixed_copies = external_mixed_copies

        if args.external_noise_copies > len(external_files):
            raise ValueError(
                "Quantidade insuficiente de ruidos externos para --external-noise-copies. "
                f"Solicitado: {args.external_noise_copies}, disponivel: {len(external_files)}"
            )
        if args.external_mixed_copies > len(external_files):
            raise ValueError(
                "Quantidade insuficiente de ruidos externos para --external-mixed-copies. "
                f"Solicitado: {args.external_mixed_copies}, disponivel: {len(external_files)}"
            )

    jobs = build_jobs(args, external_files, args.output_dir)

    metadata_rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(
            float_signal,
            original.dtype.str,
            sample_rate,
            args.active_threshold_db,
        ),
    ) as executor:
        with tqdm(total=len(jobs), unit="arquivo", desc="Gerando") as pbar:
            for row in executor.map(render_job, jobs):
                metadata_rows.append(row)
                pbar.update(1)

    write_metadata(metadata_rows, args.output_dir, args.metadata_format)

    print(f"Concluido: {len(metadata_rows)} arquivos gerados em '{args.output_dir}'.")


if __name__ == "__main__":
    # Sugestão: uv run scripts\test_gen\augment_wav_noise.py input.wav --external-noise-path noise --snr-db -10 -5 0 10 20 --copies-per-snr 2 --external-full-sweep
    # uv run scripts\test_gen\augment_wav_noise.py input.wav --external-noise-path noise --snr-db -10 -5 0 10 20 --copies-per-snr 2 --external-noise-copies 30 --external-mixed-copies 30
    main()
