"""Pydantic Settings - single source of truth for all configuration."""

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import yaml

# ── Sub-models ───────────────────────────────────────────────────────────────


class AudioConfig(BaseModel):
    sample_rate: int = 16_000
    n_fft: int = 512
    hop_length: int = 128
    win_length: int = 512
    window: str = "hann"


class DataConfig(BaseModel):
    hf_dataset: str = "JacobLinCool/VoiceBank-DEMAND-16k"
    hf_cache_dir: str | None = None
    hf_trust_remote_code: bool = False
    clean_column: str = "clean"
    noisy_column: str = "noisy"
    filename_column: str = "id"
    train_split: str = "train"
    val_split: str | None = None
    test_split: str = "test"
    val_ratio: float = 0.1
    val_split_from_test: float | None = None
    segment_length: int = 32_000
    num_workers: int = 8
    batch_size: int = 32
    prefetch_factor: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True


# ── VelocityNet sub-configs ──────────────────────────────────────────────────


class LowBandConfig(BaseModel):
    num_layers: int = 4
    num_heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.1


class HighBandConfig(BaseModel):
    num_layers: int = 4
    dilations: list[int] = [1, 2, 4, 8]
    dropout: float = 0.1


class BottleneckConfig(BaseModel):
    num_heads: int = 4
    dropout: float = 0.1


class DecoderConfig(BaseModel):
    num_layers: int = 2


class VelocityNetConfig(BaseModel):
    hidden_dim: int = 64
    f_split: int = 128
    time_emb_dim: int = 128
    low_band: LowBandConfig = Field(default_factory=LowBandConfig)
    high_band: HighBandConfig = Field(default_factory=HighBandConfig)
    bottleneck: BottleneckConfig = Field(default_factory=BottleneckConfig)
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)


# ── Model config ─────────────────────────────────────────────────────────────


class ModelConfig(BaseModel):
    save_onnx: bool = False
    gradient_checkpointing: bool = False
    velocity_net: VelocityNetConfig = Field(default_factory=VelocityNetConfig)


class ODESolverConfig(BaseModel):
    solver: Literal["euler", "midpoint", "dopri5"] = "euler"
    num_steps: int = 10
    atol: float = 1e-5
    rtol: float = 1e-5


class ODEConfig(BaseModel):
    val: ODESolverConfig = Field(default_factory=ODESolverConfig)
    test: ODESolverConfig = Field(
        default_factory=lambda: ODESolverConfig(solver="dopri5", num_steps=50),
    )


class WarmupSchedule(BaseModel):
    lambda_start: float
    lambda_end: float
    e0: float = 0.0
    k: float = 0.5


class MRSTFTLossConfig(WarmupSchedule):
    fft_sizes: list[int] = [512, 1024, 2048]
    hop_sizes: list[int] = [128, 256, 512]
    win_sizes: list[int] = [512, 1024, 2048]


class CompressedComplexLossConfig(WarmupSchedule):
    compression: float = 0.3


class LossConfig(BaseModel):
    curriculum_cfm_only_epochs: int = 50
    temporal_decay_gamma: float = 3.0
    cfm: WarmupSchedule = WarmupSchedule(lambda_start=1.0, lambda_end=2.0, e0=0, k=0.005)
    multi_resolution_stft: MRSTFTLossConfig = MRSTFTLossConfig(
        lambda_start=0.1,
        lambda_end=1.0,
        e0=50,
        k=0.5,
    )
    compressed_complex: CompressedComplexLossConfig = CompressedComplexLossConfig(
        lambda_start=0.01,
        lambda_end=0.5,
        e0=55,
        k=0.001,
    )


class TrainingConfig(BaseModel):
    max_epochs: int = 200
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    lr_scheduler: Literal["cosine_annealing", "cosine_with_warmup", "reduce_on_plateau"] = "cosine_annealing"
    lr_min: float = 1e-6
    warmup_steps: int = 1000
    lr_patience: int = 10
    lr_factor: float = 0.5
    gradient_clip_val: float = 5.0
    accumulate_grad_batches: int = 1
    precision: (
        Literal[
            "transformer-engine",
            "transformer-engine-float16",
            "16-true",
            "16-mixed",
            "bf16-true",
            "bf16-mixed",
            "32-true",
            "64-true",
        ]
        | Literal["64", "32", "16", "bf16"]
    ) = "16-mixed"
    ema_decay: float = 0.999
    early_stopping_patience: int = 30
    early_stopping_metric: str = "val_loss"


class ValidationConfig(BaseModel):
    every_n_steps: int | float = 1000
    audio_log_samples: int = 4


class TestConfig(BaseModel):
    save_audio_ratio: float = 0.0
    save_dir: str = "test_results/samples"


class InferenceConfig(BaseModel):
    chunk_size: int = 32_000
    overlap: float = 0.5
    max_batch_chunks: int = 16


class CheckpointConfig(BaseModel):
    dirpath: str = "checkpoints"
    monitor: str = "val_loss"
    mode: str = "max"
    save_top_k: int = 3
    save_last: bool = True


class LoggingConfig(BaseModel):
    log_dir: str = "tb_logs"
    name: str = "cfm-se"


class MetricsConfig(BaseModel):
    SI_SDR: bool = True
    ESTOI: bool = True
    PESQ: bool = True
    DNSMOS: bool = True


# ── Root Settings ────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    audio: AudioConfig = Field()
    data: DataConfig = Field()
    model: ModelConfig = Field()
    ode: ODEConfig = Field()
    loss: LossConfig = Field()
    training: TrainingConfig = Field()
    validation: ValidationConfig = Field()
    checkpoint: CheckpointConfig = Field()
    test: TestConfig = Field(default_factory=TestConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    logging: LoggingConfig = Field()
    metrics: MetricsConfig = Field()

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> Self:
        path = Path(path)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return cls(**data)
        raise ValueError("Config file not found")
