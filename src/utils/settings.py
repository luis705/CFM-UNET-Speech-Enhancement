from pathlib import Path

from pydantic_settings import BaseSettings
import yaml


class DatasetConfig(BaseSettings):
    hf_repo: str = "JacobLinCool/VoiceBank-DEMAND-16k"
    cache_dir: str = "./cache"


class DataConfig(BaseSettings):
    batch_size: int
    num_workers: int
    input_channels: int
    output_channels: int
    val_split: float


class AudioConfig(BaseSettings):
    # Signal settings
    sample_rate: int
    audio_length_seconds: int
    # FFT settings
    fft_size: int
    hop_length: int
    window_size: int
    # Power law compression settings
    alpha: float


class OptimConfig(BaseSettings):
    lr: float
    weight_decay: float
    pct_start: float


class TrainingConfig(BaseSettings):
    early_stop_patience: int
    max_epochs: int
    precision: str
    accum_batches: int
    grad_clip: float
    validation_slow_metrics_pct: float
    histogram_logging_frequency: int
    val_interval: float | int

    optim: OptimConfig


class ModelConfig(BaseSettings):
    # Time embedding
    time_emb_dim: int
    # Input and output projection
    projection_dim: int

    # UNet
    n_blocks: int
    compression_factor: int
    channels_factor: int

    # Attention
    num_heads: int
    ff_mult: int
    n_groups: int

    # Checkpoints
    gradient_checkpointing: bool


class OdeConfig(BaseSettings):
    val_steps: int
    test_steps: int


class LoggingConfig(BaseSettings):
    log_dir: str
    n_audios: int


class Settings(BaseSettings):
    dataset: DatasetConfig
    data: DataConfig
    audio: AudioConfig
    model: ModelConfig
    ode: OdeConfig
    training: TrainingConfig
    logging: LoggingConfig


CFG_VALUE = None


def get_settings(config_path: str = "settings.yaml"):
    global CFG_VALUE
    if CFG_VALUE is None:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path) as f:
                config_dict = yaml.safe_load(f) or {}
            CFG_VALUE = Settings(**config_dict)
        else:
            CFG_VALUE = Settings()
    return CFG_VALUE
