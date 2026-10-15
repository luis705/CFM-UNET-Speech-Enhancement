from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings
import yaml


class DatasetConfig(BaseSettings):
    hf_repo: str = "JacobLinCool/VoiceBank-DEMAND-16k"
    cache_dir: str = "./cache"


class DataConfig(BaseSettings):
    batch_size: int
    num_workers: int
    val_split: float
    augment: bool = False
    augment_pct: float = 0


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


class MRSTFTConfig(BaseSettings):
    fft_sizes: list[int] = [512, 1024, 2048]
    hop_sizes: list[int] = [120, 240, 480]
    win_sizes: list[int] = [600, 1200, 2400]
    lambda_max: float = 0.1
    start_epoch: int = 2
    end_epoch: int = 12


class CFMLossConfig(BaseSettings):
    lambda_value: float
    snr_min_val: float


class LossConfig(BaseSettings):
    cfm: CFMLossConfig
    mrstft: MRSTFTConfig

class TestConfig(BaseSettings):
    save_dir: str
    trajectory_n_audios: int = Field(default=4, ge=0)


class TrainingConfig(BaseSettings):
    early_stop_patience: int
    early_stop_min_epochs: int
    max_epochs: int
    precision: str
    accum_batches: int
    grad_clip: float
    validation_slow_metrics_pct: float
    val_interval: float | int

    optim: OptimConfig
    losses: LossConfig


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
    histogram_logging_frequency: int


class EvaluationConfig(BaseSettings):
    val_slow_metrics_percent: float = 0.0
    test_slow_metrics_percent: float = 1.0
    spk_sim_model_id: str = "microsoft/wavlm-base-plus-sv"
    spk_sim_device: str = "cpu"
    utmos_hub_repo: str = "tarepan/SpeechMOS:v1.2.0"
    utmos_model: str = "utmos22_strong"
    utmos_device: str = "cpu"


class Settings(BaseSettings):
    dataset: DatasetConfig
    data: DataConfig
    audio: AudioConfig
    model: ModelConfig
    ode: OdeConfig
    training: TrainingConfig
    logging: LoggingConfig
    evaluation: EvaluationConfig = EvaluationConfig()
    test: TestConfig


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
