from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class DataConfig:
    train_csv: str
    valid_csv: str
    max_train_stories: int
    max_valid_stories: int
    max_len: int
    token_cache_dir: str


@dataclass
class ModelConfig:
    vocab_name: str
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_dim: int


@dataclass
class TrainConfig:
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    warmup_frac: float
    grad_clip_norm: float
    seed: int


@dataclass
class OutputConfig:
    checkpoint_dir: str
    metrics_dir: str


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    output: OutputConfig


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
        output=OutputConfig(**raw["output"]),
    )