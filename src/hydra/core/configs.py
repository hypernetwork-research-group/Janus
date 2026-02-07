from dataclasses import dataclass
from pathlib import Path
from typing import Union, List

@dataclass
class DataModuleConfig:
    p: float
    q: float
    alpha: float
    walk_length: int
    samples_per_hyperedge: int
    data_dir: Path
    retain_lcc: bool
    train_split: str
    val_split: str
    predict_split: str
    val_size: Union[float, int, None]

@dataclass
class HuggingFaceDatasetsConfig:
    dataset_name: str
    cache_dir: Path

@dataclass
class DataLoaderConfig:
    pin_memory: bool
    num_workers: int | None
    persistent_workers: bool
    batch_size: int | None

@dataclass
class TrainerConfig:
    max_epochs: int
    accumulate_grad_batches: int

@dataclass(frozen=True, slots=True)
class ModelSizeConfig:
    hidden_dim: int
    num_layers: int
    heads: int

@dataclass
class OptimizerConfig:
    learning_rate: float | None
    weight_decay: float | None

@dataclass
class EarlyStoppingConfig:
    patience: int
