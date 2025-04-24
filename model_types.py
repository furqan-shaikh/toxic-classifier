from dataclasses import dataclass
from typing import List, Any

@dataclass
class Checkpoint:
    name: str
    path: str

@dataclass
class AdamOptimizer:
    lr: float
    weight_decay: float
    amsgrad: bool

@dataclass
class RunArguments:
    batch_size: int
    shuffle: bool
    epoch: int
    accumulate_grad_batches: int

@dataclass
class TrainingData:
    training_path: str
    test_path: str = ""
    validation_path: str=""

@dataclass
class Model:
    model_type: str
    model: Any
    tokenizer: Any

@dataclass
class ConfigData:
    training_data: TrainingData
    model: Model
    run_arguments: RunArguments
    checkpoint: Checkpoint
    adam_optimizer: AdamOptimizer
    classes: List[str]

