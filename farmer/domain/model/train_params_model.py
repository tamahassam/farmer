from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TrainParams:
    # 最適化のために探索するパラメータ
    model_name: str = None
    backbone: str = None
    activation: str = "softmax"
    loss: Dict[str, dict] = field(default_factory=dict)
    classification_class_weight: Dict[str, float] = field(default_factory=dict)
    batch_size: int = None
    weights_info: Dict[str, str] = field(default_factory=dict)
    learning_rate: float = None
    optimizer: str = None
    augmentation: Dict[str, int] = field(default_factory=dict)
    opt_decay: float = 0.001
    scheduler: dict = None
