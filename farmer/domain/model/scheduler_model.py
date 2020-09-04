from dataclasses import dataclass
from dataclasses import field
from typing import List, Dict


@dataclass
class LRScheduler:
    scheduler_name: str = None
    cosine_decay: bool = False
    cosine_lr_max: int = 0.01
    cosine_lr_min: int = 0.001
    scheduler_base_lr: float = 0.001
    step_lr: bool = False
    multi_step_lr: bool = False
    exponential_lr: bool = False
    step_size: int = 20
    step_gamma: float = 0.5
    milestones: List[int] = field(default_factory=list)
    exp_gamma: float = 0.95