from dataclasses import dataclass
import os
from datetime import datetime
from dataclasses import field
from typing import List
from .config_model import Config
from .image_loader_model import ImageLoader
from .optuna_model import Optuna

@dataclass
class Trainer(Config, ImageLoader, Optuna):
    train_id: int = None
    training: bool = None
    epochs: int = None
    steps: int = None
    batch_size: int = None
    learning_rate: float = None
    optimizer: str = None
    augmentation: List[str] = field(default_factory=list)
    gpu: str = None
    nb_gpu: int = None
    multi_gpu: bool = None
    loss: str = None
    trained_model_path: str = None
    model_name: str = None
    backbone: str = None
    nb_train_data: int = 0
    nb_validation_data: int = 0
    nb_test_data: int = 0
    save_pred: bool = True
    segmentation_val_step: int = 3
    n_splits: int = 5

    def __post_init__(self):
        self.task = self.get_task()
        self.gpu = str(self.gpu)
        self.nb_gpu = len(self.gpu.split(",")) if self.gpu else 0
        self.multi_gpu = self.nb_gpu > 1
        if self.batch_size:
            self.batch_size *= self.nb_gpu if self.multi_gpu else 1
        if self.result_dir is None:
            self.result_dir = datetime.today().strftime("%Y%m%d_%H%M")
        self.result_path = os.path.join(self.root_dir, self.result_dir)
        self.info_path = os.path.join(self.result_path, self.info_dir)
        self.model_path = os.path.join(self.result_path, self.model_dir)
        self.learning_path = os.path.join(self.result_path, self.learning_dir)
        self.image_path = os.path.join(self.result_path, self.image_dir)
        self.class_names = self.get_class_names()
        self.nb_classes = len(self.class_names)
        self.height, self.width = self.get_image_shape()
        self.train_dirs = [str(train_dir) for train_dir in self.train_dirs]
        self.val_dirs = [str(val_dir) for val_dir in self.val_dirs]
        self.test_dirs = [str(test_dir) for test_dir in self.test_dirs]

        self.op_batch_size = type(self.batch_size) == list
        self.op_learning_rate = type(self.learning_rate) == list