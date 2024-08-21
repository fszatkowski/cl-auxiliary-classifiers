from abc import ABC, abstractmethod
from typing import List, Optional

from torch import Tensor

from bias_correction.data_utils import Data


class BiasCorrection(ABC):
    def __init__(self, n_tasks: int, n_cls: int, classes_per_task: int, device: str):
        self.n_tasks = n_tasks
        self.n_cls = n_cls
        self.classes_per_task = classes_per_task
        self.device = device

    @abstractmethod
    def fit_bias_correction(
        self, train_data: Data, test_data: Optional[List[Data]] = None
    ):
        # Compute bias correction;
        # logits: torch.Tensor of shape (batch_size, num_ics, num_classes)
        # targets: torch.Tensor of shape (batch_size, )
        ...

    @abstractmethod
    def get_bias_matrix(self) -> Tensor:
        """Get the bias matrix of shape [n_tasks, n_cls]"""
        ...

    @abstractmethod
    def get_corrected_probabilities(self, logits):
        # Apply bias correction to logits and return the corrected probabilities
        # logits: torch.Tensor of shape (batch_size, num_ics, num_classes)
        ...
