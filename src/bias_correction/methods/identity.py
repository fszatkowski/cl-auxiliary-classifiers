from typing import List, Optional

import torch
from tqdm import tqdm

from bias_correction.data_utils import Data
from bias_correction.methods.base import BiasCorrection


class Identity(BiasCorrection):
    def __init__(
        self,
        n_tasks: int,
        n_cls: int,
        classes_per_task: int,
        device: str = "cuda",
    ):
        super().__init__(
            n_tasks=n_tasks,
            n_cls=n_cls,
            classes_per_task=classes_per_task,
            device=device,
        )

    def fit_bias_correction(
        self, train_data: Data, test_data: Optional[List[Data]] = None
    ):
        pass

    def get_corrected_probabilities(self, logits):
        return torch.softmax(logits, dim=-1)

    def get_bias_matrix(self):
        return torch.zeros((self.n_cls, self.n_tasks))
