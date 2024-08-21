from typing import List, Optional

import torch
from torch import Tensor
from tqdm import tqdm

from bias_correction.data_utils import Data
from bias_correction.methods.base import BiasCorrection


class TaskWiseTemperatureCorrection(BiasCorrection):
    def __init__(
        self,
        n_tasks: int,
        n_cls: int,
        classes_per_task: int,
        device: str = "cuda",
        base: float = 1.0,
        logit_adapter: Optional[BiasCorrection] = None,
        per_task_delta: float = 0.05,
    ):
        super().__init__(
            n_tasks=n_tasks,
            n_cls=n_cls,
            classes_per_task=classes_per_task,
            device=device,
        )
        self.logit_adapter = logit_adapter

        per_task_temp = [
            base + per_task_delta * i for i in list(reversed(list(range(self.n_tasks))))
        ]
        self._bias = torch.tensor(self.n_cls * [per_task_temp], device=self.device)
        per_task_temp_extended = []
        for i in range(self.n_tasks):
            per_task_temp_extended.extend([per_task_temp[i]] * self.classes_per_task)
        self._extended_bias = torch.tensor(per_task_temp_extended)

    def fit_bias_correction(
        self, train_data: Data, test_data: Optional[List[Data]] = None
    ):
        pass

    def get_bias_matrix(self) -> Tensor:
        assert self._bias.shape == (self.n_cls, self.n_tasks), (
            f"Expected bias matrix of shape ({self.n_cls}, {self.n_tasks}), "
            f"got {self._bias.shape}"
        )
        return self._bias

    def get_corrected_probabilities(self, logits):
        if self.logit_adapter is not None:
            bias_matrix = self.logit_adapter.get_bias_matrix()
            logits = logits + bias_matrix.unsqueeze(0).repeat_interleave(
                self.classes_per_task, dim=2
            ).to(self.device)
        preds = torch.argmax(logits, dim=-1)
        pred_mask = torch.nn.functional.one_hot(preds, num_classes=logits.shape[-1])
        temp_mask = pred_mask * self._extended_bias.to(logits.device, non_blocking=True)
        per_pred_temp = temp_mask.sum(dim=-1).unsqueeze(-1)
        return torch.softmax(logits * per_pred_temp, dim=-1)
