from typing import List, Optional

import torch
from tqdm import tqdm

from bias_correction.data_utils import Data
from bias_correction.methods.base import BiasCorrection


class TCB(BiasCorrection):
    def __init__(
        self,
        n_tasks: int,
        n_cls: int,
        classes_per_task: int,
        device: str = "cuda",
        bias: float = 0.05,
    ):
        super().__init__(
            n_tasks=n_tasks,
            n_cls=n_cls,
            classes_per_task=classes_per_task,
            device=device,
        )
        per_task_bias = [bias * i for i in list(reversed(list(range(self.n_tasks))))]
        output_biases = []
        for i in range(self.n_tasks):
            output_biases.extend([per_task_bias[i]] * self.classes_per_task)
        self._bias = torch.tensor(per_task_bias)
        self._extended_bias = torch.tensor(output_biases).unsqueeze(0).unsqueeze(0)

    def fit_bias_correction(
        self, train_data: Data, test_data: Optional[List[Data]] = None
    ):
        pass

    def get_corrected_probabilities(self, logits):
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        pred_mask = torch.nn.functional.one_hot(preds, num_classes=logits.shape[-1])
        correction = pred_mask * self._extended_bias.to(
            logits.device, non_blocking=True
        )
        return probs + correction
