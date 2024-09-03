from typing import Dict, List, Optional

import hyperopt
import numpy as np
import torch
from hyperopt import STATUS_OK, Trials, fmin, hp
from torch import Tensor
from tqdm import tqdm

from bias_correction.data_utils import Data
from bias_correction.methods.base import BiasCorrection


class IdealLCT(BiasCorrection):
    def __init__(
        self,
        n_tasks: int,
        n_cls: int,
        classes_per_task: int,
        seed: int = 0,
        n_steps=100,
        lr=0.01,
        device: str = "cuda",
    ):
        super().__init__(
            n_tasks=n_tasks,
            n_cls=n_cls,
            classes_per_task=classes_per_task,
            device=device,
        )

        self.n_steps = n_steps
        self.lr = lr
        self.seed = seed

        self._bias = None
        self._expanded_bias = None

    def fit_bias_correction(
        self, train_data: Data, test_data: Optional[List[Data]] = None
    ):
        assert test_data is not None
        logits_test = torch.cat([data.logits.to(self.device) for data in test_data], dim=0)

        preds_masks = torch.nn.functional.one_hot(
            logits_test.argmax(dim=-1), num_classes=logits_test.shape[-1]
        )
        preds_masks = 1 - preds_masks.float()
        preds_masks[preds_masks == 0] = float("-inf")
        per_task_logits_train = torch.stack(
            torch.split(logits_test * preds_masks, self.classes_per_task, dim=-1),
            dim=2,
        )  # (bs, n_cls, n_tasks, classes_per_task)
        max_ood_logits = per_task_logits_train.max(dim=-1)[0]

        bias = torch.nn.Parameter(
            torch.zeros((1, self.n_cls, self.n_tasks), device=self.device)
        )
        optimizer = torch.optim.AdamW([bias], lr=self.lr)
        pbar = tqdm(
            range(self.n_steps), desc="Optimizing LCT bias correction through SGD..."
        )

        # TODO move to GPU

        for _ in pbar:
            optimizer.zero_grad()
            adapted_logits = max_ood_logits + bias
            mean_adapted_logits = adapted_logits.mean(dim=(0, 2))
            loss = adapted_logits - mean_adapted_logits.unsqueeze(0).unsqueeze(-1)
            loss = (loss**2).mean()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})

        self._bias = bias.detach().to("cpu")
        self._expanded_bias = torch.repeat_interleave(
            self._bias, self.classes_per_task, dim=2
        ).to("cpu")

    def get_bias_matrix(self) -> Tensor:
        assert (
            self._bias is not None
        ), "Call 'fit_bias_correction' before calling 'get_bias_matrix'"
        bias = self._bias.squeeze(0)
        assert bias.shape == (self.n_cls, self.n_tasks), (
            "Incorrect shape of bias" f" {bias.shape} != {(self.n_cls, self.n_tasks)}"
        )
        return bias

    def get_corrected_probabilities(self, logits):
        assert (
            self._expanded_bias is not None
        ), "Call 'fit_bias_correction' before calling 'get_corrected_probabilities'"

        logits = logits + self._expanded_bias.to(logits.device, non_blocking=True)
        return torch.softmax(logits, dim=-1)
