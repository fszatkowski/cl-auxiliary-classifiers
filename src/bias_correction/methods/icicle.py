from typing import List, Optional

import torch
from tqdm import tqdm

from bias_correction.data_utils import Data
from bias_correction.methods.base import BiasCorrection


class ICICLE(BiasCorrection):
    def __init__(
        self,
        n_tasks: int,
        n_cls: int,
        classes_per_task: int,
        u: float = 0.1,
        eps: float = 0.01,
        batch_size: int = 1024,
        device: str = "cuda",
    ):
        """
        Bias correction method from ICICLE paper (https://arxiv.org/abs/2303.07811)

        :param u: Hyperparam 'u': the perecentage of predictions to switch during tuning the bias correction
        :param shared_correction: Whether to use the same bias correction for all ICs
        :param eps: The granularity of the bias correction searched during the optimization
        :param batch_size: The batch size used to find the bias correction values
        """
        super().__init__(
            n_tasks=n_tasks,
            n_cls=n_cls,
            classes_per_task=classes_per_task,
            device=device,
        )
        self.u = u
        self.eps = eps
        self.batch_size = batch_size

        self._bias = None
        self._expanded_bias = None

    def fit_bias_correction(
        self, train_data: Data, test_data: Optional[List[Data]] = None
    ):
        logits_train = train_data.logits
        assert len(logits_train.shape) == 3, (
            "Incorrect shape of logits"
            f" {logits_train.shape} != (batch_size, n_ics, n_classes)"
        )
        assert logits_train.shape[1] == self.n_cls, (
            "Incorrect number of classifiers in logits:"
            f" {logits_train.shape[1]} != {self.n_cls}"
        )
        assert logits_train.shape[2] == self.n_tasks * self.classes_per_task, (
            "Incorrect number of classes in logits:"
            f" {logits_train.shape[2]} != {self.n_tasks * self.classes_per_task}"
        )

        n_samples = logits_train.shape[0]
        last_task_logits = logits_train[:, :, -self.classes_per_task :]

        icicle_biases = torch.zeros((self.n_cls, self.n_tasks), device=self.device)
        ic_task_ids = [
            (ic_idx, task_idx)
            for ic_idx in range(self.n_cls)
            for task_idx in range(self.n_tasks - 1)
        ]
        for ic_idx, task_idx in tqdm(
            ic_task_ids, desc=f"Optimizing ICICLE logits (u={self.u})..."
        ):
            task_logits = logits_train[
                :,
                ic_idx,
                task_idx
                * self.classes_per_task : (task_idx + 1)
                * self.classes_per_task,
            ].unsqueeze(0)
            last_task_ic_logits = last_task_logits[:, ic_idx].unsqueeze(0)

            ic_task_bias = self.eps * torch.arange(
                0, self.batch_size, 1.0, device=self.device
            ).unsqueeze(1).unsqueeze(1)
            total_task_pred_cnts = (
                (task_logits + ic_task_bias).max(dim=-1).values
                > last_task_ic_logits.max(dim=-1).values
            ).sum(dim=1)
            total_task_pred_cnts_ratios = total_task_pred_cnts > self.u * n_samples
            while not any(total_task_pred_cnts_ratios):
                ic_task_bias += self.eps * self.batch_size
                total_task_pred_cnts = (
                    (task_logits + ic_task_bias).max(dim=-1).values
                    > last_task_ic_logits.max(dim=-1).values
                ).sum(dim=1)
                total_task_pred_cnts_ratios = total_task_pred_cnts > self.u * n_samples

            ic_task_bias = ic_task_bias[total_task_pred_cnts_ratios].min()
            icicle_biases[ic_idx, task_idx] = ic_task_bias

        self._bias = icicle_biases.to("cpu")
        self._expanded_bias = (
            torch.repeat_interleave(icicle_biases, self.classes_per_task, dim=1)
            .unsqueeze(0)
            .to("cpu")
        )

    def get_corrected_probabilities(self, logits):
        assert (
            self._expanded_bias is not None
        ), "Call 'fit_bias_correction' before calling 'get_corrected_probabilities'"

        logits = logits + self._expanded_bias.to(logits.device, non_blocking=True)
        return torch.softmax(logits, dim=-1)
