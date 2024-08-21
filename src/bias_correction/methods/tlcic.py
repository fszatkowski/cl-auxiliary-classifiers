from typing import Dict, List, Optional

import hyperopt
import numpy as np
import torch
from hyperopt import STATUS_OK, Trials, fmin, hp
from torch import Tensor
from tqdm import tqdm

from bias_correction.data_utils import Data
from bias_correction.methods.base import BiasCorrection


class TLCIC(BiasCorrection):
    def __init__(
        self,
        n_tasks: int,
        n_cls: int,
        classes_per_task: int,
        algorithm: str = "tpe",
        max_iters: int = 1000,
        hp_space: str = "normal",
        hp_mu: Optional[float] = None,
        hp_sigma: Optional[float] = None,
        hp_min: Optional[float] = None,
        hp_max: Optional[float] = None,
        seed: int = 0,
        device: str = "cuda",
    ):
        """
        TLC bias correction

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

        self.max_iters = max_iters
        self.seed = seed

        if algorithm == "tpe":
            algo = hyperopt.tpe.suggest
        elif algorithm == "random":
            algo = hyperopt.rand.suggest
        else:
            raise NotImplementedError()
        self.algo = algo

        if hp_space == "normal":
            hp_init = {
                "a": hp.normal("a", mu=hp_mu, sigma=hp_sigma),
                "b": hp.normal("b", mu=hp_mu, sigma=hp_sigma),
            }
        elif hp_space == "uniform":
            hp_init = {
                "a": hp.uniform("a", low=hp_min, high=hp_max),
                "b": hp.uniform("b", low=hp_min, high=hp_max),
            }
        else:
            raise NotImplementedError()
        self.hp_init = hp_init

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
        logits = train_data.logits
        ic_biases = []

        for ic_idx in range(self.n_cls):

            def loss_fn(bias_dict):
                bias = decode_bias_dict(bias_dict, self.n_tasks, self.device)
                bias = expand_bias(bias, self.classes_per_task).unsqueeze(dim=0)
                adapted_logits = logits[:, ic_idx] + bias
                preds = torch.argmax(adapted_logits, dim=-1)  # [batch_size, n_ics]
                pred_mask = torch.nn.functional.one_hot(
                    preds, num_classes=logits.shape[-1]
                )  # [batch_size, n_classes]
                masked_logits = torch.where(
                    pred_mask == 1, adapted_logits * -torch.inf, adapted_logits
                )  # [batch_size, n_classes]
                per_task_preds = torch.stack(
                    torch.split(masked_logits, self.classes_per_task, dim=1), dim=-1
                )  # [batch_size, n_classes_per_task, n_tasks]
                max_ood_logits = torch.max(
                    per_task_preds, dim=-2
                ).values  # [batch_size, n_tasks]
                mean_max_ood_logits = max_ood_logits.mean(dim=-1).unsqueeze(
                    1
                )  # [batch_size, 1, 1]
                ood_mse = (max_ood_logits - mean_max_ood_logits) ** 2
                return {"loss": float(torch.mean(ood_mse)), "status": STATUS_OK}

            trials = Trials()
            best_coefficients = fmin(
                fn=loss_fn,
                space=self.hp_init,
                algo=self.algo,
                max_evals=self.max_iters,
                trials=trials,
                rstate=np.random.default_rng(self.seed),
            )
            best_bias = decode_bias_dict(best_coefficients, self.n_tasks, self.device)
            ic_biases.append(best_bias)

        self._bias = torch.stack(ic_biases, dim=0).to("cpu")
        self._expanded_bias = (
            torch.repeat_interleave(self._bias, self.classes_per_task, dim=1)
            .unsqueeze(0)
            .to("cpu")
        )

    def get_corrected_probabilities(self, logits):
        assert (
            self._expanded_bias is not None
        ), "Call 'fit_bias_correction' before calling 'get_corrected_probabilities'"

        logits = logits + self._expanded_bias.to(logits.device, non_blocking=True)
        return torch.softmax(logits, dim=-1)

    def get_bias_matrix(self) -> Tensor:
        assert (
            self._bias is not None
        ), "Call 'fit_bias_correction' before calling 'get_bias_matrix'"
        assert self._bias.shape == (self.n_cls, self.n_tasks), (
            "Incorrect shape of bias"
            f" {self._bias.shape} != {(self.n_cls, self.n_tasks)}"
        )
        return self._bias


def decode_bias_dict(bias_dict: Dict, n_tasks: int, device: str) -> Tensor:
    a = bias_dict["a"]
    b = bias_dict["b"]

    dists = [n_tasks - 1 - i for i in range(n_tasks)]
    dists = torch.tensor(dists)
    b = torch.ones_like(dists) * b
    b[-1] = 0
    bias = a * dists + b

    return bias.to(device, non_blocking=True)


def expand_bias(bias: Tensor, classes_per_task: int) -> Tensor:
    return torch.repeat_interleave(bias, classes_per_task, dim=0)
