from typing import Dict, List, Optional

import hyperopt
import numpy as np
import torch
from hyperopt import STATUS_OK, Trials, fmin, hp
from torch import Tensor

from bias_correction.data_utils import Data
from bias_correction.eval import auc, evaluate_ee
from bias_correction.methods.base import BiasCorrection


class HyperoptSequentialTemperatureCorrection(BiasCorrection):
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
        device: str = "cuda",
        exit_costs: Tensor = torch.Tensor(),
        thresholds: int = 101,
        logit_adapter: Optional[BiasCorrection] = None,
        seed: int = 0,
    ):
        super().__init__(
            n_tasks=n_tasks,
            n_cls=n_cls,
            classes_per_task=classes_per_task,
            device=device,
        )
        self.logit_adapter = logit_adapter
        self.exit_costs = exit_costs
        self.thresholds = torch.linspace(0, 1, thresholds)

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
                f"t_{task_idx}": hp.normal(f"t_{task_idx}", mu=hp_mu, sigma=hp_sigma)
                for task_idx in range(self.n_tasks)
            }
        elif hp_space == "uniform":
            hp_init = {
                f"t_{task_idx}": hp.uniform(f"t_{task_idx}", low=hp_min, high=hp_max)
                for task_idx in range(self.n_tasks)
            }
        else:
            raise NotImplementedError()
        self.hp_init = hp_init

        self._bias = None
        self._extended_bias = None

    def fit_bias_correction(
        self, train_data: Data, test_data: Optional[List[Data]] = None
    ):
        assert (
            test_data is not None
        ), "Test data must be provided for hyperopt temperature correction"
        self.exit_costs = self.exit_costs.to(self.device, non_blocking=True)
        self.thresholds = self.thresholds.to(self.device, non_blocking=True)
        test_logits = torch.cat([t.logits for t in test_data], dim=0).to(
            self.device, non_blocking=True
        )
        test_targets = torch.cat([t.targets for t in test_data], dim=0).to(
            self.device, non_blocking=True
        )
        if self.logit_adapter is not None:
            bias_matrix = self.logit_adapter.get_bias_matrix()
            logits = test_logits + bias_matrix.unsqueeze(0).repeat_interleave(
                self.classes_per_task, dim=2
            ).to(self.device)
        else:
            logits = test_logits

        bias = torch.zeros((self.n_cls, self.n_tasks), device=self.device)
        for cls_idx in range(self.n_cls):

            def loss_fn(bias_dict):
                bias_cp = bias.clone()
                ic_bias = decode_bias_dict(bias_dict, self.n_tasks, self.device)
                bias_cp[cls_idx] = ic_bias
                bias_cp = (
                    bias_cp.unsqueeze(0)
                    .repeat_interleave(self.classes_per_task, dim=2)
                    .to(self.device, non_blocking=True)
                )
                preds = test_logits.argmax(dim=-1)
                pred_mask = torch.nn.functional.one_hot(
                    preds, num_classes=logits.shape[-1]
                )
                temperatures = (pred_mask * bias_cp).sum(dim=-1).unsqueeze(-1)
                corrected_preds = torch.nn.functional.softmax(
                    test_logits / temperatures, dim=-1
                )
                accs, costs = evaluate_ee(
                    corrected_preds,
                    test_targets,
                    self.exit_costs,
                    self.thresholds,
                    return_max_conf_on_no_exit=True,
                )
                area_under_curve = auc(accs, costs)
                loss = area_under_curve * (costs[-1] - costs[0])
                return 1 / float(loss)

            trials = Trials()
            best_coefficients = fmin(
                fn=loss_fn,
                space=self.hp_init,
                algo=self.algo,
                max_evals=self.max_iters,
                trials=trials,
                rstate=np.random.default_rng(self.seed),
            )
            best_ic_bias = decode_bias_dict(
                best_coefficients, self.n_tasks, self.device
            )
            bias[cls_idx] = best_ic_bias

        self.exit_costs = self.exit_costs.to("cpu")
        self.thresholds = self.thresholds.to("cpu")
        self._bias = bias.to("cpu")
        self._extended_bias = (
            self._bias.unsqueeze(0)
            .repeat_interleave(self.classes_per_task, dim=2)
            .to("cpu")
        )

    def get_bias_matrix(self) -> Tensor:
        assert self._bias.shape == (self.n_cls, self.n_tasks), (
            f"Expected bias matrix of shape ({self.n_cls}, {self.n_tasks}), "
            f"got {self._bias.shape}"
        )
        return self._bias

    def get_corrected_probabilities(self, logits):
        if self.logit_adapter is not None:
            bias_matrix = self.logit_adapter.get_bias_matrix().to(
                logits.device, non_blocking=True
            )
            logits = logits + bias_matrix.unsqueeze(0).repeat_interleave(
                self.classes_per_task, dim=2
            )
        preds = torch.argmax(logits, dim=-1)
        pred_mask = torch.nn.functional.one_hot(preds, num_classes=logits.shape[-1])
        temp_mask = pred_mask * self._extended_bias.to(logits.device, non_blocking=True)
        per_pred_temp = temp_mask.sum(dim=-1).unsqueeze(-1)
        return torch.softmax(logits / per_pred_temp, dim=-1)


def decode_bias_dict(bias_dict: Dict, n_tasks: int, device: str) -> Tensor:
    bias = torch.zeros(1, n_tasks)

    for task_idx in range(n_tasks):
        bias[0, task_idx] = bias_dict[f"t_{task_idx}"]
    return bias.to(device, non_blocking=True)
