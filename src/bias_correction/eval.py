from pathlib import Path
from typing import List

import torch
from torch import Tensor

from bias_correction.data_utils import Data
from bias_correction.methods.base import BiasCorrection


def evaluate(
    bias_correction: BiasCorrection,
    test_data: List[Data],
    exit_costs: Tensor,
    n_thresholds: int = 101,
):
    n_tasks = bias_correction.n_tasks
    n_cls = bias_correction.n_cls

    per_ic_acc = torch.zeros((n_tasks, n_cls))
    per_ic_avg_confidence = torch.zeros(
        (n_tasks, n_cls)
    )  # Average prediction confidence
    per_ic_correct_pred_confidence = torch.zeros(
        (n_tasks, n_cls)
    )  # Prediction confidence for correct cases
    per_ic_incorrect_pred_confidence = torch.zeros(
        (n_tasks, n_cls)
    )  # Prediction confidence for incorrect cases

    all_probs = []
    all_targets = []
    for task_id, data in enumerate(test_data):
        corrected_probs = bias_correction.get_corrected_probabilities(data.logits)
        targets = data.targets.unsqueeze(1)
        preds = corrected_probs.argmax(dim=-1)
        confidences = corrected_probs.max(dim=-1).values
        hits = preds == targets

        acc = hits.float().mean(dim=0)
        avg_confidence = confidences.mean(dim=0)
        corr_confidence = (confidences * hits.float()).sum(0) / hits.sum(0)
        incorr_confidence = (confidences * (~hits).float()).sum(0) / (~hits).sum(0)

        per_ic_acc[task_id] = acc
        per_ic_avg_confidence[task_id] = avg_confidence
        per_ic_correct_pred_confidence[task_id] = corr_confidence
        per_ic_incorrect_pred_confidence[task_id] = incorr_confidence

        all_probs.append(corrected_probs)
        all_targets.append(targets)

    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    n_thresholds = torch.linspace(0, 1, n_thresholds)

    acc_per_th, costs_per_th = evaluate_ee(
        all_probs, all_targets, exit_costs, n_thresholds
    )

    return {
        "acc_per_th": acc_per_th,
        "cost_per_th": costs_per_th,
        "auc": auc(acc_per_th, costs_per_th),
        "per_ic_acc": per_ic_acc,
        "per_ic_avg_confidence": per_ic_avg_confidence,
        "per_ic_correct_pred_confidence": per_ic_correct_pred_confidence,
        "per_ic_incorrect_pred_confidence": per_ic_incorrect_pred_confidence,
    }


def auc(accs: Tensor, costs: Tensor):
    # accs, costs: tensors of shape [N,]
    avg_accs = (accs[:-1] + accs[1:]) / 2
    costs = costs[1:] - costs[:-1]
    return (avg_accs * costs).sum()


def get_acc_for_budget(accs, costs, budget):
    accs, costs = accs.tolist(), costs.tolist()
    under_budget_vals = [c for c in costs if c <= budget]
    if len(under_budget_vals) == 0:
        return -1
    budget_val = max(under_budget_vals)
    budget_idx = costs.index(budget_val)
    budget_acc = accs[budget_idx]
    return budget_acc


def compute_metrics(accs: Tensor, costs: Tensor):
    output = {"auc": float(auc(accs, costs))}
    for budget in [0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 1.0]:
        output[f"acc_{budget}"] = get_acc_for_budget(accs, costs, budget)
    return output


def evaluate_ee(
    probs: Tensor,
    targets: Tensor,
    exit_costs: Tensor,
    thresholds: Tensor,
    return_max_conf_on_no_exit=True,
):
    # probs: tensor of shape [batch_size, n_ics, n_classes]
    # thresholds: tensor of shape [N_thresholds]
    # exit_costs: tensor of shape [n_ics]

    _, n_ics, n_classes = probs.shape
    max_ic_val = torch.tensor(n_ics - 1).to(probs.device)

    max_confidence_values, max_confidence_preds = torch.max(probs, dim=2)
    max_confidence_values = max_confidence_values.unsqueeze(
        dim=0
    )  # shape [1, batch_size, n_ics]
    max_confidence_preds = max_confidence_preds.unsqueeze(
        dim=0
    )  # shape [1, batch_size, n_ics]
    thresholds_tensor = thresholds.unsqueeze(dim=-1).unsqueeze(
        dim=-1
    )  # shape [N_thresholds, 1, 1]
    exit_costs_tensor = exit_costs.unsqueeze(dim=0).unsqueeze(
        dim=0
    )  # shape [1, 1, n_ics]

    # Compute exit masks
    th_satisfied = (
        max_confidence_values >= thresholds_tensor
    )  # shape [N_thresholds, batch_size, n_ics]
    exited = th_satisfied.sum(dim=2)  # shape [N_thresholds, batch_size]
    min_exit_idx = th_satisfied.float().argmax(
        dim=2
    )  # shape [N_thresholds, batch_size]

    # Set exit id to the exit idx if network exited, else return max IC idx
    exit_idx = torch.where(
        exited != 0, min_exit_idx, max_ic_val
    )  # shape [N_thresholds, batch_size]

    # Compute predictions for each sample
    # Get predictions for samples that exited
    per_ic_preds_exited = (
        torch.nn.functional.one_hot(min_exit_idx, num_classes=n_ics)
        * max_confidence_preds
    ).sum(
        dim=2
    )  # shape [N_thresholds, batch_size]

    if return_max_conf_on_no_exit:
        # Get predictions for samples that didn't exit
        max_conf_pred_values = torch.argmax(
            max_confidence_values, dim=2
        )  # shape [1, batch_size]
        max_conf_preds_mask = torch.nn.functional.one_hot(
            max_conf_pred_values, num_classes=n_ics
        )  # shape [1, batch_size, n_ics]
        max_conf_preds = (max_conf_preds_mask * max_confidence_preds).sum(
            dim=2
        )  # shape [1, batch_size]
        exit_output = torch.where(
            exited != 0, per_ic_preds_exited, max_conf_preds
        )  # shape [N_thresholds, batch_size]
    else:
        last_layer_preds = max_confidence_preds[:, :, -1]  # shape [1, batch_size]
        exit_output = torch.where(exited != 0, per_ic_preds_exited, last_layer_preds)

    # Compute cost for each sample
    exit_idx_mask = torch.nn.functional.one_hot(
        exit_idx, num_classes=n_ics
    )  # shape [N_thresholds, batch_size, n_ics]
    exit_costs_per_sample = (exit_costs_tensor * exit_idx_mask).sum(
        dim=2
    )  # shape [N_thresholds, batch_size, n_ics]

    targets_tensor = targets.squeeze().unsqueeze(dim=0)  # shape [1, batch_size]
    acc_per_th = (exit_output == targets_tensor).sum(dim=1) / targets_tensor.shape[
        1
    ]  # shape [N_thresholds]
    cost_per_th = exit_costs_per_sample.mean(dim=1)  # shape [N_thresholds]

    return acc_per_th, cost_per_th


def evaluate_ics(probs: Tensor, targets: Tensor, n_tasks: int, output_path: Path):
    _, n_ics, n_classes = probs.shape

    per_task_data = []
    task_size = targets.shape[0] // n_tasks
    for i in range(n_tasks):
        targets_task = targets[i * task_size : (i + 1) * task_size].unsqueeze(1)
        preds_task = probs[i * task_size : (i + 1) * task_size].argmax(dim=-1)
        acc_task = (preds_task == targets_task).float().sum(dim=0) / targets_task.shape[
            0
        ]
        per_task_data.append(acc_task)

    accs = torch.stack(per_task_data, dim=0)
    torch.save(accs, output_path)
