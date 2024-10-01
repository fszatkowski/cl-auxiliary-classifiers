from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from plotting.utils import decode_path

sns.set_style("whitegrid")


def compute_scores(
    results_dir: Path, methods: List[str], device: str = "cpu"
) -> List[Dict]:
    setting, method, ee_config, seed = decode_path(results_dir.parent)

    output_dirs = results_dir.parent / "outputs"
    n_tasks = len(list(output_dirs.glob("after_*")))
    last_task_id = n_tasks - 1
    last_task_outputs_dir = output_dirs / f"after_{last_task_id}"
    last_task_batches = list(last_task_outputs_dir.rglob("*.pt"))

    outputs = []
    targets = []
    for batch_path in last_task_batches:
        batch = torch.load(batch_path, map_location=device)
        logits = batch["logits"]
        logits = torch.stack(
            [torch.cat(task_logits, dim=-1) for task_logits in logits], dim=1
        )
        outputs.append(logits)
        targets.append(batch["targets"])

    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)

    output_dicts = []
    base_output = {
        "setting": setting,
        "method": method,
        "ee_config": ee_config,
        "seed": seed,
    }
    for selection_method in methods:
        output_dict = deepcopy(base_output)
        if selection_method == "last":
            preds = outputs[:, -1, :].argmax(dim=-1)
            hits = (preds == targets).float()
        elif selection_method == "max_conf":
            max_conf = outputs.softmax(dim=-1).max(dim=-1).values
            ac_idx = max_conf.argmax(dim=-1)
            ac_mask = torch.nn.functional.one_hot(ac_idx, outputs.shape[1]).unsqueeze(
                -1
            )
            preds = (ac_mask * outputs).sum(dim=1).argmax(dim=-1)
            hits = (preds == targets).float()
        elif selection_method == "min_entropy":
            probs = outputs.softmax(dim=-1)
            log_probs = torch.log(probs + 1e-9)
            entropy = -torch.sum(probs * log_probs, dim=-1)
            ac_idx = entropy.argmin(dim=-1)
            ac_mask = torch.nn.functional.one_hot(ac_idx, outputs.shape[1]).unsqueeze(
                -1
            )
            preds = (ac_mask * outputs).sum(dim=1).argmax(dim=-1)
            hits = (preds == targets).float()
        elif selection_method == "wavg_compute":
            probs = outputs.softmax(dim=-1)
            weights = (
                torch.tensor([0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05])
                .unsqueeze(0)
                .unsqueeze(2)
            )
            wavg_probs = (probs * weights).sum(dim=1) / weights.sum(dim=1)
            preds = wavg_probs.argmax(dim=-1)
            hits = (preds == targets).float()
        elif selection_method == "wavg_conf":
            probs = outputs.softmax(dim=-1)
            weights = probs.max(dim=-1).values.unsqueeze(-1)
            wavg_probs = (probs * weights).sum(dim=1) / weights.sum(dim=1)
            preds = wavg_probs.argmax(dim=-1)
            hits = (preds == targets).float()
        elif selection_method == "avg":
            probs = outputs.softmax(dim=-1)
            avg_probs = probs.mean(dim=1)
            preds = avg_probs.argmax(dim=-1)
            hits = (preds == targets).float()
        elif selection_method.startswith("first_"):
            th = float(selection_method.replace("first_", "").split("_")[0])
            max_conf = outputs.softmax(dim=-1).max(dim=-1).values
            above_th = max_conf >= th
            would_return = above_th.sum(dim=-1) > 0
            exit_idx = above_th.float().argmax(dim=-1)

            if selection_method.endswith("_or_last"):
                exit_indices = torch.where(would_return, exit_idx, outputs.shape[1] - 1)
            elif selection_method.endswith("_or_max_conf"):
                exit_indices = torch.where(
                    would_return, exit_idx, max_conf.argmax(dim=-1)
                )
            else:
                raise NotImplementedError(
                    f"Unknown selection method: {selection_method}"
                )
            try:
                ac_mask = torch.nn.functional.one_hot(
                    exit_indices, outputs.shape[1]
                ).unsqueeze(-1)
            except:
                breakpoint()
            preds = (ac_mask * outputs).sum(dim=1).argmax(dim=-1)
            hits = (preds == targets).float()
        else:
            raise NotImplementedError(f"Unknown selection method: {selection_method}")

        acc = hits.mean()
        output_dict["acc"] = float(acc)
        output_dict["selection_method"] = selection_method
        output_dicts.append(output_dict)

    return output_dicts


if __name__ == "__main__":
    ROOT_DIR = Path("results_analysis")
    OUTPUT_DIR = Path("analysis_outputs/outputs_combination")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    result_dirs = ROOT_DIR.rglob("ee_eval.npy")

    outputs = []
    for result_dir in tqdm(list(result_dirs)):
        averaged_scores = compute_scores(
            result_dir.parent,
            methods=[
                "last",
                "max_conf",
                "min_entropy",
                "avg",
                "wavg_compute",
                "wavg_conf",
                "first_0.5_or_last",
                "first_0.75_or_last",
                "first_0.9_or_last",
                "first_0.95_or_last",
                "first_0.98_or_last",
                "first_0.99_or_last",
                "first_0.5_or_max_conf",
                "first_0.75_or_max_conf",
                "first_0.9_or_max_conf",
                "first_0.95_or_max_conf",
                "first_0.98_or_max_conf",
                "first_0.99_or_max_conf",
            ],
            device=device,
        )

        outputs.extend(averaged_scores)

    df = pd.DataFrame(outputs)
    df = df.sort_values(
        by=["setting", "method", "ee_config", "selection_method"], ascending=True
    )

    # Compute mean and average score over seeds for all selection methods
    df = (
        df.groupby(["setting", "method", "ee_config", "selection_method"])
        .aggregate({"acc": ["mean", "std"]})
        .reset_index()
    )
    df.columns = [
        "setting",
        "method",
        "ee_config",
        "selection_method",
        "acc_mean",
        "acc_std",
    ]
    df.to_csv(OUTPUT_DIR / "ee_scores.csv")
