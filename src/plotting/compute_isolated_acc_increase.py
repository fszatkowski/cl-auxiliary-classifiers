import itertools
from collections import defaultdict
from math import ceil
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from metrics import _CKA, cka
from plotting.utils import decode_path

sns.set_style("whitegrid")

AVG_TASK_NAME = "Avg"
WAVG_TASK_NAME = "WAvg"


def get_unique_acc_data(path: Path, device: str):
    logits_dir = path / "outputs"
    after_task_dirs = list(logits_dir.glob("after_*"))
    final_dir = max(after_task_dirs, key=lambda x: int(x.stem.split("_")[-1]))
    task_dirs = sorted(list(final_dir.glob("t_*")))

    n_tasks = len(task_dirs)
    if n_tasks == 5:
        taw_offsets = [0, 20, 40, 60, 80]
    elif n_tasks == 10:
        taw_offsets = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    elif n_tasks == 6:
        taw_offsets = [0, 50, 60, 70, 80, 90]
    elif n_tasks == 11:
        taw_offsets = [0, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    else:
        raise NotImplementedError(
            f"Found unknown number of tasks: {n_tasks} for {path}"
        )

    unique_acc_matrix = torch.zeros((7, n_tasks))
    for task_id in range(len(task_dirs)):
        batches_paths = sorted(list((final_dir / f"t_{task_id}").glob("*.pt")))
        unique_acc = torch.zeros(7)
        total = 0
        for batch in batches_paths:
            data = torch.load(batch, map_location=device)

            preds_tag = torch.stack(
                [torch.cat(ic_data, dim=-1) for ic_data in data["logits"]], dim=1
            ).argmax(dim=-1)
            targets_tag = data["targets"]

            hits = preds_tag == targets_tag.unsqueeze(1)
            hits_sum = hits.sum(1)
            unique_hits = hits_sum == 1
            filtered_hits = hits[unique_hits]
            per_ac_hits = filtered_hits.sum(0)
            unique_acc += per_ac_hits
            batch_size = targets_tag.shape[0]
            total += batch_size

        unique_acc_increase = unique_acc / total
        unique_acc_matrix[:, task_id] = unique_acc_increase

    return unique_acc_matrix


def compute_unique_acc_between_tasks(
    result_paths: List[Path], output_dir: Path, device: str
):
    print(f"Found {len(result_paths)} results:")
    for result_path in result_paths:
        setting, method, ee_config, seed = decode_path(result_path)
        print(f"{setting}\t{method}\t{ee_config}\t{seed}")

    unique_acc_data = defaultdict(list)
    for result_path in tqdm(result_paths, desc="Computing unique acc..."):
        setting, method, ee_config, seed = decode_path(result_path)
        unique_acc_data[(setting, method, ee_config)].append(
            get_unique_acc_data(result_path, device)
        )

    for key, accs_data in unique_acc_data.items():
        unique_acc_data[key] = torch.stack(accs_data, dim=0).mean(dim=0)

    iterator = tqdm(list(unique_acc_data.keys()), desc="Saving the data...")
    output_dir.mkdir(exist_ok=True, parents=True)
    for setting, method, ee_config in iterator:
        data = unique_acc_data[(setting, method, ee_config)]
        plot_output_dir = output_dir / f"{setting}"
        plot_output_dir.mkdir(exist_ok=True, parents=True)
        data = data * 100
        torch.save(data, plot_output_dir / f"{method}_{ee_config}.pt")


if __name__ == "__main__":
    root = Path(__file__).parent.parent.parent
    result_dirs = root / "results_analysis"
    result_paths = sorted(list(result_dirs.glob("CIFAR100x10/*_sdn*/*/*"))) + sorted(
        list(result_dirs.glob("CIFAR100x5/*_sdn*/*/*"))
    )
    output_dir = root / "analysis_outputs" / "unique_acc_analysis"

    if torch.cuda.is_available():
        device = "cuda"
        print("Using GPU")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS")
    else:
        device = "cpu"
        print("Using CPU")
    compute_unique_acc_between_tasks(result_paths, output_dir, device=device)
