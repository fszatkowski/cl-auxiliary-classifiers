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
from plotting.icml.common import *
sns.set_style("whitegrid")

AVG_TASK_NAME = "Avg"
WAVG_TASK_NAME = "WAvg"



METHOD_NAMES = {
    "finetuning_ex0": "FT",
    "finetuning_ex2000": "FT+Ex",
    "joint": "Joint",
    "bic": "BiC",
    "lwf": "LwF",
}



def get_unique_acc_data(path: Path, device: str):
    logits_dir = path / "outputs"
    after_task_dirs = list(logits_dir.glob("after_*"))
    final_dir = max(after_task_dirs, key=lambda x: int(x.stem.split("_")[-1]))
    task_dirs = sorted(list(final_dir.glob("t_*")))

    n_tasks = len(task_dirs)

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


def plot_unique_acc_between_tasks(
    result_paths: List[Path], output_dir: Path, device: str
):
    outputs = []
    print(f"Found {len(result_paths)} results:")
    for data_path in tqdm(result_paths):
        data = torch.load(data_path, map_location=device)

        setting = data_path.parent.name
        method = None
        for method_name in METHOD_NAMES.keys():
            if data_path.name.startswith(method_name):
                method = METHOD_NAMES[method_name]
        if method is None:
            raise ValueError(f"Could not find method for {data_path.name}")

        if "detach" in data_path.name:
            ee_setup = "LP"
        else:
            ee_setup = "AC"

        per_ac_avg_acc = data.mean(dim=1)
        for ac_idx, ac_acc in enumerate(per_ac_avg_acc):
            outputs.append({
                "setting": setting,
                "method": method,
                "ee_setup": ee_setup,
                "ac_idx": ac_idx,
                "acc": float(ac_acc)
            })

    df = pd.DataFrame(outputs)
    df = df[df['method'] != 'Joint']
    for setting in df['setting'].unique():
        for setup in df['ee_setup'].unique():
            df_setup = df[(df['ee_setup'] == setup) & (df['setting'] == setting)]

            plt.cla()
            plt.clf()
            plt.figure()

            plot = sns.lineplot(
                data=df_setup,
                x="ac_idx",
                y="acc",
                hue="method",
                 palette=METHOD_TO_COLOR,
                linewidth=LINEWIDTH,
                )

            plot.set_title(f"{setting} | {setup}", fontsize=FONTSIZE_TITLE)
            plot.set_xlabel(None)
            plot.set_ylabel("Unique accuracy", fontsize=FONTSIZE_LABELS)

            plot.set_xticklabels(
                [
                    "",
                    "L1.B3",
                    "L1.B5",
                    "L2.B2",
                    "L2.B4",
                    "L3.B1",
                    "L3.B3",
                    "Final",
                ],
                fontsize=FONTSIZE_TICKS,
            )
            plot.set_yticklabels(plot.get_yticks(), fontsize=FONTSIZE_TICKS)

            # Plot legend
            handles, labels = plot.get_legend_handles_labels()
            plot.legend(
                title=None,
                handles=handles,
                labels=labels,
                loc="lower right",
                fontsize=FONTSIZE_LEGEND
            )
            plt.tight_layout()
            output_dir.mkdir(exist_ok=True, parents=True)
            plot.get_figure().savefig(         output_dir / f"{setting}_{setup}_lineplot.png")
            plot.get_figure().savefig(         output_dir / f"{setting}_{setup}_lineplot.pdf")


if __name__ == "__main__":
    root = Path(__file__).parent.parent.parent.parent
    input_dir = root / "analysis_outputs" / "unique_acc_analysis"
    output_dir = root / "icml_data" / "unique_acc"
    result_paths = sorted(list(input_dir.rglob("*.pt")))

    if torch.cuda.is_available():
        device = "cuda"
        print("Using GPU")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS")
    else:
        device = "cpu"
        print("Using CPU")
    plot_unique_acc_between_tasks(result_paths, output_dir, device=device)
