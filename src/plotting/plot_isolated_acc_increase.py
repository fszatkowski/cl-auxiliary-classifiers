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

CMAP = "Greens_r"
TITLE_FONTSIZE = 18
TICKS_FONTSIZE = 16
LABELS_FONTSIZE = 18
DATA_FONTSIZE = 12

VMIN = -3
VMAX = 7

METHOD_NAMES = {
    "finetuning_ex0": "FT",
    "finetuning_ex2000": "FT+Ex",
    "joint": "Joint",
    "bic": "BiC",
    "lwf": "LwF",
}


def plot_unique_acc(data: torch.Tensor, output_path: Path, title: str = None):
    plt.cla()
    plt.clf()
    plt.figure()

    data_expanded = torch.zeros(data.shape[0], data.shape[1] + 1)
    data_expanded[:, :-1] = data
    data_expanded[:, -1] = torch.mean(data, dim=1)

    plot = sns.heatmap(
        data_expanded,
        cmap=CMAP,
        vmin=0,
        vmax=10,
        annot=True,
        fmt=".2f",
        cbar=False,
        annot_kws={"size": DATA_FONTSIZE},
    )
    plot.set_title(title, fontsize=TITLE_FONTSIZE)
    plot.set_xlabel("Task index", fontsize=LABELS_FONTSIZE)
    # Set custom yticks
    if data.shape[0] == 7:
        yticklabels = [
            "L1.B3",
            "L1.B5",
            "L2.B2",
            "L2.B4",
            "L3.B1",
            "L3.B3",
            "Final",
        ]
    else:
        yticklabels = ["L3.B5"]
    plot.set_yticklabels(yticklabels, fontsize=TICKS_FONTSIZE, rotation=0)
    plot.set_xticklabels(
        list(range(1, data_expanded.shape[1])) + ["Avg"], fontsize=TICKS_FONTSIZE
    )

    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plot.get_figure().savefig(str(output_path) + ".pdf")
    plot.get_figure().savefig(str(output_path) + ".png")


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


def plot_unique_acc_between_tasks(
    result_paths: List[Path], output_dir: Path, device: str
):
    # print(f"Found {len(result_paths)} results:")
    # for data_path in tqdm(result_paths, desc="Parsing results data..."):
    #     data = torch.load(data_path, map_location=device)
    #
    #     setting = data_path.parent.name
    #
    #     method = None
    #     for method_name in METHOD_NAMES.keys():
    #         if data_path.name.startswith(method_name):
    #             method = METHOD_NAMES[method_name]
    #     if method is None:
    #         raise ValueError(f"Could not find method for {data_path.name}")
    #
    #     if "detach" in data_path.name:
    #         ee_setup = "Linear probing"
    #     else:
    #         ee_setup = "ACs"
    #
    #     output_path = output_dir / f"{setting}" / f"{method}_{ee_setup}"
    #     output_path.parent.mkdir(exist_ok=True, parents=True)
    #     plot_unique_acc(
    #         data=data,
    #         output_path=output_path,
    #         title=f"{setting} | {method} ({ee_setup}) | Unique acc",
    #     )
    setting_to_data = {}
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
            ee_setup = "Linear probing"
        else:
            ee_setup = "ACs"

        if (setting, ee_setup) in setting_to_data:
            setting_to_data[(setting, ee_setup)][method] = data
        else:
            setting_to_data[(setting, ee_setup)] = {method: data}

    methods = ['FT', 'FT+Ex', 'LwF', 'BiC']
    for (setting, ee_setup), method_data in setting_to_data.items():
        plt.cla()
        plt.clf()

        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
        for idx in range(len(methods)):
            method = methods[idx]
            data = method_data[method]

            data_expanded = torch.zeros(data.shape[0], data.shape[1] + 1)
            data_expanded[:, :-1] = data
            data_expanded[:, -1] = torch.mean(data, dim=1)
            data = data_expanded

            method_plot = sns.heatmap(
                data,
                vmin=VMIN,
                vmax=VMAX,
                cmap=CMAP,
                ax=axes[idx],
                cbar=False,
                annot=True,
                fmt=".2f",
                annot_kws={"size": DATA_FONTSIZE},
            )
            method_plot.set_title(f"{setting} | {method}", fontsize=TITLE_FONTSIZE)
            method_plot.set_xlabel("Task", fontsize=LABELS_FONTSIZE)
            if "x10" in setting:
                method_plot.set_xticklabels(
                    list(range(1, 11)) + ["Avg"], fontsize=TICKS_FONTSIZE
                )
            elif "x5" in setting:
                method_plot.set_xticklabels(
                    list(range(1, 6)) + ["Avg"], fontsize=TICKS_FONTSIZE
                )
            else:
                raise NotImplementedError()
            if idx == 0:
                method_plot.set_yticklabels(
                    [
                        "L1.B3",
                        "L1.B5",
                        "L2.B2",
                        "L2.B4",
                        "L3.B1",
                        "L3.B3",
                        "Final",
                    ],
                    fontsize=TICKS_FONTSIZE,
                    rotation=0,
                )
            else:
                method_plot.set_yticklabels([], fontsize=TICKS_FONTSIZE, rotation=0)

        plt.tight_layout()
        output_path = output_dir / f"{setting}_{ee_setup}.pdf"
        output_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_path)


if __name__ == "__main__":
    root = Path(__file__).parent.parent.parent
    input_dir = root / "analysis_outputs" / "unique_acc_analysis"
    output_dir = root / "analysis_outputs" / "unique_acc_plots"
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
