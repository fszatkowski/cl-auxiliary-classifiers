import itertools
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

CMAP = "rocket"
TITLE_FONTSIZE = 18
TICKS_FONTSIZE = 16
LABELS_FONTSIZE = 18
DATA_FONTSIZE = 14

VMIN = 0.3
VMAX = 0.9

METHOD_NAMES = {
    "finetuning_ex0": "FT",
    "finetuning_ex2000": "FT+Ex",
    "joint": "Joint",
    "bic": "BiC",
    "lwf": "LwF",
}


def plot_task_cka(data: torch.Tensor, output_path: Path, title: str = None):
    num_tasks = data.shape[1]
    nan_mask = np.isnan(data)
    task_is_nan = nan_mask.sum(axis=0) > 0
    data = data[:, ~task_is_nan]
    start_task_idx = num_tasks - data.shape[1]

    plt.cla()
    plt.clf()
    plt.figure()

    plot = sns.heatmap(
        data,
        cmap=CMAP,
        vmin=VMIN,
        vmax=VMAX,
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
            "L3.B5",
        ]
    else:
        yticklabels = ["L3.B5"]
    plot.set_yticklabels(yticklabels, fontsize=TICKS_FONTSIZE, rotation=0)
    plot.set_xticklabels(
        list(range(start_task_idx + 1, num_tasks + 1)), fontsize=TICKS_FONTSIZE
    )

    output_path = str(output_path).replace(" ", "_")
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plot.get_figure().savefig(output_path + ".pdf")
    plot.get_figure().savefig(output_path + ".png")


def _parse_path(path: Path):
    cl_method = None
    for k in METHOD_NAMES:
        if k in str(path.parent):
            cl_method = METHOD_NAMES[k]
    assert cl_method is not None

    if "detach" in str(path.parent):
        ac_setup = "Linear probing"
    elif "no_ee" in str(path.parent):
        ac_setup = "No ACs"
    else:
        ac_setup = "ACs"

    setting = str(path.parent.parent.name)
    task_id = int(path.name.split("_")[-1].replace(".pt", ""))

    return setting, cl_method, ac_setup, task_id


def plot_cka(result_paths: List[Path], output_dir: Path):
    sorted_data = {}
    for result_path in result_paths:
        setting, cl_method, ac_setup, task_id = _parse_path(result_path)
        sorted_data[(setting, cl_method, ac_setup, task_id)] = torch.load(result_path)

    # print("Plotting CKA plots...")
    # print(f"Found {len(result_paths)} results:")
    # for (setting, method, ac_setup, task_id), cka_data in tqdm(sorted_data.items()):
    #     if task_id != 0:
    #         continue
    #     cka_output_path = (
    #             output_dir / f"{setting}" / f"{method}_{ac_setup}" / f"task_{task_id + 1}"
    #     )
    #     if torch.all(torch.isnan(cka_data)):
    #         continue
    #     plot_task_cka(
    #         cka_data,
    #         cka_output_path,
    #         title=f"{setting} | {method} ({ac_setup}), task {task_id + 1}",
    #     )

    settings = sorted(list(set([k[0] for k in sorted_data.keys()])))
    methods = ["FT", "FT+Ex", "LwF", "BiC"]
    ac_setups = ["ACs", "Linear probing"]

    output_dir.mkdir(exist_ok=True, parents=True)
    for setting in settings:
        for ac_setup in ac_setups:
            cka_data = {
                method: sorted_data[(setting, method, ac_setup, 0)]
                for method in methods
            }

            plt.cla()
            plt.clf()
            fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

            for idx in range(len(methods)):
                method = methods[idx]
                data = cka_data[method]
                nan_mask = np.isnan(data)
                task_is_nan = nan_mask.sum(axis=0) > 0
                data = data[:, ~task_is_nan]
                ax = axes[idx]
                method_plot = sns.heatmap(
                    data,
                    vmin=VMIN,
                    vmax=VMAX,
                    cmap=CMAP,
                    ax=ax,
                    cbar=False,
                    annot=True,
                    fmt=".2f",
                    annot_kws={"size": DATA_FONTSIZE},
                )
                method_plot.set_title(f"{setting} | {method}", fontsize=TITLE_FONTSIZE)
                method_plot.set_xlabel("Task index", fontsize=LABELS_FONTSIZE)
                if "x10" in setting:
                    method_plot.set_xticklabels(
                        list(range(2, 11)), fontsize=TICKS_FONTSIZE
                    )
                elif "x5" in setting:
                    method_plot.set_xticklabels(
                        list(range(2, 6)), fontsize=TICKS_FONTSIZE
                    )
                else:
                    raise NotImplementedError()
                if idx == 0:
                    method_plot.set_yticklabels(
                        [
                            "AC1",
                            "AC2",
                            "AC3",
                            "AC4",
                            "AC5",
                            "AC6",
                            "Final"
                        ],
                        fontsize=TICKS_FONTSIZE,
                        rotation=0,
                    )
                else:
                    method_plot.set_yticklabels([], fontsize=TICKS_FONTSIZE, rotation=0)

            fig.tight_layout()
            # fig.savefig(output_dir / f"{setting}_{ac_setup}.png")
            fig.savefig(output_dir / f"{setting}_{ac_setup.replace(' ', '_')}.pdf")


if __name__ == "__main__":
    root = Path(__file__).parent.parent.parent
    between_task_cka_data_dir = root / "analysis_outputs" / "cka_between_tasks"
    between_task_cka_paths = sorted(
        list(between_task_cka_data_dir.glob("*/*/raw_data*"))
    )
    between_task_cka_paths = [
        p for p in between_task_cka_paths if "no_ee" not in str(p)
    ]
    output_dir = root / "analysis_outputs" / "cka" / "between_tasks"
    plot_cka(between_task_cka_paths, output_dir)

    wrt_first_task_cka_data_dir = root / "analysis_outputs" / "cka_wrt_first_task"
    wrt_first_task_cka_paths = sorted(
        list(wrt_first_task_cka_data_dir.glob("*/*/raw_data*"))
    )
    wrt_first_task_cka_paths = [
        p for p in wrt_first_task_cka_paths if "no_ee" not in str(p)
    ]
    output_dir = root / "analysis_outputs" / "cka" / "wrt_first_task"
    plot_cka(wrt_first_task_cka_paths, output_dir)
