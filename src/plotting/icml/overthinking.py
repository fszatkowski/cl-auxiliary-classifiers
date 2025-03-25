import itertools
from math import ceil
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from metrics import _CKA, cka
from plotting.icml.common import *
from plotting.utils import decode_path

sns.set_style("whitegrid")

AVG_TASK_NAME = "Avg"
WAVG_TASK_NAME = "WAvg"


def get_overthinking_data(path: Path, device: str):
    setting, method, ee_config, seed = decode_path(path)
    logits_dir = path / "outputs"
    after_task_dirs = list(logits_dir.glob("after_*"))
    final_dir = max(after_task_dirs, key=lambda x: int(x.stem.split("_")[-1]))
    task_dirs = sorted(list(final_dir.glob("t_*")))

    outputs = []
    for task_id in range(len(task_dirs)):
        batches_paths = sorted(list((final_dir / f"t_{task_id}").glob("*.pt")))
        hits_tag_final = 0
        hits_tag_any = 0
        total = 0
        for batch in batches_paths:
            data = torch.load(batch, map_location=device)

            preds_tag = torch.stack(
                [torch.cat(ic_data, dim=-1) for ic_data in data["logits"]], dim=1
            ).argmax(dim=-1)
            targets_tag = data["targets"]

            hits_tag_final += (preds_tag[:, -1] == targets_tag).sum().item()
            hits_tag_any += (
                ((preds_tag == targets_tag.unsqueeze(1)).sum(dim=-1) > 0).sum().item()
            )

            batch_size = targets_tag.shape[0]
            total += batch_size

        tag_acc_final = hits_tag_final / total
        tag_acc_any = hits_tag_any / total

        # TODO check averaging for bugs
        outputs.append(
            {
                "setting": setting,
                "method": method,
                "ee_config": ee_config,
                "seed": seed,
                "task_id": task_id,
                "tag_acc_final": tag_acc_final,
                "tag_acc_any": tag_acc_any,
                "tag_overthinking": tag_acc_any - tag_acc_final,
                "total": total,
            }
        )

    total_samples = sum([o["total"] for o in outputs])
    avg_tag_acc_final = sum([o["tag_acc_final"] for o in outputs]) / len(outputs)
    avg_tag_acc_any = sum([o["tag_acc_any"] for o in outputs]) / len(outputs)

    output_avg_acc = {
        "setting": setting,
        "method": method,
        "ee_config": ee_config,
        "seed": seed,
        "task_id": AVG_TASK_NAME,
        "tag_acc_final": avg_tag_acc_final,
        "tag_acc_any": avg_tag_acc_any,
        "tag_overthinking": avg_tag_acc_any - avg_tag_acc_final,
        "total": total_samples,
    }

    wavg_tag_acc_final = (
        sum([o["tag_acc_final"] * o["total"] for o in outputs]) / total_samples
    )
    wavg_tag_acc_any = (
        sum([o["tag_acc_any"] * o["total"] for o in outputs]) / total_samples
    )

    output_wavg_acc = {
        "setting": setting,
        "method": method,
        "ee_config": ee_config,
        "seed": seed,
        "task_id": WAVG_TASK_NAME,
        "tag_acc_final": wavg_tag_acc_final,
        "tag_acc_any": wavg_tag_acc_any,
        "tag_overthinking": wavg_tag_acc_any - wavg_tag_acc_final,
        "total": total_samples,
    }

    outputs.append(output_avg_acc)
    outputs.append(output_wavg_acc)

    return outputs


def setup_fn(x):
    if "detach" in x:
        return "LP"
    else:
        return "AC"


def method_fn(x):
    if x["setting"] == "CIFAR100x1":
        method = "Joint"
    else:
        method = x["method"]
        if "lwf" in method:
            method = "LwF"
        elif "bic" in method:
            method = "BiC"
        elif "ex2000" in method:
            method = "FT+Ex"
        else:
            method = "FT"
    return method


def plot_overthinking_between_tasks(
    result_paths: List[Path], output_dir: Path, device: str
):
    output_dir.mkdir(exist_ok=True, parents=True)
    print("Plotting overthinking...")
    print(f"Found {len(result_paths)} results:")

    artifact_path = output_dir.parent.joinpath('artifacts').joinpath("overthinking_data.csv")
    if artifact_path.exists():
        df = pd.read_csv(artifact_path)
    else:
        overthinking_data = []
        for result_path in tqdm(result_paths, desc="Computing overthinking..."):
            overthinking_data.extend(get_overthinking_data(result_path, device))

        df = pd.DataFrame(overthinking_data)
        df.to_csv(artifact_path, index=False)

    df = df[df["task_id"] != WAVG_TASK_NAME]
    df["cl_method"] = df.apply(method_fn, axis=1)
    df["setup"] = df["method"].apply(setup_fn)
    df['rel_overthinking'] = df['tag_overthinking'] / df['tag_acc_final']
    df['rel_overthinking'] = df['rel_overthinking'] * 100
    df['tag_overthinking'] = df['tag_overthinking'] * 100

    # output_dir.mkdir(exist_ok=True, parents=True)
    # for setting in ["CIFAR100x5", "CIFAR100x10"]:
    #     for setup in ["LP", "AC"]:
    #         tmp_df = df[(df["setting"] == setting) & (df["setup"] == setup)]
    #         joint_df = df[df["cl_method"] == "Joint"]
    #         tmp_df = pd.concat([tmp_df, joint_df])
    #         tmp_df = tmp_df.sort_values(by=['cl_method'], key=lambda x: x.map({'Joint': 0, 'FT': 1, 'FT+Ex': 2, 'LwF': 3, "BiC": 4}), ascending=True)
    #
    #         plt.clf()
    #         plt.cla()
    #         plt.figure()
    #
    #         plot = sns.barplot(
    #             tmp_df,
    #             x="task_id",
    #             y="tag_overthinking",
    #             hue="cl_method",
    #             palette=METHOD_TO_COLOR,
    #         )
    #         # Set legend title
    #         handles, labels = plot.get_legend_handles_labels()
    #         plot.legend(
    #             handles=handles,
    #             labels=labels,
    #             title=None,
    #             loc="upper right",
    #             fontsize=FONTSIZE_LEGEND,
    #         )
    #         plot.set_title(None)
    #         plot.set_xlabel("Task ID", fontsize=FONTSIZE_LABELS)
    #         plot.set_ylabel("Overthinking", fontsize=FONTSIZE_LABELS)
    #         plot.set_xticklabels(plot.get_xticklabels(), fontsize=FONTSIZE_TICKS)
    #         plot.set_yticklabels(plot.get_yticklabels(), fontsize=FONTSIZE_TICKS)
    #         plt.tight_layout()
    #         plot.get_figure().savefig(
    #             output_dir.joinpath(f"overthinking_per_task_{setting}_{setup}.pdf")
    #         )
            # plot.get_figure().savefig(
            #     output_dir.joinpath(f"overthinking_per_task_{setting}_{setup}.png")
            # )

    for setup in ["LP", "AC"]:
        plt.cla()
        plt.clf()
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
        for setting_idx, setting in enumerate(["CIFAR100x5", "CIFAR100x10"]):
            tmp_df = df[(df["setting"] == setting) & (df["setup"] == setup)]
            joint_df = df[df["cl_method"] == "Joint"]
            tmp_df = pd.concat([tmp_df, joint_df])
            tmp_df = tmp_df[tmp_df["task_id"] == AVG_TASK_NAME]
            tmp_df = tmp_df.sort_values(by=['cl_method'], key=lambda x: x.map({'Joint': 0, 'FT': 1, 'FT+Ex': 2, 'LwF': 3, "BiC": 4}), ascending=True)

            plot = sns.barplot(
                tmp_df,
                ax=axes[setting_idx],
                x="cl_method",
                y="rel_overthinking",
                hue="cl_method",
                palette=METHOD_TO_COLOR,
                legend=False
            )

            plot.set_title(f"{setting} | {setup}", fontsize=FONTSIZE_TITLE)
            plot.set_xlabel(None)
            if setting_idx == 0:
                plot.set_ylabel("Overthinking", fontsize=FONTSIZE_LABELS)
            else:
                plot.set_ylabel("")
            plot.set_xticklabels(plot.get_xticklabels(), fontsize=FONTSIZE_TICKS)
            plot.set_yticklabels(plot.get_yticklabels(), fontsize=FONTSIZE_TICKS)
        plt.tight_layout()
        fig.savefig(output_dir.joinpath(f"overthinking_{setup}.pdf"))
        # fig.savefig(output_dir.joinpath(f"overthinking_{setup}.png"))

    for setup in ["LP", "AC"]:
        for  setting in ["CIFAR100x5", "CIFAR100x10"]:
            plt.cla()
            plt.clf()
            plt.figure()

            tmp_df = df[(df["setting"] == setting) & (df["setup"] == setup)]
            joint_df = df[df["cl_method"] == "Joint"]
            tmp_df = pd.concat([tmp_df, joint_df])
            tmp_df = tmp_df[tmp_df["task_id"] == AVG_TASK_NAME]
            tmp_df = tmp_df.sort_values(by=['cl_method'], key=lambda x: x.map({'Joint': 0, 'FT': 1, 'FT+Ex': 2, 'LwF': 3, "BiC": 4}), ascending=True)

            plot = sns.barplot(
                tmp_df,
                x="cl_method",
                y="rel_overthinking",
                hue="cl_method",
                palette=METHOD_TO_COLOR,
                legend=False
            )

            # plot.set_title(f"{setting} | {setup}", fontsize=FONTSIZE_TITLE)
            plot.set_title(None)
            plot.set_xlabel(None)
            plot.set_ylabel("Overthinking [%]", fontsize=FONTSIZE_LABELS)
            # Make y axis logarithmic
            plot.set_yscale("log")
            # Make y ticks use standard notation
            plot.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            # Add ticks for 10, 50, 100
            plot.get_yaxis().set_ticks([10, 25, 50, 100, 200])

            plot.set_xticklabels(plot.get_xticklabels(), fontsize=FONTSIZE_TICKS-2)
            plot.set_yticklabels(plot.get_yticklabels(), fontsize=FONTSIZE_TICKS-2)
            plt.tight_layout()
            plot.get_figure().savefig(output_dir.joinpath(f"overthinking_{setting}_{setup}.pdf"))
        # fig.savefig(output_dir.joinpath(f"overthinking_{setup}.png"))


if __name__ == "__main__":
    root = Path(__file__).parent.parent.parent.parent
    result_dirs = root / "results_analysis"
    result_paths = (
        sorted(list(result_dirs.glob("CIFAR100x1/*_sdn*/*/*")))
        + sorted(list(result_dirs.glob("CIFAR100x5/*_sdn*/*/*")))
        + sorted(list(result_dirs.glob("CIFAR100x10/*_sdn*/*/*")))
    )
    output_dir = root / "icml_data" / "overthinking"

    if torch.cuda.is_available():
        device = "cuda"
        print("Using GPU")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS")
    else:
        device = "cpu"
        print("Using CPU")
    plot_overthinking_between_tasks(result_paths, output_dir, device=device)
