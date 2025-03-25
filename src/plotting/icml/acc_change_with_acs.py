from copy import deepcopy
from math import ceil
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from plotting.utils import decode_path
from plotting.icml.common import *

sns.set_style("whitegrid")

AVG_TASK_NAME = "Avg"


def load_ic_data(path: Path):
    results_file = path / "results" / "ee_eval.npy"
    setting, method, ee_config, seed = decode_path(path)
    data = np.load(results_file, allow_pickle=True).item()
    outputs = []
    for task_id, task_metrics in data.items():
        ic_acc_tag = task_metrics["per_ic_acc"]["tag"] * 100
        ic_acc_taw = task_metrics["per_ic_acc"]["taw"] * 100
        if task_id == "avg":
            task_id = AVG_TASK_NAME
        for ic_idx, (acc_tag, acc_taw) in enumerate(zip(ic_acc_tag, ic_acc_taw)):
            outputs.append(
                {
                    "setting": setting,
                    "method": method,
                    "ee_config": ee_config,
                    "seed": seed,
                    "task_id": task_id,
                    "ic_idx": ic_idx,
                    "acc": acc_tag,
                    "acc_type": "tag",
                }
            )
            outputs.append(
                {
                    "setting": setting,
                    "method": method,
                    "ee_config": ee_config,
                    "seed": seed,
                    "task_id": task_id,
                    "ic_idx": ic_idx,
                    "acc": acc_taw,
                    "acc_type": "taw",
                }
            )
    return outputs

def plot_ic_accs(result_paths: List[Path], output_dir: Path):
    print("Plotting final per-IC accuracy plots")

    outputs = []
    for result_path in tqdm(result_paths, desc="Parsing results data..."):
        outputs.extend(load_ic_data(result_path))

    df = pd.DataFrame(outputs)
    output_dir.mkdir(exist_ok=True, parents=True)

    settings = list(df["setting"].unique())
    methods = list(df["method"].unique())
    ee_configs = list(df["ee_config"].unique())

    assert settings == ["CIFAR100x10", "CIFAR100x5"]
    assert methods == [
        "bic_num_exemplars_2000_lamb_2",
        "bic_num_exemplars_2000_lamb_2_detach",
        "finetuning_ex0",
        "finetuning_ex0_detach",
        "finetuning_ex2000",
        "finetuning_ex2000_detach",
        "joint_ex_detach",
        "lwf_lamb_1.0",
        "lwf_lamb_1.0_detach",
    ]
    assert ee_configs == ["cifar100_resnet32_sdn"]

    filtered_methods = [
        "bic_num_exemplars_2000_lamb_2",
        "bic_num_exemplars_2000_lamb_2_detach",
        "finetuning_ex0",
        "finetuning_ex0_detach",
        "finetuning_ex2000",
        "finetuning_ex2000_detach",
        "lwf_lamb_1.0",
        "lwf_lamb_1.0_detach",
    ]
    df = df[df["method"].isin(filtered_methods)]

    methods = [
        "bic_num_exemplars_2000_lamb_2",
        "finetuning_ex0",
        "finetuning_ex2000",
        "lwf_lamb_1.0",
    ]
    iterator = tqdm(
        [(s, m, ee) for s in settings for m in methods for ee in ee_configs],
        desc="Plotting...",
    )

    for setting, method, ee_config in iterator:
        df_trained = df[
            (df["setting"] == setting)
            & (df["method"] == method)
            & (df["ee_config"] == ee_config)
        ]
        df_trained_tag = df_trained[df_trained["acc_type"] == "tag"]

        df_detached = df[
            (df["setting"] == setting)
            & (df["method"] == method + "_detach")
            & (df["ee_config"] == ee_config)
        ]
        df_detached_tag = df_detached[df_detached["acc_type"] == "tag"]

        df_trained_tag = df_trained_tag.sort_values(
            by=[
                "setting",
                "method",
                "ee_config",
                "seed",
                "task_id",
                "ic_idx",
                "acc_type",
            ]
        )
        df_detached_tag = df_detached_tag.sort_values(
            by=[
                "setting",
                "method",
                "ee_config",
                "seed",
                "task_id",
                "ic_idx",
                "acc_type",
            ]
        )

        df_diff_tag = deepcopy(df_trained_tag)
        df_diff_tag["acc"] = (
            df_trained_tag["acc"].values - df_detached_tag["acc"].values
        )
        if "finetuning_ex0" in method:
            output_filename = "ft"
        elif "finetuning_ex2000" in method:
            output_filename = "ft_ex"
        elif "lwf" in method:
            output_filename = "lwf"
        elif "bic" in method:
            output_filename = "bic"
        else:
            raise NotImplementedError()

        output_path_tag = output_dir / "tag" / f"{setting}" / f"{output_filename}.png"
        if "bic" in method:
            method_name = "BiC"
        elif "lwf" in method:
            method_name = "LwF"
        elif "ex2000" in method:
            method_name = "FT+Ex"
        else:
            method_name = "FT"

    def rename_fn(method_name):
        if "bic" in method_name.lower():
            return "BiC"
        elif "lwf" in method_name.lower():
            return "LwF"
        elif "ex2000" in method_name.lower():
            return "FT+Ex"
        else:
            return "FT"

    plt.cla()
    plt.clf()
    for setting in ["CIFAR100x5", "CIFAR100x10"]:
        plt.figure()
        avg_acc_df = df[
            (df["setting"] == setting)
            & (df["task_id"] == "Avg")
            & (df["acc_type"] == "tag")
        ]
        avg_acc_df = avg_acc_df[["method", "seed", "ic_idx", "acc"]]

        avg_acc_df_detached = avg_acc_df[avg_acc_df["method"].str.contains("detach")]
        avg_acc_df_trained = avg_acc_df[~avg_acc_df["method"].str.contains("detach")]

        avg_acc_df_detached = avg_acc_df_detached.sort_values(
            by=["method", "seed", "ic_idx"]
        )
        avg_acc_df_trained = avg_acc_df_trained.sort_values(
            by=["method", "seed", "ic_idx"]
        )

        diff = avg_acc_df_trained["acc"].values - avg_acc_df_detached["acc"].values
        diff_df = deepcopy(avg_acc_df_trained)
        diff_df["method"] = diff_df["method"].apply(rename_fn)
        diff_df["acc"] = diff

        plot = sns.lineplot(
            x="ic_idx",
            y="acc",
            hue="method",
            palette=METHOD_TO_COLOR,
            data=diff_df,
            legend=True,
        )
        plot.set_title(None)
        # plot.set_xlabel("Classifier", fontsize=FONTSIZE_LABELS)
        plot.set_xlabel(None)
        plot.set_ylabel("Accuracy change", fontsize=FONTSIZE_LABELS)


        xticklabels = [
            "",
            "AC1",
            "AC2",
            "AC3",
            "AC4",
            "AC5",
            "AC6",
            "Final",
        ]
        plot.set_xticklabels(xticklabels, fontsize=FONTSIZE_TICKS, rotation=0)
        yticklabels = plot.get_yticklabels()
        fixed_yticklabels = []
        for label in yticklabels:
            if "0" in label.get_text() or "−" in label.get_text():
                fixed_yticklabels.append(label)
            else:
                fixed_yticklabels.append("+" + label.get_text())

        plot.set_yticklabels(fixed_yticklabels, fontsize=FONTSIZE_TICKS)
        handles, labels = plot.get_legend_handles_labels()
        handles = handles[1:] + [handles[0]]
        labels = labels[1:] + [labels[0]]
        # labels_ordered = ['FT', "FT+Ex", "LwF", "BiC"]
        # # Reorder original handles and labels to follow the set order
        # handles = [handles[labels_ordered.index(label)] for label in labels]
        # labels = [labels_ordered.index(label) for label in labels]
        plot.legend(
            handles,
            labels,
            fontsize=FONTSIZE_LEGEND,
            title=None,
            ncol=2,
        )

        plt.tight_layout()
        plot.get_figure().savefig(output_dir / f"acc_change_{setting}.pdf")
        # plot.get_figure().savefig(output_dir / f"acc_change_{setting}.png")


if __name__ == "__main__":
    root = Path(__file__).parent.parent.parent.parent
    result_dirs = root / "results_analysis"
    # results = sorted(list(result_dirs.glob('CIFAR*/finetuning*_sdn/*/*')) + list(result_dirs.glob('CIFAR*/finetuning*0/*/*')))
    result_paths = sorted(
        list(result_dirs.glob("CIFAR100x10/*_sdn*/*/*"))
        + list(result_dirs.glob("CIFAR100x5/*_sdn*/*/*"))
    )
    output_dir = root / "icml_data" / 'acc_diff_ac_lp'

    plot_ic_accs(result_paths, output_dir)
