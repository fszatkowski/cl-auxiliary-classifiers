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

sns.set_style("whitegrid")

AVG_TASK_NAME = "Avg"
COLOR_PALETTE_NAME = "Greens_d"
AVG_COLOR = "tab:blue"
MARKER = "X"
MARKER_SIZE = 8

TITLE_FONTSIZE = 18
TICKS_FONTSIZE = 14
LABELS_FONTSIZE = 18
LEGEND_FONTSIZE = 14

DIFF_PALETTE = {
    "BiC": "tab:green",
    "LwF": "tab:blue",
    "FT": "tab:red",
    "FT+Ex": "tab:orange",
}


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


def plot_per_ic_acc(
    data: pd.DataFrame, output_path: Path, plot_avg: bool = True, title: str = None
):
    plt.cla()
    plt.clf()
    plt.figure()

    n_tasks = data["task_id"].nunique() - 1
    colors = reversed(sns.color_palette(COLOR_PALETTE_NAME, n_colors=n_tasks))
    color_palette = {i: color for i, color in enumerate(colors)}
    if not plot_avg:
        data = data[~data["task_id"] == AVG_TASK_NAME]
    else:
        color_palette[AVG_TASK_NAME] = AVG_COLOR
    line_styles = {
        k: (1, 1) if k == AVG_TASK_NAME else (1, 0) for k in color_palette.keys()
    }

    hue_order = [t for t in range(n_tasks)]
    if plot_avg:
        hue_order = hue_order + [AVG_TASK_NAME]
    plot = sns.lineplot(
        x="ic_idx",
        y="acc",
        hue="task_id",
        hue_order=hue_order,
        data=data,
        legend=True,
        palette=color_palette,
        style="task_id",
        markers={k: MARKER for k in color_palette.keys()},
        markersize=MARKER_SIZE,
        dashes=line_styles,
    )

    plot.set_title(title, fontsize=TITLE_FONTSIZE)
    plot.set_xlabel("Classifier", fontsize=LABELS_FONTSIZE)
    plot.set_ylabel("Accuracy change", fontsize=LABELS_FONTSIZE)

    xticklabels = [
        "",
        "L1.B3",
        "L1.B5",
        "L2.B2",
        "L2.B4",
        "L3.B1",
        "L3.B3",
        "L3.B5",
    ]

    plot.set_xticklabels(xticklabels, fontsize=TICKS_FONTSIZE, rotation=0)

    if n_tasks > 6:
        ncols = ceil((n_tasks + int(plot_avg)) / 2)
    else:
        ncols = n_tasks + int(plot_avg)
    plot.legend(title="Task ID", ncol=ncols, handlelength=1)
    # Make lines on the legend shorter
    # Iterate over the legned lines
    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plot.get_figure().savefig(str(output_path))
    plot.get_figure().savefig(str(output_path).replace("png", "pdf"))


def plot_diff_df(df: pd.DataFrame, output_path: Path, title: str = None):
    plt.cla()
    plt.clf()
    plt.figure()

    plot = sns.lineplot(
        x="ic_idx", y="acc", hue="method", palette=DIFF_PALETTE, data=df
    )
    plot.set_title(title, fontsize=TITLE_FONTSIZE)
    plot.set_xlabel("Classifier", fontsize=LABELS_FONTSIZE)
    plot.set_ylabel("Accuracy change", fontsize=LABELS_FONTSIZE)

    xticklabels = [
        "",
        "L1.B3",
        "L1.B5",
        "L2.B2",
        "L2.B4",
        "L3.B1",
        "L3.B3",
        "L3.B5",
    ]

    plot.set_xticklabels(xticklabels, fontsize=TICKS_FONTSIZE, rotation=0)

    plot.legend(fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_FONTSIZE)

    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plot.get_figure().savefig(str(output_path))
    plot.get_figure().savefig(str(output_path).replace("pdf", "png"))


def plot_ic_accs(result_paths: List[Path], output_dir: Path):
    print("Plotting final per-IC accuracy plots")
    print(f"Found {len(result_paths)} results:")
    for result_path in result_paths:
        setting, method, ee_config, seed = decode_path(result_path)
        print(f"{setting}\t{method}\t{ee_config}\t{seed}")

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
        # plot_per_ic_acc(
        #     df_diff_tag,
        #     output_path_tag,
        #     plot_avg=True,
        #     title=f"{setting} | {method_name}",
        # )

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
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
    for i, setting in enumerate(["CIFAR100x5", "CIFAR100x10"]):
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
            palette=DIFF_PALETTE,
            data=diff_df,
            ax=axs[i],
            legend=i == 0,
        )
        plot.set_title(f"{setting}", fontsize=TITLE_FONTSIZE)
        plot.set_xlabel("Classifier", fontsize=LABELS_FONTSIZE)
        if i == 0:
            plot.set_ylabel("Accuracy change", fontsize=LABELS_FONTSIZE)
        else:
            plot.set_ylabel(None)

        xticklabels = [
            "",
            "L1.B3",
            "L1.B5",
            "L2.B2",
            "L2.B4",
            "L3.B1",
            "L3.B3",
            "Final",
        ]
        plot.set_xticklabels(xticklabels, fontsize=TICKS_FONTSIZE, rotation=0)
        yticklabels = plot.get_yticklabels()
        fixed_yticklabels = []
        for label in yticklabels:
            if "0" in label.get_text() or "−" in label.get_text():
                fixed_yticklabels.append(label)
            else:
                fixed_yticklabels.append("+" + label.get_text())

        plot.set_yticklabels(fixed_yticklabels, fontsize=TICKS_FONTSIZE)
        if i == 0:
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
                fontsize=LEGEND_FONTSIZE,
                title_fontsize=LEGEND_FONTSIZE,
                ncol=2,
                loc="lower left",
            )

    plt.tight_layout()
    fig.savefig(output_dir / f"acc_change_when_training_acs.pdf")


if __name__ == "__main__":
    root = Path(__file__).parent.parent.parent
    result_dirs = root / "results_analysis"
    # results = sorted(list(result_dirs.glob('CIFAR*/finetuning*_sdn/*/*')) + list(result_dirs.glob('CIFAR*/finetuning*0/*/*')))
    result_paths = sorted(
        list(result_dirs.glob("CIFAR100x10/*_sdn*/*/*"))
        + list(result_dirs.glob("CIFAR100x5/*_sdn*/*/*"))
    )
    output_dir = root / "analysis_outputs" / "acc_change_when_training_acs"

    plot_ic_accs(result_paths, output_dir)
