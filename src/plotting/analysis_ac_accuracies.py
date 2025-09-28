from math import ceil
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from plotting.common import *
from plotting.utils import decode_path

sns.set_style("whitegrid")


def load_ic_data(path: Path):
    results_file = path / "results" / "ee_eval.npy"
    setting, method, ee_config, seed = decode_path(path)
    data = np.load(results_file, allow_pickle=True).item()
    outputs = []
    for task_id, task_metrics in data.items():
        ic_acc_tag = task_metrics["per_ic_acc"]["tag"] * 100
        ic_acc_taw = task_metrics["per_ic_acc"]["taw"] * 100
        if task_id == "avg":
            continue

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

    plot.set_title(title, fontsize=FONTSIZE_TITLE)
    plot.set_xlabel("Classifier", fontsize=FONTSIZE_LABELS)
    plot.set_ylabel("Accuracy", fontsize=FONTSIZE_LABELS)
    plot.set_xticklabels(
        ["", "L1.B3", "L1.B5", "L2.B2", "L2.B4", "L3.B1", "L3.B3", "Final"],
        fontsize=FONTSIZE_TICKS,
    )
    plot.set_yticklabels(plot.get_yticklabels(), fontsize=FONTSIZE_TICKS)

    if n_tasks > 6:
        ncols = ceil((n_tasks + int(plot_avg)) / 2)
    else:
        ncols = n_tasks + int(plot_avg)

    # Get plot handles and labels
    handles, labels = plot.get_legend_handles_labels()
    modified_labels = []
    for l in labels:
        if l == AVG_TASK_NAME:
            modified_labels.append(l)
        else:
            modified_labels.append(str(int(l) + 1))
    plot.legend(
        handles,
        modified_labels,
        title="Task ID",
        ncol=ncols,
        handlelength=1,
        fontsize=FONTSIZE_LEGEND,
        title_fontsize=FONTSIZE_LEGEND,
    )

    # Make lines on the legend shorter
    # Iterate over the legned lines
    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plot.get_figure().savefig(str(output_path))
    plot.get_figure().savefig(str(output_path).replace(".png", ".pdf"))


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
    df = df[df["acc_type"] == "tag"]
    output_dir.mkdir(exist_ok=True, parents=True)

    settings = ["CIFAR100x10", "CIFAR100x5"]
    methods = list(df["method"].unique())
    ee_configs = list(df["ee_config"].unique())

    dfs = {}
    for setting, method, ee_config in tqdm(
        [(s, m, ee) for s in settings for m in methods for ee in ee_configs],
    ):
        df_tmp = df[
            (df["setting"] == setting)
            & (df["method"] == method)
            & (df["ee_config"] == ee_config)
        ]
        df_tmp = df_tmp[df_tmp["acc_type"] == "tag"]

        if "detach" in method:
            setup = "Linear probing"
        else:
            setup = "ACs"

        if method.startswith("bic"):
            method = "BiC"
        elif method.startswith("lwf"):
            method = "LwF"
        elif method.startswith("finetuning_ex0"):
            method = "FT"
        elif method.startswith("finetuning_ex2000"):
            method = "FT+Ex"
        else:
            method = "Joint"

        dfs[(setting, method, setup)] = df_tmp

    methods = ["FT", "FT+Ex", "LwF", "BiC"]
    for setting in ["CIFAR100x10", "CIFAR100x5"]:
        for ee_config in ["Linear probing", "ACs"]:
            method_dfs = []
            for method in methods:
                df_tmp = dfs[(setting, method, ee_config)]
                method_dfs.append(df_tmp)

            plt.clf()
            plt.cla()
            fig, axes = plt.subplots(1, 4, figsize=(22, 4))

            for idx in range(len(methods)):
                df_tmp = method_dfs[idx]
                method = methods[idx]
                ax = axes[idx]

                new_datapoints = []
                seeds = df_tmp["seed"].unique()
                task_ids = df_tmp["task_id"].unique()
                cls_ids = sorted(df_tmp["ic_idx"].unique())
                for seed in seeds:
                    for task_id in task_ids:
                        task_cls_accs = [
                            df_tmp[
                                (df_tmp["seed"] == seed)
                                & (df_tmp["task_id"] == task_id)
                                & (df_tmp["ic_idx"] == cls_idx)
                            ]["acc"].values[0]
                            for cls_idx in cls_ids
                        ]
                        acc_difs = [
                            cls_acc - task_cls_accs[-1] for cls_acc in task_cls_accs
                        ]
                        relative_acc_diffs = [
                            100 * acc_dif / task_cls_accs[-1] for acc_dif in acc_difs
                        ]

                        for ic_idx, ac_diff, ac_rel_diff in zip(
                            cls_ids, acc_difs, relative_acc_diffs
                        ):
                            new_datapoints.append(
                                {
                                    "task_id": task_id,
                                    "ic_idx": ic_idx,
                                    "seed": seed,
                                    "final_acc": task_cls_accs[-1],
                                    "method": method,
                                    "ee_config": ee_config,
                                    "acc_diff": ac_diff,
                                    "rel_acc_diff": ac_rel_diff,
                                }
                            )

                parsed_acc_df = pd.DataFrame(new_datapoints)
                parsed_acc_df = parsed_acc_df[parsed_acc_df["ic_idx"] != 6]

                n_tasks = df_tmp["task_id"].nunique()
                colors = list(reversed(sns.color_palette("tab10", n_colors=n_tasks)))
                color_palette = {i: color for i, color in enumerate(colors)}

                # Do not plot error background for seeds
                method_plot = sns.lineplot(
                    x="ic_idx",
                    y="acc_diff",
                    hue="task_id",
                    data=parsed_acc_df,
                    palette=color_palette,
                    # style="task_id",
                    ax=ax,
                    legend=idx == 3,
                    ci=None,
                )

                # Make the lower limit 0
                method_plot.set_ylim(bottom=0)
                method_plot.set_title(f"{setting} | {method}", fontsize=FONTSIZE_TITLE)
                # method_plot.set_xlabel("Classifier", fontsize=FONTSIZE_LABELS)
                method_plot.set_xlabel(None)
                method_plot.set_xticklabels(
                    ["", "AC1", "AC2", "AC3", "AC4", "AC5", "AC6"],
                    fontsize=FONTSIZE_TICKS - 2,
                )

                if idx == 0:
                    method_plot.set_ylabel(
                        "Accuracy difference", fontsize=FONTSIZE_LABELS
                    )
                else:
                    method_plot.set_ylabel(None)

                method_plot.set_yticklabels(
                    method_plot.get_yticklabels(), fontsize=FONTSIZE_TICKS
                )
                if idx == 3:
                    plot_handles, plot_labels = method_plot.get_legend_handles_labels()
                    plt.legend(
                        plot_handles,
                        plot_labels,
                        title="Task ID",
                        fontsize=FONTSIZE_LEGEND - 3,
                        loc="upper right",
                        ncol=5,
                    )

            fig.tight_layout()
            fig.savefig(
                str(output_dir / f"{setting}_{ee_config}.pdf").replace(" ", "_")
            )

            # fig.savefig(
            #     str(output_dir / f"{setting}_{ee_config}.png").replace(" ", "_")
            # )


if __name__ == "__main__":
    root = Path(__file__).parent.parent.parent.parent
    result_dirs = root / "results_analysis"
    result_paths = sorted(list(result_dirs.glob("CIFAR*/*_sdn*/*/*")))
    output_dir = root / "icml_data" / "ac_acc_analysis"

    plot_ic_accs(result_paths, output_dir)
