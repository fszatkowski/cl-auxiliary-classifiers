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
    plot.set_title(title)
    plot.set_xlabel("Classifier Index")
    plot.set_ylabel("Accuracy")

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
    iterator = tqdm(
        [(s, m, ee) for s in settings for m in methods for ee in ee_configs],
        desc="Plotting...",
    )
    for setting, method, ee_config in iterator:
        try:
            df_tmp = df[
                (df["setting"] == setting)
                & (df["method"] == method)
                & (df["ee_config"] == ee_config)
                ]
            df_tmp_taw = df_tmp[df_tmp["acc_type"] == "taw"]
            df_tmp_tag = df_tmp[df_tmp["acc_type"] == "tag"]

            output_path_tag = (
                    output_dir / "tag" / f"{setting}" / f"{method}_{ee_config}.png"
            )
            output_path_taw = (
                    output_dir / "taw" / f"{setting}" / f"{method}_{ee_config}.png"
            )
            plot_per_ic_acc(
                df_tmp_tag,
                output_path_tag,
                plot_avg=True,
                title=f"{setting} | {method}_{ee_config} | TAg",
            )
            plot_per_ic_acc(
                df_tmp_taw,
                output_path_taw,
                plot_avg=True,
                title=f"{setting} | {method}_{ee_config} | TAW",
            )
        except Exception as e:
            print(
                "Encountered error for setting, method, ee_config:",
                setting,
                method,
                ee_config,
            )
            print(e)


if __name__ == "__main__":
    root = Path(__file__).parent.parent.parent
    result_dirs = root / "results"
    # results = sorted(list(result_dirs.glob('CIFAR*/finetuning*_sdn/*/*')) + list(result_dirs.glob('CIFAR*/finetuning*0/*/*')))
    result_paths = sorted(list(result_dirs.glob("CIFAR*/*_sdn/*/*")))
    output_dir = root / "analysis_outputs" / "ic_acc"

    plot_ic_accs(result_paths, output_dir)
