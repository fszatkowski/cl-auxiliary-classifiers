from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import Tensor

from bias_correction.methods.base import BiasCorrection


def plot_biases(bias_correctors: Dict[str, BiasCorrection], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for method_name, bias_corrector in bias_correctors.items():
        assert bias_corrector.get_bias_matrix().shape == (
            bias_corrector.n_cls,
            bias_corrector.n_tasks,
        ), (
            f"Incorrect shape of bias for method {method_name}"
            f" {bias_corrector.get_bias_matrix().shape} != {(bias_corrector.n_cls, bias_corrector.n_tasks)}"
        )
        plt.figure()
        plt.cla()
        plt.clf()
        bias_matrix = bias_corrector.get_bias_matrix()
        plot = sns.heatmap(bias_matrix.T, fmt=".2f", annot=True, cmap="Blues")
        plot.set_xlabel("IC")
        plot.set_ylabel("Task")
        plot.get_figure().savefig(str(output_dir / f"bias_{method_name}.png"))


def plot_cost_vs_acc_comparison(
    method_to_path: Dict[str, Path], output_path: Path
) -> None:
    assert len(method_to_path) > 0

    plt.clf()
    plt.cla()
    plt.figure()

    for method_name, path in method_to_path.items():
        data = torch.load(path)
        cost = data["cost_per_th"].tolist()
        acc = data["acc_per_th"].tolist()
        plot = sns.lineplot(x=cost, y=acc, label=method_name)

    plot.set_xlabel("Cost")
    plot.set_xlabel("Accuracy")
    plot.get_figure().savefig(str(output_path))


def plot_ic_stats(method_to_path: Dict[str, Path], output_dir: Path) -> None:
    assert len(method_to_path) > 0
    output_dir.mkdir(parents=True, exist_ok=True)

    for method_name, path in method_to_path.items():
        data = torch.load(path)
        plot_heatmap(
            data["per_ic_acc"],
            output_dir / f"acc_{method_name}.png",
            title=f"Accuracy ({method_name})",
        )
        plot_heatmap(
            data["per_ic_avg_confidence"],
            output_dir / f"avg_conf_{method_name}.png",
            title=f"Average Confidence ({method_name})",
        )
        plot_heatmap(
            data["per_ic_correct_pred_confidence"],
            output_dir / f"corr_conf_{method_name}.png",
            title=f"Correct Confidence ({method_name})",
        )
        plot_heatmap(
            data["per_ic_incorrect_pred_confidence"],
            output_dir / f"incorr_conf_{method_name}.png",
            title=f"Incorrect Confidence ({method_name})",
        )


def plot_heatmap(
    data: Tensor,
    output_path: Path,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
):
    plt.clf()
    plt.cla()
    plt.figure()
    plot = sns.heatmap(data, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
    plot.get_figure().savefig(str(output_path))
    if title is not None:
        plot.set_title(title)
    if xlabel is not None:
        plot.set_xlabel(xlabel)
    if ylabel is not None:
        plot.set_ylabel(ylabel)
    plot.get_figure().savefig(str(output_path))
