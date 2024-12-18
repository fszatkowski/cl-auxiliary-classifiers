from pathlib import Path

import seaborn as sns
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

VMIN = 0
VMAX = 100

TICKS_FONTSIZE = 10
LABEL_FONTSIZE = 12
TITLE_FONTSIZE = 14
ANNOT_FONTSIZE = 8


CMAP_IC = "YlOrRd"
CMAP_ACC = "YlGn"


def plot_heatmap(data, ax, title, ylabels=True, cmap="YlGnBu", vmax=VMAX, vmin=VMIN):
    heatmap = sns.heatmap(
        data,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        ax=ax,
        cbar=False,
        annot=True,
        fmt=".1f",
        annot_kws={"fontsize": ANNOT_FONTSIZE},
    )
    heatmap.set_title(title)

    num_acs = data.shape[0]
    num_tasks = data.shape[1] - 1
    task_labels = [str(task_id + 1) for task_id in range(num_tasks)] + ["Avg"]
    heatmap.set_xticklabels(task_labels, fontsize=TICKS_FONTSIZE)
    if ylabels:
        ic_labels = ["L1.B3", "L1.B5", "L2.B2", "L2.B4", "L3.B1", "L3.B3", "Final"]
        heatmap.set_yticklabels(ic_labels, fontsize=TICKS_FONTSIZE)
    else:
        heatmap.set_yticks([])
        heatmap.set_yticklabels([])

    heatmap.set_xlabel("Task ID", fontsize=LABEL_FONTSIZE)
    heatmap.set_ylabel(None)
    heatmap.set_title(title, fontsize=TITLE_FONTSIZE)


if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent.parent
    output_dir = root_dir / "analysis_outputs/decision_masks/plots"
    masks_path = root_dir / "analysis_outputs/decision_masks/decision_masks.pt"
    output_dir.mkdir(exist_ok=True, parents=True)

    masks_data = torch.load(masks_path)
    setting_method_to_data = {
        (data["setting"], data["method"]): {
            "ic_selection": data["ic_selection"],
            "accs": data["accs"],
        }
        for data in masks_data
    }

    methods = set(k[1] for k in setting_method_to_data.keys())
    settings = ["CIFAR100x5", "CIFAR100x10"]
    for method in methods:
        plt.cla()
        plt.clf()
        fig, ax = plt.subplots(ncols=4, figsize=(18, 5))

        data_5 = setting_method_to_data[("CIFAR100x5", method)]
        data_10 = setting_method_to_data[("CIFAR100x10", method)]

        plot_heatmap(
            data_5["ic_selection"],
            ax[0],
            f"CIFAR100x5, {method} | Classifier Selection",
            cmap=CMAP_IC,
            vmax=60,
        )
        plot_heatmap(
            data_5["accs"],
            ax[1],
            f"CIFAR100x5, {method} | Accuracy",
            ylabels=False,
            cmap=CMAP_ACC,
            vmax=85,
        )

        plot_heatmap(
            data_10["ic_selection"],
            ax[2],
            f"CIFAR100x10, {method} | Classifier Selection",
            ylabels=False,
            cmap=CMAP_IC,
            vmax=60,
        )
        plot_heatmap(
            data_10["accs"],
            ax[3],
            f"CIFAR100x10, {method} | Accuracy",
            ylabels=False,
            cmap=CMAP_ACC,
            vmax=85,
        )
        plt.tight_layout()
        fig.savefig(output_dir / f"{method}.pdf")
