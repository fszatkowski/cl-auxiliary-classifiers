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

AVG_TASK_NAME = "Avg"
WAVG_TASK_NAME = "WAvg"
CMAP = "Blues"


def plot_overthinking(
    data: np.ndarray,
    ykey: str,
    output_path: Path,
    title: str = None,
    plot_avg: bool = True,
    plot_wavg: bool = False,
):
    plt.cla()
    plt.clf()
    plt.figure()

    if not plot_avg:
        data = data[~(data["task_id"] == AVG_TASK_NAME)]
    if not plot_wavg:
        data = data[~(data["task_id"] == WAVG_TASK_NAME)]

    plot = sns.barplot(data, x="task_id", y=ykey, hue="method")
    plot.set_title(title)
    plot.set_ylabel("Overthinking")
    plot.set_xlabel("Task ID")

    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plot.get_figure().savefig(str(output_path))


def get_overthinking_data(path: Path, device: str):
    setting, method, ee_config, seed = decode_path(path)
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
    outputs.append(output_avg_acc)

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
        "task_id": AVG_TASK_NAME,
        "tag_acc_final": wavg_tag_acc_final,
        "tag_acc_any": wavg_tag_acc_any,
        "tag_overthinking": wavg_tag_acc_any - wavg_tag_acc_final,
        "total": total_samples,
    }
    outputs.append(output_wavg_acc)

    return outputs


def plot_overthinking_between_tasks(
    result_paths: List[Path], output_dir: Path, device: str
):
    print("Plotting overthinking...")
    print(f"Found {len(result_paths)} results:")
    for result_path in result_paths:
        setting, method, ee_config, seed = decode_path(result_path)
        print(f"{setting}\t{method}\t{ee_config}\t{seed}")

    overthinking_data = []
    for result_path in tqdm(result_paths, desc="Computing overthinking..."):
        overthinking_data.extend(get_overthinking_data(result_path, device))

    df = pd.DataFrame(overthinking_data)
    settings = df["setting"].unique()
    methods = df["method"].unique()
    ee_configs = df["ee_config"].unique()
    iterator = tqdm(
        list(itertools.product(settings, methods, ee_configs)), desc="Plotting..."
    )

    output_dir.mkdir(exist_ok=True, parents=True)
    for setting, method, ee_config in iterator:
        tmp_df = df[
            (df["setting"] == setting)
            & (df["method"] == method)
            & (df["ee_config"] == ee_config)
        ]
        if len(tmp_df) == 0:
            continue

        for acc in ["TAg"]:
            plot_output_dir = output_dir / acc.lower() / f"{setting}"
            plot_output_dir.mkdir(exist_ok=True, parents=True)
            plot_overthinking(
                data=tmp_df,
                output_path=plot_output_dir / f"{method}_{ee_config}.png",
                ykey=f"{acc.lower()}_overthinking",
                title=f"{setting} | {method}_{ee_config} | Overthinking {acc}",
                plot_avg=True,
                plot_wavg=False,
            )


if __name__ == "__main__":
    root = Path(__file__).parent.parent.parent
    result_dirs = root / "results_analysis"
    result_paths = sorted(list(result_dirs.glob("CIFAR*/*_sdn*/*/*")))
    output_dir = root / "analysis_outputs" / "final_overthinking_analysis"

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
