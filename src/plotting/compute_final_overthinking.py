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


def get_overthinking_data(path: Path, device: str):
    setting, method, ee_config, seed = decode_path(path)
    logits_dir = path / "outputs"
    after_task_dirs = list(logits_dir.glob("after_*"))
    final_dir = max(after_task_dirs, key=lambda x: int(x.stem.split("_")[-1]))
    task_dirs = sorted(list(final_dir.glob("t_*")))

    n_tasks = len(task_dirs)
    if n_tasks == 5:
        task_sizes = [20] * 5
    elif n_tasks == 10:
        task_sizes = [10] * 10
    elif n_tasks == 6:
        task_sizes = [50] + 5 * [10]
    elif n_tasks == 11:
        task_sizes = [50] + 10 * [5]
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
    avg_tag_acc_final = sum(
        [
            outputs[i]["tag_acc_final"] * task_size
            for i, task_size in enumerate(task_sizes)
        ]
    ) / sum(task_sizes)
    avg_tag_acc_any = sum(
        [
            outputs[i]["tag_acc_any"] * task_size
            for i, task_size in enumerate(task_sizes)
        ]
    ) / sum(task_sizes)

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
    return outputs


def compute_overthinking_between_tasks(
    result_paths: List[Path], output_dir: Path, device: str
):
    print("Computing overthinking...")
    print(f"Found {len(result_paths)} results:")
    for result_path in result_paths:
        setting, method, ee_config, seed = decode_path(result_path)
        print(f"{setting}\t{method}\t{ee_config}\t{seed}")

    overthinking_data = []
    for result_path in tqdm(result_paths, desc="Computing overthinking..."):
        overthinking_data.extend(get_overthinking_data(result_path, device))

    df = pd.DataFrame(overthinking_data)
    output_dir.mkdir(exist_ok=True, parents=True)
    df.to_csv(output_dir / "data.csv", index=False)


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
    compute_overthinking_between_tasks(result_paths, output_dir, device=device)
