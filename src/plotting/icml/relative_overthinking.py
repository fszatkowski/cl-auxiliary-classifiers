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
from plotting.icml.common import *
from plotting.utils import decode_path

sns.set_style("whitegrid")

AVG_TASK_NAME = "Avg"


def get_overthinking_data(path: Path, device: str):
    setting, method, ee_config, seed = decode_path(path)
    logits_dir = path / "outputs"
    after_task_dirs = list(logits_dir.glob("after_*"))
    final_dir = max(after_task_dirs, key=lambda x: int(x.stem.split("_")[-1]))
    task_dirs = sorted(list(final_dir.glob("t_*")))

    outputs = []
    for task_id in range(len(task_dirs)):
        batches_paths = sorted(list((final_dir / f"t_{task_id}").glob("*.pt")))

        hits_tag_overthinking = None
        total_overthought_samples = 0

        for batch in batches_paths:
            data = torch.load(batch, map_location=device)

            preds_tag = torch.stack(
                [torch.cat(ic_data, dim=-1) for ic_data in data["logits"]], dim=1
            ).argmax(dim=-1)
            targets_tag = data["targets"]
            hits_tag = (preds_tag == targets_tag.unsqueeze(1))
            final_classifier_hits = hits_tag[:, -1]
            ac_hits = hits_tag[:, :-1]
            any_ac_hits = ac_hits.sum(1) > 0
            overthinking_mask = any_ac_hits & ~final_classifier_hits
            per_cls_overthinking = ac_hits[overthinking_mask].sum(0)

            if hits_tag_overthinking is None:
                hits_tag_overthinking = per_cls_overthinking
            else:
                hits_tag_overthinking += per_cls_overthinking
            total_overthought_samples += overthinking_mask.sum()

        for ac_idx, cls_hits in enumerate(hits_tag_overthinking):
            per_ac_overthinking_hits = int(cls_hits)
            per_ac_overthinking_perc = per_ac_overthinking_hits / int(total_overthought_samples)
            outputs.append(
                {
                    "setting": setting,
                    "method": method,
                    "ee_config": ee_config,
                    "seed": seed,
                    "task_id": task_id,
                    "ac_idx": ac_idx,
                    "overthinking_hits": per_ac_overthinking_hits,
                    "overthinking_perc": per_ac_overthinking_perc,
                    "total_samples": int(total_overthought_samples)                }
        )

    unique_acs = set([o["ac_idx"] for o in outputs])
    for ac_idx in unique_acs:
        outputs_ac = [o for o in outputs if o["ac_idx"] == ac_idx]
        ac_total_samples = sum([o["total_samples"] for o in outputs_ac])
        ac_overthinking_hits = sum([o["overthinking_hits"] for o in outputs_ac])
        ac_overthinking_perc = ac_overthinking_hits / ac_total_samples

        output_ac_avg = {
            "setting": setting,
            "method": method,
            "ee_config": ee_config,
            "seed": seed,
            "task_id": AVG_TASK_NAME,
            "ac_idx": ac_idx,
            "overthinking_hits": ac_overthinking_hits,
            "overthinking_perc": ac_overthinking_perc,
            "total_samples": ac_total_samples
        }

        outputs.append(output_ac_avg)

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

    artifact_path = output_dir.parent.joinpath('artifacts').joinpath("relative_overthinking_data.csv")
    artifact_path.parent.mkdir(exist_ok=True, parents=True)
    if artifact_path.exists():
        df = pd.read_csv(artifact_path)
    else:
        overthinking_data = []
        for result_path in tqdm(result_paths, desc="Computing overthinking..."):
            overthinking_data.extend(get_overthinking_data(result_path, device))

        df = pd.DataFrame(overthinking_data)
        df.to_csv(artifact_path, index=False)

    df["cl_method"] = df.apply(method_fn, axis=1)
    df["setup"] = df["method"].apply(setup_fn)
    df['overthinking_perc'] *= 100

    # for setup in ["LP", "AC"]:
    #     plt.cla()
    #     plt.clf()
    #     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
    #     for setting_idx, setting in enumerate(["CIFAR100x5", "CIFAR100x10"]):
    #         tmp_df = df[(df["setting"] == setting) & (df["setup"] == setup)]
    #         tmp_df = tmp_df[tmp_df["task_id"] == AVG_TASK_NAME]
    #         tmp_df = tmp_df.sort_values(by=['cl_method'], key=lambda x: x.map({'Joint': 0, 'FT': 1, 'FT+Ex': 2, 'LwF': 3, "BiC": 4}), ascending=True)
    #
    #         plot = sns.barplot(
    #             tmp_df,
    #             ax=axes[setting_idx],
    #             x="ac_idx",
    #             y="overthinking_perc",
    #             hue="cl_method",
    #             palette=METHOD_TO_COLOR,
    #             legend=False
    #         )
    #
    #         plot.set_title(None)
    #         plot.set_xlabel(None)
    #         if setting_idx == 0:
    #             plot.set_ylabel("Miscls. samples [%]", fontsize=FONTSIZE_LABELS)
    #         else:
    #             plot.set_ylabel("")
    #         plot.set_xticklabels(["L1.B3", "L1.B5", "L2.B2", "L2.B4", "L3.B1", "L3.B3"], fontsize=FONTSIZE_TICKS-4)
    #         plot.set_yticklabels(plot.get_yticklabels(), fontsize=FONTSIZE_TICKS)
    #         # if setting_idx == 1:
    #         #     # Make legend without title
    #         #     handles, labels = plot.get_legend_handles_labels()
    #         #     plot.legend(
    #         #         handles=handles,
    #         #         labels=labels,
    #         #         title=None,
    #         #         fontsize=FONTSIZE_LEGEND-4,
    #         #     )
    #     plt.tight_layout()
    #     fig.savefig(output_dir.joinpath(f"rel_overthinking_{setup}.pdf"))
    #     # fig.savefig(output_dir.joinpath(f"rel_overthinking_{setup}.png"))

    for setup in ["LP", "AC"]:
        for setting in ["CIFAR100x5", "CIFAR100x10"]:
            # Close everything

            plt.cla()
            plt.clf()
            plt.figure()

            tmp_df = df[(df["setting"] == setting) & (df["setup"] == setup)]
            tmp_df = tmp_df[tmp_df["task_id"] == AVG_TASK_NAME]
            tmp_df = tmp_df.sort_values(by=['cl_method'], key=lambda x: x.map({'Joint': 0, 'FT': 1, 'FT+Ex': 2, 'LwF': 3, "BiC": 4}), ascending=True)

            plot = sns.barplot(
                tmp_df,
                x="ac_idx",
                y="overthinking_perc",
                hue="cl_method",
                palette=METHOD_TO_COLOR,
                legend=False
            )

            # plot.set_title(f"{setting} | {setup}", fontsize=FONTSIZE_TITLE)
            plot.set_title(None)
            plot.set_xlabel(None)
            plot.set_ylabel("Miscls. samples [%]", fontsize=FONTSIZE_LABELS)

            plot.set_xticklabels(["AC1", "AC2", "AC3", "AC4", "AC5", "AC6"], fontsize=FONTSIZE_TICKS)
            plot.set_yticklabels(plot.get_yticklabels(), fontsize=FONTSIZE_TICKS)
            # if setting_idx == 1:
            #     # Make legend without title
            #     handles, labels = plot.get_legend_handles_labels()
            #     plot.legend(
            #         handles=handles,
            #         labels=labels,
            #         title=None,
            #         fontsize=FONTSIZE_LEGEND-4,
            #     )
            plt.tight_layout()
            plot.get_figure().savefig(output_dir.joinpath(f"rel_overthinking_{setting}_{setup}.pdf"))
            # plot.get_figure().savefig(output_dir.joinpath(f"rel_overthinking_{setting}_{setup}.png"))

if __name__ == "__main__":
    root = Path(__file__).parent.parent.parent.parent
    result_dirs = root / "results_analysis"
    result_paths = (
         sorted(list(result_dirs.glob("CIFAR100x5/*_sdn*/*/*")))
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
