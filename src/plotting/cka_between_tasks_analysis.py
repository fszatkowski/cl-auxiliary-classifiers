from math import ceil
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from metrics import cka, _CKA
from plotting.utils import decode_path

sns.set_style("whitegrid")

AVG_TASK_NAME = "Avg"
CMAP = "Blues"


def plot_feature_cka(
        data: np.ndarray, output_path: Path, title: str = None
):
    plt.cla()
    plt.clf()
    plt.figure()

    nan_mask = np.isnan(data)
    if sum(sum(nan_mask)) > 0:
        print(f"WARNING: NaN values in CKA matrix")
        data[nan_mask] = 0

    # Highlight nonexistent data
    data[data == -1] = np.nan

    plot = sns.heatmap(data, cmap=CMAP, vmin=0, vmax=1, annot=True, fmt=".2f")
    plot.set_title(title)
    plot.set_ylabel("Classifier index")
    plot.set_xlabel("Current task index")

    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plot.get_figure().savefig(str(output_path))


def compute_cka(path: Path, device: str):
    data_dir = path / 'outputs'
    assert data_dir.exists(), f"Data directory does not exist: {data_dir}"
    post_task_dirs = sorted(list(data_dir.glob('after_*')))
    assert len(post_task_dirs) > 0, 'No post-task directories found'
    task_ids = sorted([int(p.stem.split('_')[-1]) for p in post_task_dirs])

    cka_data = {}
    indices = tqdm([(task_data_id, after_task_id) for task_data_id in range(0, len(task_ids)) for after_task_id in
                    range(task_data_id + 1, len(task_ids))], desc="Computing CKA...")
    for task_data_id, after_task_id in indices:
        prev_features_paths = sorted(
            list((data_dir / f'after_{after_task_id - 1}' / f't_{task_data_id}').glob('*.pt')))
        cur_features_paths = sorted(list((data_dir / f'after_{after_task_id}' / f't_{task_data_id}').glob('*.pt')))
        assert len(prev_features_paths) == len(cur_features_paths), "Different number of batches found"

        sample_features = torch.load(prev_features_paths[0], map_location="cpu")['features']
        if isinstance(sample_features, torch.Tensor):
            ee = False
        elif isinstance(sample_features, list):
            n_cls = len(sample_features)
            ee = True
        else:
            raise NotImplementedError(f"Unknown features type: {type(sample_features)}")

        if not ee:
            # No early exits
            features_prev = []
            features_cur = []
            for batch_idx in range(len(prev_features_paths)):
                features_prev_task = torch.load(prev_features_paths[batch_idx], map_location=device)['features']
                features_cur_task = torch.load(cur_features_paths[batch_idx], map_location=device)['features']
                features_prev.append(features_prev_task)
                features_cur.append(features_cur_task)
            prev_features = torch.cat(features_prev, dim=0)
            cur_features = torch.cat(features_cur, dim=0)
            prev_features = prev_features.view(prev_features.shape[0], -1)
            cur_features = cur_features.view(cur_features.shape[0], -1)
            cka_data[(0, after_task_id, task_data_id)] = _CKA(prev_features, cur_features)
        else:
            for cls_ix in range(n_cls):
                features_prev = []
                features_cur = []
                for batch_idx in range(len(prev_features_paths)):
                    features_prev_task = torch.load(prev_features_paths[batch_idx], map_location="cpu")['features'][
                        cls_ix]
                    features_cur_task = torch.load(cur_features_paths[batch_idx], map_location="cpu")['features'][
                        cls_ix]
                    features_prev.append(features_prev_task)
                    features_cur.append(features_cur_task)
                prev_features = torch.cat(features_prev, dim=0)
                cur_features = torch.cat(features_cur, dim=0)
                prev_features = prev_features.view(prev_features.shape[0], -1)
                cur_features = cur_features.view(cur_features.shape[0], -1)
                cka_data[(cls_ix, after_task_id, task_data_id)] = _CKA(prev_features, cur_features)

    return cka_data


def plot_cka_between_tasks(result_paths: List[Path], output_dir: Path, device: str):
    print("Plotting CKA plots...")
    print(f"Found {len(result_paths)} results:")
    for result_path in result_paths:
        setting, method, ee_config, seed = decode_path(result_path)
        print(f"{setting}\t{method}\t{ee_config}\t{seed}")

    cka_data_to_avg = {}
    for result_path in tqdm(result_paths, desc="Parsing results data..."):
        setting, method, ee_config, seed = decode_path(result_path)
        cka = compute_cka(result_path, device=device)
        n_cls = max(k[0] for k in cka.keys()) + 1
        n_tasks = max(k[1] for k in cka.keys()) + 1
        cka_matrix = torch.ones((n_cls, n_tasks, n_tasks)) * -1
        for (cls_idx, after_task_idx, task_idx), _cka in cka.items():
            cka_matrix[cls_idx, after_task_idx, task_idx] = _cka
        if (setting, method, ee_config) not in cka_data_to_avg:
            cka_data_to_avg[(setting, method, ee_config)] = [cka_matrix]
        else:
            cka_data_to_avg[(setting, method, ee_config)].append(cka_matrix)

    avg_cka_data = {}
    for (setting, method, ee_config), cka_matrices in cka_data_to_avg.items():
        cka_matrices = torch.stack(cka_matrices, dim=0)
        mean_matrix = cka_matrices.mean(dim=0)
        avg_cka_data[(setting, method, ee_config)] = mean_matrix

    output_dir.mkdir(exist_ok=True, parents=True)
    for (setting, method, ee_config), cka_data in tqdm(avg_cka_data.items(), desc="Plotting..."):
        plot_output_dir = output_dir / f"{setting}" / f"{method}_{ee_config}"
        plot_output_dir.mkdir(exist_ok=True, parents=True)
        n_tasks = cka_data.shape[2]
        for task_id in range(n_tasks):
            plot_feature_cka(
                data=cka_data[:, :, task_id].detach().cpu().numpy(),
                output_path=plot_output_dir / f"task_{task_id}.png",
                title=f'{setting} | {method}_{ee_config} | Task {task_id} data',
            )


if __name__ == "__main__":
    root = Path(__file__).parent.parent.parent
    result_dirs = root / "results_analysis"
    result_paths = sorted(list(result_dirs.glob("CIFAR*/*/*/*/outputs")))
    result_paths = [p.parent for p in result_paths]
    output_dir = root / "analysis_outputs" / "cka_between_tasks"

    if torch.cuda.is_available():
        device = "cuda"
        print("Using GPU")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS")
    else:
        device = "cpu"
        print("Using CPU")
    plot_cka_between_tasks(result_paths, output_dir, device=device)
