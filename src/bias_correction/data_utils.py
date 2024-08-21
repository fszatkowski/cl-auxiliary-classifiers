from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor


@dataclass
class Data:
    task_id: int
    logits: Tensor
    targets: Tensor


def load_dir(dir_path: Path) -> Tuple[Tensor, Tensor]:
    outputs_tag = []
    targets = []
    logit_files = list(dir_path.glob("*.pt"))
    assert len(logit_files) > 0, f"No logit files found in {dir_path}"

    for p in logit_files:
        data = torch.load(p, map_location="cpu")
        outputs_tag.append(data["outputs_tag"])
        targets.append(data["targets"])
    return torch.cat(outputs_tag, dim=0), torch.cat(targets, dim=0)


def load_data(data_dir: Path) -> Tuple[Data, List[Data]]:
    # Shapes: outputs: [batch_size, n_ics, n_tasks], targets: [batch_size]
    train_data_path = data_dir / "logits_train"
    test_data_path = data_dir / "logits_test"

    train_dir = list(train_data_path.glob("*"))[0]  # We only have last task data
    final_task_id = int(train_dir.name.split("_")[1])
    train_logits, train_targets = load_dir(train_dir)
    train_data = Data(task_id=final_task_id, logits=train_logits, targets=train_targets)

    test_data = []
    test_dirs = sorted(list(test_data_path.glob("*")))
    for test_dir in test_dirs:
        test_logits, test_targets = load_dir(test_dir)
        task_id = int(test_dir.name.split("_")[1])
        test_data.append(
            Data(task_id=task_id, logits=test_logits, targets=test_targets)
        )
    test_data = sorted(test_data, key=lambda x: x.task_id)

    return train_data, test_data


def load_exit_costs(input_dir: Path) -> Tensor:
    data = np.load(input_dir / "results" / "ee_eval.npy", allow_pickle=True).item()[
        "avg"
    ]
    exit_costs = data["exit_costs"] / data["baseline_cost"]
    exit_costs = torch.Tensor(exit_costs)
    return exit_costs
