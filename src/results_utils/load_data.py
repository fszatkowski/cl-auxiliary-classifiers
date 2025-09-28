from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Metadata:
    setting: str
    exp_name: str
    tag: str
    seed: Optional[int] = None


@dataclass
class Scores:
    metadata: Metadata
    early_exit: bool


@dataclass
class ScoresStandard(Scores):
    tag_acc_final: float
    tag_acc_std: Optional[float] = None
    per_task_acc_matrix: Optional[np.ndarray] = None
    per_task_acc_std: Optional[np.ndarray] = None
    per_task_forgetting_matrix: Optional[np.ndarray] = None
    per_task_forgetting_std: Optional[np.ndarray] = None
    num_seeds: int = 1


@dataclass
class ScoresEE(Scores):
    per_ic_cost: np.ndarray
    per_ic_acc: np.ndarray
    per_th_cost: np.ndarray
    per_th_acc: np.ndarray
    per_th_std: Optional[np.ndarray] = None
    per_task_acc_matrix: Optional[np.ndarray] = None
    per_task_acc_std: Optional[np.ndarray] = None
    per_task_forgetting_matrix: Optional[np.ndarray] = None
    per_task_forgetting_std: Optional[np.ndarray] = None
    per_ic_per_task_acc: Optional[np.ndarray] = None
    num_seeds: int = 1


def load_data(
    results_dir: Path, downsample: bool, per_task_acc: bool = False
) -> Optional[Scores]:
    ee_scores_path = results_dir / "ee_eval.npy"
    if ee_scores_path.exists():
        return parse_ee_scores(
            ee_scores_path, downsample=downsample, per_task_acc=per_task_acc
        )
    standard_scores_paths = list(results_dir.glob("avg_accs_tag*"))
    if len(standard_scores_paths) > 0:
        return parse_standard_scores(
            standard_scores_paths[0], per_task_acc=per_task_acc
        )
    # No results found
    print(f"No results found in {results_dir}")
    return None


def parse_path(path: Path) -> Metadata:
    setting = path.parent.parent.parent.parent.parent.name
    exp_name = path.parent.parent.parent.parent.name
    seed = int(path.parent.parent.parent.name.replace("seed", ""))
    tag = path.parent.parent.name
    return Metadata(setting=setting, exp_name=exp_name, tag=tag, seed=seed)


def parse_standard_scores(path: Path, per_task_acc) -> Scores:
    metadata = parse_path(path)
    with path.open("r") as f:
        tag_accs = f.read().strip().split("\t")
        tag_acc_final = float(tag_accs[-1])

    if per_task_acc:
        metrics_path = path.parent / "final_metrics.npy"
        data = np.load(metrics_path, allow_pickle=True).item()
        per_task_acc_matrix = data["acc_tag"]
        per_task_forgetting_matrix = data["forg_tag"]
        per_task_acc_std = None
        per_task_forgetting_std = None
    else:
        per_task_acc_matrix = None
        per_task_acc_std = None
        per_task_forgetting_matrix = None
        per_task_forgetting_std = None

    return ScoresStandard(
        metadata,
        early_exit=False,
        tag_acc_final=tag_acc_final * 100,
        per_task_acc_matrix=per_task_acc_matrix,
        per_task_acc_std=per_task_acc_std,
        per_task_forgetting_matrix=per_task_forgetting_matrix,
        per_task_forgetting_std=per_task_forgetting_std,
    )


def load_per_task_ee_eval(
    ee_eval_paths: List[Path],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_tasks = len(ee_eval_paths)
    per_task_acc_matrix = np.zeros((num_tasks, num_tasks))
    per_task_per_ac_acc_matrix = None
    ee_eval_paths = sorted(ee_eval_paths)
    for task_id, path in enumerate(ee_eval_paths):
        data = np.load(path, allow_pickle=True).item()
        assert (
            len(data.keys()) == task_id + 2
        )  # account for zero-indexing and additional avg label

        if per_task_per_ac_acc_matrix is None:
            per_task_per_ac_acc_matrix = np.zeros(
                (num_tasks, num_tasks, len(data[0]["per_ic_acc"]["tag"]))
            )

        for key in data.keys():
            if key == "avg":
                continue

            per_task_acc_matrix[task_id, key] = data[key]["per_th_acc"]["tag"][-1]
            per_task_per_ac_acc_matrix[task_id, key] = data[key]["per_ic_acc"]["tag"]

    per_task_forgetting_matrix = np.zeros((num_tasks, num_tasks))
    for task_id in range(num_tasks):
        per_task_forgetting_matrix[task_id:, task_id] = (
            per_task_acc_matrix[task_id, task_id]
            - per_task_acc_matrix[task_id:, task_id]
        )

    return per_task_acc_matrix, per_task_forgetting_matrix, per_task_per_ac_acc_matrix


def parse_ee_scores(path: Path, downsample: bool, per_task_acc: bool) -> Scores:
    metadata = parse_path(path)
    data = np.load(path, allow_pickle=True).item()
    data = data["avg"]
    exit_costs = data["exit_costs"]
    baseline_cost = data["baseline_cost"]

    # Compute scores and costs for isolated ICs
    per_ic_cost = exit_costs / baseline_cost
    per_ic_acc = data["per_ic_acc"]["tag"]

    # Compute scores and costs for each threshold
    per_th_acc = data["per_th_acc"]["tag"]
    per_th_exit_cnt = data["per_th_exit_cnt"]["tag"]
    per_th_cost = (per_th_exit_cnt * exit_costs).sum(1) / baseline_cost

    if downsample:
        per_th_cost = per_th_cost[::10]
        per_th_acc = per_th_acc[::10]

    if per_task_acc:
        ee_eval_paths = list(path.parent.glob("ee_eval_*.npy"))
        per_task_acc_matrix, per_task_forgetting_matrix, per_task_per_ac_acc_matrix = (
            load_per_task_ee_eval(ee_eval_paths)
        )
        per_task_acc_std = None
        per_task_forgetting_std = None
    else:
        per_task_acc_matrix = None
        per_task_forgetting_matrix = None
        per_task_acc_std = None
        per_task_forgetting_std = None
        per_task_per_ac_acc_matrix = None

    return ScoresEE(
        metadata,
        early_exit=True,
        per_ic_cost=per_ic_cost,
        per_ic_acc=100 * per_ic_acc,
        per_th_cost=per_th_cost,
        per_th_acc=100 * per_th_acc,
        per_task_acc_matrix=per_task_acc_matrix,
        per_task_acc_std=per_task_acc_std,
        per_task_forgetting_matrix=per_task_forgetting_matrix,
        per_task_forgetting_std=per_task_forgetting_std,
        per_ic_per_task_acc=per_task_per_ac_acc_matrix,
    )


def average_scores(scores: list, per_task_acc: bool) -> list:
    uniques_keys = set(
        (
            scores.metadata.setting,
            scores.early_exit,
            scores.metadata.exp_name,
            scores.metadata.tag,
        )
        for scores in scores
    )
    uniques_keys = sorted(uniques_keys, key=lambda x: (x[0], x[1], x[2], x[3]))

    print()
    print("Parsing results...")
    print()

    averaged_scores = [
        _average_scores(
            scores,
            setting=keys[0],
            early_exit=keys[1],
            exp_name=keys[2],
            tag=keys[3],
            per_task_acc=per_task_acc,
        )
        for keys in uniques_keys
    ]
    averaged_scores = [s for s in averaged_scores if s is not None]
    return averaged_scores


def _average_scores(
    scores: list,
    setting: str,
    early_exit: bool,
    exp_name: str,
    tag: str,
    per_task_acc: bool = False,
) -> Optional[Scores]:
    filtered_scores = [
        s
        for s in scores
        if (
            s.metadata.setting == setting
            and s.early_exit == early_exit
            and s.metadata.exp_name == exp_name
            and s.metadata.tag == tag
        )
    ]
    print(
        f"Found {len(filtered_scores)} for {setting}, {early_exit}, {exp_name}, {tag}"
    )
    if not early_exit:  # standard model
        avg_score = np.mean([s.tag_acc_final for s in filtered_scores])
        avg_score_std = np.std([s.tag_acc_final for s in filtered_scores])

        if per_task_acc:
            stacked_per_task_acc = np.stack(
                [s.per_task_acc_matrix for s in filtered_scores], axis=0
            )
            per_task_acc_matrix = np.mean(stacked_per_task_acc, axis=0)
            per_task_acc_std = np.std(stacked_per_task_acc, axis=0)

            stacked_per_task_forgetting = np.stack(
                [s.per_task_forgetting_matrix for s in filtered_scores], axis=0
            )
            per_task_forgetting_matrix = np.mean(stacked_per_task_forgetting, axis=0)
            per_task_forgetting_std = np.std(stacked_per_task_forgetting, axis=0)
        else:
            per_task_acc_matrix = None
            per_task_acc_std = None
            per_task_forgetting_matrix = None
            per_task_forgetting_std = None

        return ScoresStandard(
            metadata=filtered_scores[0].metadata,
            early_exit=early_exit,
            tag_acc_final=avg_score,
            tag_acc_std=avg_score_std,
            per_task_acc_matrix=per_task_acc_matrix,
            per_task_acc_std=per_task_acc_std,
            per_task_forgetting_matrix=per_task_forgetting_matrix,
            per_task_forgetting_std=per_task_forgetting_std,
            num_seeds=len(filtered_scores),
        )
    else:  # early exit model
        try:
            avg_per_ic_cost = np.mean([s.per_ic_cost for s in filtered_scores], axis=0)
            avg_per_ic_acc = np.mean([s.per_ic_acc for s in filtered_scores], axis=0)

            avg_per_th_cost = np.mean([s.per_th_cost for s in filtered_scores], axis=0)
            avg_per_th_acc = np.mean([s.per_th_acc for s in filtered_scores], axis=0)
            per_th_std = np.std([s.per_th_acc for s in filtered_scores], axis=0)

            if per_task_acc:
                stacked_per_task_acc = np.stack(
                    [s.per_task_acc_matrix for s in filtered_scores], axis=0
                )
                per_task_acc_matrix = np.mean(stacked_per_task_acc, axis=0)
                per_task_acc_std = np.std(stacked_per_task_acc, axis=0)

                stacked_per_task_forgetting = np.stack(
                    [s.per_task_forgetting_matrix for s in filtered_scores], axis=0
                )
                per_task_forgetting_matrix = np.mean(
                    stacked_per_task_forgetting, axis=0
                )
                per_task_forgetting_std = np.std(stacked_per_task_forgetting, axis=0)
                per_ic_per_task_acc = np.stack(
                    [s.per_ic_per_task_acc for s in filtered_scores], axis=0
                ).mean(axis=0)
            else:
                per_task_acc_matrix = None
                per_task_acc_std = None
                per_task_forgetting_matrix = None
                per_task_forgetting_std = None
                per_ic_per_task_acc = None

            return ScoresEE(
                metadata=filtered_scores[0].metadata,
                early_exit=early_exit,
                per_ic_cost=avg_per_ic_cost,
                per_ic_acc=avg_per_ic_acc,
                per_th_cost=avg_per_th_cost,
                per_th_acc=avg_per_th_acc,
                per_th_std=per_th_std,
                per_task_acc_matrix=per_task_acc_matrix,
                per_task_acc_std=per_task_acc_std,
                per_task_forgetting_matrix=per_task_forgetting_matrix,
                per_task_forgetting_std=per_task_forgetting_std,
                per_ic_per_task_acc=per_ic_per_task_acc,
                num_seeds=len(filtered_scores),
            )
        except Exception as e:
            print(e)
            return None


def load_averaged_scores(
    root_dir: Path,
    downsample: bool = False,
    filter: Callable = None,
    per_task_acc: bool = False,
) -> list:
    results_paths = Path(root_dir).rglob("results")
    results_paths = sorted(results_paths)
    if filter is not None:
        results_paths = [p for p in results_paths if filter(p)]
    parsed_scores = [
        load_data(path, downsample, per_task_acc) for path in results_paths
    ]
    parsed_scores = [p for p in parsed_scores if p is not None]

    averaged_scores = average_scores(parsed_scores, per_task_acc=per_task_acc)
    print(f"Parsed {len(averaged_scores)} scores")
    return averaged_scores


def parse_score(score: Scores) -> Dict:
    if isinstance(score, ScoresEE):
        return {
            "setting": score.metadata.setting,
            "exp_name": score.metadata.exp_name,
            "early_exit": score.early_exit,
            "seed": score.metadata.seed,
            "acc": score.per_th_acc[-1],
        }
    elif isinstance(score, ScoresStandard):
        return {
            "setting": score.metadata.setting,
            "exp_name": score.metadata.exp_name,
            "early_exit": score.early_exit,
            "seed": score.metadata.seed,
            "acc": score.tag_acc_final,
        }
    else:
        raise NotImplementedError()


def load_scores_for_table(root_dir: Path, filter: Callable = None) -> pd.DataFrame:
    results_paths = Path(root_dir).rglob("results")
    if filter is not None:
        results_paths = [p for p in results_paths if filter(p)]
    parsed_scores = [load_data(path, downsample=False) for path in results_paths]
    parsed_scores = [p for p in parsed_scores if p is not None]
    output_dicts = [parse_score(score) for score in parsed_scores]
    return pd.DataFrame(output_dicts)
