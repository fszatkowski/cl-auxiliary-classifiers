from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

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


@dataclass
class ScoresEE(Scores):
    per_ic_cost: np.ndarray
    per_ic_acc: np.ndarray
    per_th_cost: np.ndarray
    per_th_acc: np.ndarray
    per_th_std: Optional[np.ndarray] = None


def load_data(results_dir: Path, downsample: bool) -> Optional[Scores]:
    ee_scores_path = results_dir / "ee_eval.npy"
    if ee_scores_path.exists():
        return parse_ee_scores(ee_scores_path, downsample=downsample)
    standard_scores_paths = list(results_dir.glob("avg_accs_tag*"))
    if len(standard_scores_paths) > 0:
        return parse_standard_scores(standard_scores_paths[0])
    # No results found
    print(f"No results found in {results_dir}")
    return None


def parse_path(path: Path) -> Metadata:
    setting = path.parent.parent.parent.parent.parent.name
    exp_name = path.parent.parent.parent.parent.name
    seed = int(path.parent.parent.parent.name.replace("seed", ""))
    tag = path.parent.parent.name
    return Metadata(setting=setting, exp_name=exp_name, tag=tag, seed=seed)


def parse_standard_scores(path: Path) -> Scores:
    metadata = parse_path(path)
    with path.open("r") as f:
        tag_accs = f.read().strip().split("\t")
        tag_acc_final = float(tag_accs[-1])
    return ScoresStandard(metadata, early_exit=False, tag_acc_final=tag_acc_final * 100)


def parse_ee_scores(path: Path, downsample: bool) -> Scores:
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

    return ScoresEE(
        metadata,
        early_exit=True,
        per_ic_cost=per_ic_cost,
        per_ic_acc=100 * per_ic_acc,
        per_th_cost=per_th_cost,
        per_th_acc=100 * per_th_acc,
    )


def average_scores(scores: list) -> list:
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
            scores, setting=keys[0], early_exit=keys[1], exp_name=keys[2], tag=keys[3]
        )
        for keys in uniques_keys
    ]
    averaged_scores = [s for s in averaged_scores if s is not None]
    return averaged_scores


def _average_scores(
    scores: list, setting: str, early_exit: bool, exp_name: str, tag: str
) -> Scores:
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
        return ScoresStandard(
            metadata=filtered_scores[0].metadata,
            early_exit=early_exit,
            tag_acc_final=avg_score,
            tag_acc_std=avg_score_std,
        )
    else:  # early exit model
        try:
            avg_per_ic_cost = np.mean([s.per_ic_cost for s in filtered_scores], axis=0)
            avg_per_ic_acc = np.mean([s.per_ic_acc for s in filtered_scores], axis=0)

            avg_per_th_cost = np.mean([s.per_th_cost for s in filtered_scores], axis=0)
            avg_per_th_acc = np.mean([s.per_th_acc for s in filtered_scores], axis=0)
            per_th_std = np.std([s.per_th_acc for s in filtered_scores], axis=0)
            return ScoresEE(
                metadata=filtered_scores[0].metadata,
                early_exit=early_exit,
                per_ic_cost=avg_per_ic_cost,
                per_ic_acc=avg_per_ic_acc,
                per_th_cost=avg_per_th_cost,
                per_th_acc=avg_per_th_acc,
                per_th_std=per_th_std,
            )
        except Exception as e:
            print(e)
            return None


def load_averaged_scores(
    root_dir: Path, downsample: bool = False, filter: Callable = None
) -> list:
    results_paths = Path(root_dir).rglob("results")
    if filter is not None:
        results_paths = [p for p in results_paths if filter(p)]
    parsed_scores = [load_data(path, downsample) for path in results_paths]
    parsed_scores = [p for p in parsed_scores if p is not None]
    averaged_scores = average_scores(parsed_scores)
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
