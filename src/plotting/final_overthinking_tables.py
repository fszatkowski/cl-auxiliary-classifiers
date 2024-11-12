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

METHOD_NAMES = {
    "finetuning_ex0": "FT",
    "finetuning_ex2000": "FT+Ex",
    "joint": "Joint",
    "bic": "BiC",
    "lwf": "LwF",
}


def parse_method_name(method_name):
    for key, value in METHOD_NAMES.items():
        if method_name.startswith(key):
            return value
    raise ValueError("Invalid method name: {}".format(method_name))


def parse_ee_setting(method_name):
    if "detach" in method_name:
        return "LP"
    else:
        return "AC"


def generate_overthinking_tables(input_path: Path, output_dir: Path, device: str):
    df = pd.read_csv(input_path)
    df = df[["setting", "method", "ee_config", "seed", "task_id", "tag_overthinking"]]
    df["cl_method"] = df["method"].apply(lambda x: parse_method_name(x))
    df["ee_setting"] = df["method"].apply(lambda x: parse_ee_setting(x))
    df["tag_overthinking"] = df["tag_overthinking"].astype(float) * 100
    df = df[
        ["setting", "cl_method", "ee_setting", "seed", "task_id", "tag_overthinking"]
    ]
    df = df.sort_values(by=["setting", "cl_method", "ee_setting", "seed", "task_id"])
    df = (
        df.groupby(by=["setting", "cl_method", "ee_setting", "task_id"])
        .agg({"tag_overthinking": ["mean", "std"]})
        .reset_index()
    )
    df.columns = ["Setting", "Method", "EE Setting", "Task ID", "OT(mean)", "OT(std)"]
    df["OT(mean)"] = round(df["OT(mean)"], 2)
    df["OT(std)"] = round(df["OT(std)"], 2)
    df["OT"] = df["OT(mean)"].astype(str) + "$\pm$" + df["OT(std)"].astype(str)
    df = df[["Setting", "Method", "EE Setting", "Task ID", "OT"]]

    settings = df["Setting"].unique()
    ee_settings = df["EE Setting"].unique()

    output_dir.mkdir(exist_ok=True, parents=True)
    for setting, ee_setting in itertools.product(settings, ee_settings):
        tmp_df = df[(df["Setting"] == setting) & (df["EE Setting"] == ee_setting)]
        if len(tmp_df) == 0:
            continue

        tmp_df = tmp_df[["Method", "Task ID", "OT"]]
        tmp_records = tmp_df.to_records(index=False)
        tmp_dicts = [
            {"Method": record[0], "Task ID": record[1], "OT": record[2]}
            for record in tmp_records
        ]

        task_ids = sorted(list(set([d["Task ID"] for d in tmp_dicts])))
        methods = ["FT", "FT+Ex", "LwF", "BiC"]

        outputs = []
        for method in methods:
            output = {"Method": method}
            for task_id in task_ids:
                targets = [
                    d
                    for d in tmp_dicts
                    if d["Method"] == method and d["Task ID"] == task_id
                ]
                assert len(targets) == 1
                overthinking = targets[0]["OT"]
                output[task_id] = overthinking
            outputs.append(output)

        output_path = output_dir / f"{setting}_{ee_setting}.csv"
        pd.DataFrame(outputs).to_csv(output_path, index=False)


if __name__ == "__main__":
    root = Path(__file__).parent.parent.parent
    input_path = root / "analysis_outputs" / "final_overthinking_analysis" / "data.csv"
    output_dir = root / "analysis_outputs" / "final_overthinking_tables"

    if torch.cuda.is_available():
        device = "cuda"
        print("Using GPU")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS")
    else:
        device = "cpu"
        print("Using CPU")
    generate_overthinking_tables(input_path, output_dir, device=device)
