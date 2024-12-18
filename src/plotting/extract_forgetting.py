import datetime
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd


def _extract_forgetting(path):
    with path.open("r") as f:
        txt = f.read()
        accs = [float(f) for f in txt.strip().split("\t")]
        forgetting = 100 * (accs[0] - accs[-1]) / accs[0]
        return forgetting


def parse_exp_name(path):
    if "ensembling" in path.name:
        setup = "ensemblig"
    elif "cascading" in path.name:
        setup = "cascading"
    elif "dense" in path.name:
        setup = "12AC"
    elif "sparse" in path.name:
        setup = "3AC"
    elif "detach" in path.name:
        setup = "detach"
    elif "sdn" in path.name:
        setup = "ACs"
    else:
        setup = "Base"

    if path.name.startswith("ancl"):
        method = "ANCL"
    elif path.name.startswith("ewc"):
        method = "EWC"
    elif path.name.startswith("lwf"):
        method = "LwF"
    elif path.name.startswith("bic"):
        method = "BiC"
    elif path.name.startswith("ssil"):
        method = "SSIL"
    elif path.name.startswith("lode"):
        method = "LODE"
    elif path.name.startswith("er"):
        method = "ER"
    elif path.name.startswith("gdumb"):
        method = "GDUMB"
    elif path.name.startswith("finetuning_ex0"):
        method = "FT"
    elif path.name.startswith("finetuning_ex2000"):
        method = "FT+Ex"
    else:
        raise ValueError()

    seed_dirs = list(path.glob("seed*"))
    if len(seed_dirs) != 3:
        print(f"Fount {len(seed_dirs)} seed dirs for {str(path)}")

    cls_to_forgetting = {}

    for seed_dir in seed_dirs:
        avg_acc_files = list(seed_dir.rglob("avg_accs_tag*"))

        cls_to_acc_files = {}
        for file in avg_acc_files:
            if file.parent.name.startswith("ic"):
                if file.parent.parent.name not in cls_to_acc_files:
                    cls_to_acc_files[file.parent.name] = []
                cls_to_acc_files[file.parent.name].append(file)
            else:
                if "final" not in file.parent.parent.name:
                    cls_to_acc_files["final"] = []
                cls_to_acc_files["final"].append(file)

        for cls in cls_to_acc_files:
            if cls not in cls_to_forgetting:
                cls_to_forgetting[cls] = []

            cls_filenames = sorted(cls_to_acc_files[cls])
            if len(cls_filenames) == 1:
                forgetting = _extract_forgetting(cls_filenames[0])
            elif len(cls_filenames) > 1:
                timestamps = [
                    k.name.split("avg_accs_tag-")[1].split(".")[0]
                    for k in cls_filenames
                ]
                datetimes = [
                    datetime.datetime.strptime(t, "%Y-%m-%d-%H-%M") for t in timestamps
                ]
                # Pick the latest filename
                latest_file = cls_filenames[datetimes.index(max(datetimes))]
                forgetting = _extract_forgetting(latest_file)
            else:
                continue
            cls_to_forgetting[cls].append(forgetting)

    mean_forgetting = {}
    std_forgetting = {}
    for key, forgetting_vals in cls_to_forgetting.items():
        forgetting_vals = np.array(forgetting_vals)
        mean_forgetting[key] = round(forgetting_vals.mean(), 2)
        std_forgetting[key] = round(forgetting_vals.std(), 2)

    return setup, method, mean_forgetting, std_forgetting


if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent.parent
    results_dir = root_dir / "results"

    settings = ["CIFAR100x5", "CIFAR100x10"]

    outputs = []
    for setting in settings:
        for path in results_dir.joinpath(setting).glob("*"):
            setup, method, mean_forg, std_forg = parse_exp_name(path)
            output = {
                "Setting": setting,
                "Method": method,
                "Setup": setup,
            }
            for key in mean_forg:
                output[f"{key}"] = f"{mean_forg[key]}\pm{std_forg[key]}"
            outputs.append(output)

    df = pd.DataFrame(outputs)
    df = df[(df["Setup"] == "Base") | (df["Setup"] == "ACs")]

    df = df[
        [
            "Setting",
            "Method",
            "Setup",
            "ic0",
            "ic1",
            "ic2",
            "ic3",
            "ic4",
            "ic5",
            "final",
        ]
    ]
    df.columns = [
        "Setting",
        "Method",
        "Setup",
        "AC1",
        "AC2",
        "AC3",
        "AC4",
        "AC5",
        "AC6",
        "Final",
    ]
    df = df.sort_values(by=["Setting", "Method", "Setup"])
    df.to_csv("forgetting.csv", index=False)
