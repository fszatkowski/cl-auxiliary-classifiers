import datetime
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd


def _extract_train_time(path):
    with path.open("r") as f:
        txt_lines = f.readlines()
        txt_lines = [l for l in txt_lines if l.startswith("[Elapsed time =")]
        if len(txt_lines) == 0:
            return None
        else:
            time = txt_lines[0].split("[Elapsed time =")[1].split("]")[0].strip()

            if "h" in time:
                time = float(time.split("h")[0].strip()) * 60
            else:
                print(f"Unknown time format: {time}")
                breakpoint()

            return time


def parse_exp_name(path):
    if "ensembling" in path.name:
        setup = "ensemblig"
    elif "cascading" in path.name:
        setup = "cascading"
    elif "weighting" in path.name:
        setup = "weighting"
    elif "dense" in path.name:
        setup = "12AC"
    elif "sparse" in path.name:
        setup = "3AC"
    elif "detach" in path.name:
        setup = "6LP"
    elif "sdn" in path.name:
        setup = "6AC"
    else:
        setup = "base"

    if path.name.startswith("ancl"):
        method = "ANCL"
    elif path.name.startswith("ewc"):
        method = "EWC"
    elif path.name.startswith("der++"):
        method = "DER++"
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

    valid_files = []
    seed_dirs = list(path.glob("seed*"))
    if len(seed_dirs) != 3:
        print(f"Fount {len(seed_dirs)} seed dirs for {str(path)}")

    for seed_dir in seed_dirs:
        stdout_files = list(seed_dir.rglob("stdout*"))

        file_to_time = {}
        for file in stdout_files:
            time = _extract_train_time(stdout_files[0])
            if time is not None:
                file_to_time[file] = time

        if len(file_to_time) == 1:
            valid_files.append(stdout_files[0])
        elif len(file_to_time) > 1:
            keys = list(file_to_time.keys())
            timestamps = [k.name.split("stdout-")[1].split(".")[0] for k in keys]
            # Parse timestamps to comparable objects from original format
            # E.g.: ['2024-09-16-02-09', '2024-07-30-07-17']
            datetimes = [
                datetime.datetime.strptime(t, "%Y-%m-%d-%H-%M") for t in timestamps
            ]
            # Pick the latest filename
            valid_files.append(keys[datetimes.index(max(datetimes))])

    valid_times = [_extract_train_time(file) for file in valid_files]
    if len(valid_times) == 0:
        print("ERROR")

    try:
        mean_time = sum(valid_times) / len(valid_times)
    except Exception:
        print(f"Error for {path}")
        mean_time = -100

    return setup, method, mean_time


def load_data(results_dir, model_name):
    memory_files = list(results_dir.rglob("memory_stats.json"))
    outputs = []
    for file in memory_files:
        setting = file.parent.parent.parent.parent.name
        exp_name = file.parent.parent.parent.name
        if "resnet32_dense" in exp_name:
            setup = "12AC"
        elif "resnet32_sparse" in exp_name:
            setup = "3AC"
        elif "resnet32_sdn" in exp_name:
            setup = "6AC"
        elif "vgg19_medium" in exp_name:
            setup = "10AC"
        else:
            setup = "Base"

        if exp_name.startswith("ancl"):
            method = "ANCL"
        elif exp_name.startswith("ewc"):
            method = "EWC"
        elif exp_name.startswith("der++"):
            method = "DER++"
        elif exp_name.startswith("lwf"):
            method = "LwF"
        elif exp_name.startswith("bic"):
            method = "BiC"
        elif exp_name.startswith("ssil"):
            method = "SSIL"
        elif exp_name.startswith("lode"):
            method = "LODE"
        elif exp_name.startswith("er"):
            method = "ER"
        elif exp_name.startswith("gdumb"):
            method = "GDUMB"
        elif exp_name.startswith("finetuning_ex0"):
            method = "FT"
        elif exp_name.startswith("finetuning_ex2000"):
            method = "FT+Ex"
        else:
            raise ValueError()

        with file.open("r") as f:
            data = json.load(f)

        outputs.append(
            {
                "Setting": setting,
                "Method": method,
                "Setup": setup,
                "Model": model_name,
                "Memory": data["cuda_call_total"] - data["cuda_call_free"],
            }
        )

    return pd.DataFrame(outputs)


if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent.parent.parent
    data_resnet32 = load_data(root_dir / "results_memory_usage", model_name="resnet32")
    data_vgg = load_data(root_dir / "results_memory_vgg19", model_name="vgg19")
    data = pd.concat([data_resnet32, data_vgg])
    data["ACs"] = data["Setup"].map(
        {"Base": 0, "3AC": 3, "6AC": 6, "10AC": 10, "12AC": 12}
    )
    data = data.sort_values(by=["Model", "Setting", "Method", "ACs"], ascending=True)
    output_dir = root_dir / "memory_usage_stats"
    output_dir.mkdir(exist_ok=True, parents=True)

    SETTINGS = ["CIFAR100x5"]
    MODEL_NAMES = ["resnet32", "vgg19"]
    METHODS = [
        "FT",
        "FT+Ex",
        "GDUMB",
        "ANCL",
        "BiC",
        "ER",
        "EWC",
        "LODE",
        "LwF",
        "SSIL",
    ]

    data = data[data["Method"].isin(METHODS)]

    for model_name in MODEL_NAMES:
        for setting in SETTINGS:
            df = data[(data["Model"] == model_name) & (data["Setting"] == setting)]
            # Change df to the format where each row is different setup, each column different method and values correspond to the memory
            df = df[["Setting", "Method", "ACs", "Memory"]]
            df = df.pivot(index="ACs", columns="Method", values="Memory")
            df["Avg"] = df.mean(axis=1)
            df = df.round(2)
            df.to_csv(output_dir / f"{model_name}_{setting}.csv", index=True)
            md_string = df.to_markdown(index=True)
            with open(output_dir / f"{model_name}_{setting}.md", "w+") as f:
                f.write(md_string)
