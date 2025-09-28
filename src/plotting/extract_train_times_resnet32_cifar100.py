import datetime
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

    mean_time = sum(valid_times) / len(valid_times)

    return setup, method, mean_time


if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent.parent.parent
    results_dir = root_dir / "results"
    assert results_dir.exists()
    output_dir = root_dir / "train_times" / "resnet32_cifar100"
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = ["CIFAR100x5", "CIFAR100x10"]

    outputs = []
    for setting in settings:
        for path in results_dir.joinpath(setting).glob("*"):
            if (
                "detach" in path.name
                or "weighting" in path.name
                or "cascading" in path.name
                or "ensembling" in path.name
                or "der++" in path.name
            ):
                continue
            setup, method, mean_time = parse_exp_name(path)
            outputs.append(
                {
                    "Setting": setting,
                    "Method": method,
                    "Setup": setup,
                    "Time": mean_time,
                }
            )

    df = pd.DataFrame(outputs)
    df = df[["Setting", "Method", "Setup", "Time"]]
    df = df.sort_values(by=["Setting", "Method", "Setup"])
    df["Hours"] = round(df["Time"] / 60, 2)
    df.columns = ["Setting", "Method", "Setup", "Time[min]", "Time[h]"]

    df = df[
        (df["Setup"] == "base")
        | (df["Setup"] == "3AC")
        | (df["Setup"] == "6AC")
        | (df["Setup"] == "12AC")
    ]
    df["ACs"] = df["Setup"].map({"base": 0, "3AC": 3, "6AC": 6, "12AC": 12})
    df = df.sort_values(by=["Setting", "Method", "ACs"])
    df = df[["Setting", "Method", "ACs", "Time[h]"]]
    # Compute time overhead compared to the row with the same setting and method but 0 ACs
    time_vals = deepcopy(df["Time[h]"].values)
    for i in range(0, len(time_vals) // 4):
        time_vals[4 * i : 4 * i + 4] = time_vals[4 * i]
    df["Ref time [h]"] = time_vals
    df["Overhead"] = df["Time[h]"] / df["Ref time [h]"] - 1.0
    df["Overhead"] = df["Overhead"].apply(lambda x: "{:.0%}".format(x))
    df = df[["Setting", "Method", "ACs", "Time[h]", "Overhead"]]
    df.to_csv("train_times.csv", index=False)

    for setting in settings:
        setting_df = df[df["Setting"] == setting]
        methods = setting_df["Method"].unique()
        acs = setting_df["ACs"].unique()
        keys = ["ACs"] + methods

        setting_outputs = []
        for ac in acs:
            setting_ac_df = setting_df[setting_df["ACs"] == ac]
            output = {"ACs": ac}
            times_to_avg = []
            for row in setting_ac_df.to_dict("records"):
                time = row["Time[h]"]
                times_to_avg.append(time)
                method = row["Method"]
                output[method] = time
            output["Avg"] = round(sum(times_to_avg) / len(times_to_avg), 2)
            setting_outputs.append(output)
        pd.DataFrame(setting_outputs).round(1).to_csv(
            output_dir / f"{setting}.csv", index=False
        )
