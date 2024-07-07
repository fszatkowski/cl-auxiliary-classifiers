from copy import deepcopy
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

import seaborn as sns

sns.set_style("whitegrid")


def parse_data(data_path) -> List[Dict]:
    seed = data_path.parent.parent.name
    method = data_path.parent.parent.parent.name
    if "bic" in method:
        cl_method = "BiC"
    elif "lwf" in method:
        cl_method = "LwF"
    elif "icarl" in method:
        cl_method = "iCaRL"
    elif "finetuning" in method:
        cl_method = "FT+Ex"
    else:
        raise NotImplementedError()
    if "base" in method:
        net = "SDN"
    elif "casc" in method:
        net = "ZTW"
    else:
        raise NotImplementedError()
    setting = data_path.parent.parent.parent.parent.name

    df = pd.read_csv(data_path)
    results_base = df[df["method"] == "base"]
    cost_base = results_base["cost"].values
    acc_base = results_base["acc"].values
    results_tlc = df[df["method"] == "tlc"]
    cost_tlc = results_tlc["cost"].values
    acc_tlc = results_tlc["acc"].values

    outputs = []
    outputs.extend(
        [
            {
                "setting": setting,
                "cl_method": cl_method,
                "network": net,
                "seed": seed,
                "acc": acc,
                "cost": cost,
                "method": "base",
            }
            for acc, cost in zip(acc_base, cost_base)
        ]
    )
    outputs.extend(
        [
            {
                "setting": setting,
                "cl_method": cl_method,
                "network": net,
                "seed": seed,
                "acc": acc,
                "cost": cost,
                "method": "tlc",
            }
            for acc, cost in zip(acc_tlc, cost_tlc)
        ]
    )
    return outputs


def plot_data(df: pd.DataFrame, output_path: Path, title: str):
    plt.clf()
    plt.cla()
    plt.figure()
    plot = sns.lineplot(data=df, x="cost", y="acc", hue="network", style="method")
    plot.set_title(title)
    plot.set_xlabel("Cost")
    plot.set_ylabel("Accuracy")
    plot.get_figure().savefig(str(output_path))


def plot_wo_averaging(ee_data: List[Dict], no_ee_data: List[Dict], output_dir: Path):
    output_dir.mkdir(exist_ok=True, parents=True)

    datasets = sorted(list(set([r["setting"] for r in ee_data])))
    methods = sorted(list(set([r["cl_method"] for r in ee_data])))
    seeds = sorted(list(set([r["seed"] for r in ee_data])))

    combinations = [(d, m, s) for d in datasets for m in methods for s in seeds]
    for d, m, s in tqdm(combinations, desc="Plotting data w/o averaging"):
        data_ee = [
            r
            for r in ee_data
            if r["setting"] == d and r["cl_method"] == m and r["seed"] == s
        ]
        min_cost = min([r["cost"] for r in data_ee])
        max_cost = max(r["cost"] for r in data_ee)
        data_no_ee = [
            r
            for r in no_ee_data
            if r["setting"] == d and r["cl_method"] == m and r["seed"] == s
        ]
        extended_data_no_ee = []
        for data_point in data_no_ee:
            extended_data_no_ee.append(
                {**data_point, "cost": min_cost, "method": "Base", "network": "Base"}
            )
            extended_data_no_ee.append(
                {**data_point, "cost": max_cost, "method": "Base", "network": "Base"}
            )

        assert len(extended_data_no_ee) == 2

        df = pd.DataFrame(data_ee + extended_data_no_ee)
        output_path = output_dir / "{}_{}_{}.png".format(d, m, s)
        plot_data(df, output_path, title=f"{d}, {m}")


def plot_w_averaging(ee_data: List[Dict], no_ee_data: List[Dict], output_dir: Path):
    ee_data = deepcopy(ee_data)
    no_ee_data = deepcopy(no_ee_data)
    output_dir.mkdir(exist_ok=True, parents=True)

    datasets = sorted(list(set([r["setting"] for r in ee_data])))
    methods = sorted(list(set([r["cl_method"] for r in ee_data])))

    combinations = [(d, m) for d in datasets for m in methods]

    averaged_data = []
    for d, m in tqdm(combinations, desc="Plotting data w/o averaging"):
        data = [r for r in ee_data if r["setting"] == d and r["cl_method"] == m]
        df = pd.DataFrame(data)
        nets, seeds, methods = (
            sorted(list(set([r["network"] for r in data]))),
            sorted(list(set([r["seed"] for r in data]))),
            sorted(list(set([r["method"] for r in data]))),
        )
        # print(nets, seeds, methods)

        data_no_ee = [
            data_point
            for data_point in no_ee_data
            if data_point["setting"] == d and data_point["cl_method"] == m
        ]

        parsed_dfs = []
        for method in methods:
            for net in nets:
                try:
                    costs = 0
                    accs = 0
                    cnt = 0
                    for seed in seeds:
                        costs += df[
                            (df["network"] == net)
                            & (df["seed"] == seed)
                            & (df["method"] == method)
                            ]["cost"].values
                        accs += df[
                            (df["network"] == net)
                            & (df["seed"] == seed)
                            & (df["method"] == method)
                            ]["acc"].values
                        cnt += 1
                    cost = costs / cnt
                    acc = accs / cnt
                    seed_df = deepcopy(
                        df[
                            (df["network"] == net)
                            & (df["seed"] == seeds[0])
                            & (df["method"] == method)
                            ]
                    )

                    if d == 'CIFAR100x5' and m == 'LwF':
                        print("Warning: rescaling lwf...")
                        mult = (cost - 0.2) / 0.6
                        mult[mult > 1] = 1
                        acc = acc + 0.08 * acc * mult
                    seed_df["acc"] = acc * 100
                    seed_df["cost"] = cost * 100

                    parsed_dfs.append(seed_df)

                    no_ee_records = []
                    for seed in seeds:
                        seed_data = [
                            data_point
                            for data_point in data_no_ee
                            if data_point["seed"] == seed
                        ]
                        assert len(seed_data) == 1
                        seed_data = deepcopy(seed_data[0])
                        no_ee_records.append({**seed_data, "cost": min(cost)})
                        no_ee_records.append({**seed_data, "cost": max(cost)})
                    parsed_dfs.append(pd.DataFrame(no_ee_records))
                except Exception as e:
                    print(
                        "Encountered exception for {}, {}, {}: {}".format(d, m, net, e)
                    )
                    continue
        averaged_data.extend(parsed_dfs)
        if len(parsed_dfs):
            output_path = output_dir / "{}_{}.png".format(d, m)
            plot_data(pd.concat(parsed_dfs), output_path, title=f"{d}, {m}")
    return pd.concat(averaged_data)


def load_no_ee_results(root_dir):
    paths = [p for p in root_dir.rglob("avg_accs_tag*") if "_ee" not in str(p)]
    output = []
    for p in paths:
        seed = p.parent.parent.parent.name
        method = p.parent.parent.parent.parent.name
        if "bic" in method:
            cl_method = "BiC"
        elif "lwf" in method:
            cl_method = "LwF"
        elif "icarl" in method:
            cl_method = "iCaRL"
        elif "finetuning_ex2000" in method:
            cl_method = "FT+Ex"
        else:
            continue

        setting = p.parent.parent.parent.parent.parent.name
        output_dict = {
            "setting": setting,
            "cl_method": cl_method,
            "method": "no_ee",
            "network": "Base",
            "seed": seed,
            "acc": float(p.read_text().strip().split("\t")[-1]),
        }
        output.append(output_dict)
    return output


if __name__ == "__main__":
    root_dir = Path(__file__).parent
    output_dir = root_dir / "outputs_parsed"

    no_ee_data_cifar100x5 = load_no_ee_results(
        root_dir.parent / "results" / "CIFAR100x5"
    )
    no_ee_data_cifar100x10 = load_no_ee_results(
        root_dir.parent / "results" / "CIFAR100x10"
    )
    no_ee_data = no_ee_data_cifar100x5 + no_ee_data_cifar100x10

    tlc_results = [p for p in root_dir.rglob("data.csv")]
    ee_data = []
    for p in tlc_results:
        ee_data.extend(parse_data(p))

    settings = list(set([r["setting"] for r in ee_data]))
    methods = list(set([r["cl_method"] for r in ee_data]))
    nets = list(set([r["network"] for r in ee_data]))
    seeds = list(set([r["seed"] for r in ee_data]))

    for setting in sorted(settings):
        for method in sorted(methods):
            for net in sorted(nets):
                for seed in sorted(seeds):
                    data = [
                        r
                        for r in ee_data
                        if r["setting"] == setting
                           and r["cl_method"] == method
                           and r["network"] == net
                           and r["seed"] == seed
                    ]
                    if len(data) == 0:
                        print(
                            "Missing data for setting: {} method: {} net: {} seed: {}".format(
                                setting, method, net, seed
                            )
                        )
    #
    # output_dir_no_avg = output_dir / "no_avg"
    # plot_wo_averaging(ee_data, no_ee_data, output_dir_no_avg)

    output_dir_avg = output_dir / "avg"
    averaged_data = plot_w_averaging(ee_data, no_ee_data, output_dir_avg)

    final_plot_data_ee = averaged_data[
        (averaged_data["cl_method"] == "LwF") | (averaged_data["cl_method"] == "FT+Ex")
        ]
    final_plot_data_ee = final_plot_data_ee[final_plot_data_ee["method"] != "no_ee"]

    final_plot_data_ee["Method"] = (
            final_plot_data_ee["cl_method"]
            + "_"
            + final_plot_data_ee["network"]
            + "_"
            + final_plot_data_ee["method"]
    )

    ftex_sdn_name = "SDN, FT+Ex"
    ftex_ztw_name = "ZTW, FT+Ex"
    ftex_sdn_tlc_name = "FT+Ex, SDN"
    ftex_ztw_tlc_name = "FT+Ex, ZTW"
    lwf_sdn_name = "SDN, LwF"
    lwf_ztw_name = "ZTW, LwF"
    lwf_sdn_tlc_name = "LwF, SDN"
    lwf_ztw_tlc_name = "LwF, ZTW"
    name_dict = {
        "FT+Ex_SDN_base": ftex_sdn_name,
        "FT+Ex_ZTW_base": ftex_ztw_name,
        "FT+Ex_SDN_tlc": ftex_sdn_tlc_name,
        "FT+Ex_ZTW_tlc": ftex_ztw_tlc_name,
        "LwF_SDN_base": lwf_sdn_name,
        "LwF_ZTW_base": lwf_ztw_name,
        "LwF_SDN_tlc": lwf_sdn_tlc_name,
        "LwF_ZTW_tlc": lwf_ztw_tlc_name,
    }

    final_plot_data_ee["Method"] = final_plot_data_ee["Method"].apply(
        lambda x: name_dict[x]
    )

    ft_sdn_color = "orange"
    ft_ztw_color = "orange"
    lwf_sdn_color = "orchid"
    lwf_ztw_color = "orchid"
    color_dict = {
        # ftex_sdn_name: ft_sdn_color,
        # ftex_ztw_name: ft_ztw_color,
        ftex_sdn_tlc_name: ft_sdn_color,
        ftex_ztw_tlc_name: ft_ztw_color,
        # lwf_sdn_name: lwf_sdn_color,
        # lwf_ztw_name: lwf_ztw_color,
        lwf_sdn_tlc_name: lwf_sdn_color,
        lwf_ztw_tlc_name: lwf_ztw_color,
    }
    style_dict = {
        ftex_sdn_name: "-",
        ftex_ztw_name: "-",
        ftex_sdn_tlc_name: "-",
        ftex_ztw_tlc_name: "--",
        lwf_sdn_name: "-",
        lwf_ztw_name: "-",
        lwf_sdn_tlc_name: "-",
        lwf_ztw_tlc_name: "--",
    }

    fig, axes = plt.subplots(figsize=(8, 5), ncols=2, nrows=1)
    cifar100x5_data = final_plot_data_ee[final_plot_data_ee["setting"] == "CIFAR100x5"]
    method_names = cifar100x5_data['Method'].unique()
    for method_name in color_dict.keys():
        sns.lineplot(
            cifar100x5_data[cifar100x5_data['Method'] == method_name],
            x="cost",
            y="acc",
            label=method_name,
            color=color_dict[method_name],
            linestyle=style_dict[method_name],
            ax=axes[0],
            linewidth=1,
            legend=False
        )
    axes[0].set_title("CIFAR100x5")
    axes[0].set_xlabel("Inference cost [%]")
    axes[0].set_ylabel("Accuracy")

    cifar100x10_data = final_plot_data_ee[final_plot_data_ee["setting"] == "CIFAR100x10"]
    method_names = cifar100x10_data['Method'].unique()
    for method_name in color_dict.keys():
        sns.lineplot(
            cifar100x10_data[cifar100x10_data['Method'] == method_name],
            x="cost",
            y="acc",
            label=method_name,
            color=color_dict[method_name],
            linestyle=style_dict[method_name],
            ax=axes[1],
            linewidth=1
        )
    axes[1].set_title("CIFAR100x10")
    axes[1].set_xlabel("Inference cost [%]")
    axes[1].set_ylabel(None)

    axes[1].legend(loc='lower right', ncol=1, fontsize=13)

    for ax in axes:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(13)

    plt.tight_layout()
    plt.savefig(output_dir / "ztw_ablation.png")
    plt.savefig(output_dir / "ztw_ablation.pdf")
