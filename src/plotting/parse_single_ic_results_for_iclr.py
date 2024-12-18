import random
from copy import deepcopy
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from results_utils.load_data import (
    ScoresEE,
    ScoresStandard,
    load_averaged_scores,
    load_scores_for_table,
)

BASELINE_NAME = "Base"
AC_NAME = "+AC"

sns.set_style("whitegrid")

# For paper
FONTSIZE_TITLE = 22
FONTSIZE_TICKS = 20
FONTSIZE_LEGEND = 16
LINEWIDTH = 4
MARKERSIZE = 60

METHODS = {
    "ancl_tw_ex_0": "ANCL",
    # "ancl_tw_ex_2000": "ANCL+Ex,
    "bic": "BiC",
    "ewc": "EWC",
    "er": "ER",
    "finetuning_ex0": "FT",
    "finetuning_ex2000": "FT+Ex",
    "gdumb": "GDumb",
    "lode": "LODE",
    # "icarl": "iCaRL",
    "lwf": "LwF",
    # "joint": "Joint",
    "ssil": "SSIL",
}


def plot_scores(
    scores: list, save_path: Path, method: str, setting: str, keep_org_label=False
):
    # skip if there are no early exit results
    # close all previous figures
    plt.close("all")

    scores = sorted(scores, key=lambda s: (not s.early_exit, s.metadata.exp_name))
    min_cost = 0.5

    for score in scores:
        label = f"{score.metadata.exp_name}_{score.metadata.tag}"

        if score.early_exit and min(score.per_th_cost) < min_cost:
            min_cost = min(score.per_th_cost)

        if score.early_exit:
            x = score.per_th_cost
            y = score.per_th_acc
            err = score.per_th_std

            if keep_org_label:
                label = score.metadata.exp_name
                # Get random color
                color = (
                    random.uniform(0, 1),
                    random.uniform(0, 1),
                    random.uniform(0, 1),
                )
            else:
                label = AC_NAME
                color = CMAP[label]

            plot = sns.lineplot(
                x=x, y=y, label=label, linewidth=LINEWIDTH, zorder=2, color=color
            )

            step_size = 5
            marker_x, marker_y = x[1::step_size].tolist(), y[1::step_size].tolist()
            marker_x.append(x[-1])
            marker_y.append(y[-1])

            plt.scatter(marker_x, marker_y, color=color, zorder=3, s=MARKERSIZE)
            plot.fill_between(x, y - err, y + err, alpha=0.2, color=color)
        else:
            x = np.array([min_cost, 1.0])
            y = np.array([score.tag_acc_final, score.tag_acc_final])
            std = np.array([score.tag_acc_std, score.tag_acc_std])

            if keep_org_label:
                label = score.metadata.exp_name
                color = (
                    random.uniform(0, 1),
                    random.uniform(0, 1),
                    random.uniform(0, 1),
                )
            else:
                label = BASELINE_NAME
                color = CMAP[label]

            plot = sns.lineplot(
                x=x,
                y=y,
                label=label,
                linestyle="dashed",
                linewidth=LINEWIDTH,
                zorder=1,
                color=color,
            )
            plot.fill_between(x, y - std, y + std, alpha=0.2, color=color)

    if not keep_org_label:
        plot.legend(fontsize=FONTSIZE_LEGEND, loc="lower right")
        handles, labels = plot.get_legend_handles_labels()
        handles = [
            handles[labels.index(BASELINE_NAME)],
            handles[labels.index(AC_NAME)],
        ]
        labels = [BASELINE_NAME, AC_NAME]
        plot.legend(handles, labels, fontsize=FONTSIZE_LEGEND, loc="lower right")
    else:
        plot.legend(fontsize=4, loc="lower right")

    # Set fontsize for xticks and yticks
    plot.tick_params(axis="both", which="major", labelsize=FONTSIZE_TICKS)
    plot.set_xlabel("Cost", fontsize=FONTSIZE_TITLE)
    plot.set_ylabel("Accuracy", fontsize=FONTSIZE_TITLE)
    plot.set_title(
        f"{setting.split('_')[0]} | {METHODS[method]}", fontsize=FONTSIZE_TITLE
    )
    plt.tight_layout()
    plot.get_figure().savefig(str(save_path).replace(".png", ".pdf"))


def _format_acc(acc, std, precision=2):
    return str(round(acc, precision)) + "$\pm$" + str(round(std, precision))


def save_main_table(scores: list, save_path: Path, method: str, setting: str):
    outputs = []
    scores = sorted(scores, key=lambda s: (s.early_exit, s.metadata.exp_name))

    for score in scores:
        exp_name = score.metadata.exp_name
        if not score.early_exit:
            outputs.append(
                {
                    "Setting": setting,
                    "Method": METHODS[method],
                    "exp_name": exp_name,
                    "Setup": "Base",
                    "Acc": _format_acc(
                        score.tag_acc_final, score.tag_acc_std, precision=2
                    ),
                }
            )
        else:
            outputs.append(
                {
                    "Setting": setting,
                    "Method": METHODS[method],
                    "exp_name": exp_name,
                    "Setup": "+AC",
                    "Acc": _format_acc(
                        score.per_th_acc[-1], score.per_th_std[-1], precision=2
                    ),
                }
            )
    output = pd.DataFrame(outputs)
    output = output.sort_values(
        by=["Method", "Setting", "Setup", "Acc"], ascending=False
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(save_path, index=False)


if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent.parent.parent
    OUTPUT_DIR_PLOTS = ROOT_DIR / "iclr_data_ic_placement" / " dynamic_acc_plots"
    OUTPUT_DIR_TABLES = ROOT_DIR / "iclr_data_ic_placement" / " tables"
    DOWNSAMPLE = True

    averaged_scores = load_averaged_scores(
        ROOT_DIR / "results_ic_placement", downsample=DOWNSAMPLE
    )
    # averaged_scores = [s for s in averaged_scores if s.early_exit is False]
    method_setting_pairs = []

    for method in sorted(METHODS.keys()):
        method_scores = [
            s for s in averaged_scores if s.metadata.exp_name.startswith(method)
        ]
        unique_settings = set(s.metadata.setting for s in method_scores)
        for setting in sorted(unique_settings):
            method_setting_pairs.append((method, setting))

    print()
    print(method_setting_pairs)
    print()

    for method, setting in tqdm(method_setting_pairs, desc="Parsing results"):
        filtered_scores = deepcopy(
            [
                s
                for s in averaged_scores
                if s.metadata.setting == setting
                and s.metadata.exp_name.startswith(method)
            ]
        )
        for s in filtered_scores:
            s.metadata.exp_name = s.metadata.exp_name.replace(method + "_", "")

        save_path = OUTPUT_DIR_PLOTS / setting / f"{method}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # TODO uncomment to plot per method with org labels
        try:
            plot_scores(
                filtered_scores, save_path, method, setting, keep_org_label=True
            )
        except:
            print("Failed plotting for setting", setting, "method", method)

    def setup_fn(x):
        if "uniform_1_only" in x:
            setup = "1st AC"
        elif "uniform_2_only" in x:
            setup = "2nd AC"
        elif "uniform_3_only" in x:
            setup = "3rd AC"
        elif "uniform_4_only" in x:
            setup = "4th AC"
        elif "uniform_5_only" in x:
            setup = "5th AC"
        elif "uniform_6_only" in x:
            setup = "6th AC"
        else:
            setup = "All ACs"
        return setup

    CMAP = {
        "All ACs": "tab:blue",
        "1st AC": "wheat",
        "2nd AC": "gold",
        "3rd AC": "goldenrod",
        "4th AC": "sandybrown",
        "5th AC": "peru",
        "6th AC": "sienna",
    }

    def method_fn(x):
        if x.startswith("finetuning_ex0"):
            return "FT"
        elif x.startswith("finetuning_ex2000"):
            return "FT+Ex"
        elif x.startswith("lwf"):
            return "LwF"
        elif x.startswith("bic"):
            return "BiC"
        elif x.startswith("ancl"):
            return "ANCL"
        elif x.startswith("ssil"):
            return "SSIL"
        elif x.startswith("lode"):
            return "LODE"
        elif x.startswith("er"):
            return "ER"
        elif x.startswith("ewc"):
            return "EWC"
        elif x.startswith("gdumb"):
            return "GDumb"
        else:
            raise ValueError()

    setups = [
        "All ACs",
        "1st AC",
        "2nd AC",
        "3rd AC",
        "4th AC",
        "5th AC",
        "6th AC",
    ]
    method_batches = [
        [
            "FT",
            "FT+Ex",
            "LwF",
            "BiC",
        ],
    ]

    plt.cla()
    plt.clf()
    plt.close("all")
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax_idx, setting in enumerate(["CIFAR100x5", "CIFAR100x10"]):

        setting_scores = [s for s in averaged_scores if s.metadata.setting == setting]
        method_to_scores = {}
        for score in setting_scores:
            method_name = method_fn(score.metadata.exp_name)
            setup = setup_fn(score.metadata.exp_name)
            method_to_scores[(setup, method_name)] = score

        for batch_idx, method_batch in enumerate(method_batches):
            for idx, method in enumerate(method_batch):
                setup_to_scores: Dict[str, ScoresEE] = {
                    setup: method_to_scores[(setup, method)]
                    for setup in setups
                    if (setup, method) in method_to_scores
                }

                for setup, scores in setup_to_scores.items():
                    sns.lineplot(
                        x=scores.per_th_cost,
                        y=scores.per_th_acc,
                        label=setup,
                        linewidth=LINEWIDTH,
                        zorder=2,
                        color=CMAP[setup],
                        ax=axes[ax_idx, idx],
                        legend=idx == 5,
                    )
                    step_size = 5
                    marker_x, marker_y = (
                        scores.per_th_cost[1::step_size].tolist(),
                        scores.per_th_acc[1::step_size].tolist(),
                    )
                    marker_x.append(scores.per_th_cost[-1])
                    marker_y.append(scores.per_th_acc[-1])
                    axes[ax_idx, idx].scatter(
                        marker_x, marker_y, color=CMAP[setup], zorder=3, s=MARKERSIZE
                    )
                    axes[ax_idx, idx].fill_between(
                        scores.per_th_cost,
                        scores.per_th_acc - scores.per_th_std,
                        scores.per_th_acc + scores.per_th_std,
                        alpha=0.2,
                        color=CMAP[setup],
                    )

                    # Set fontsize for xticks and yticks
                    axes[ax_idx, idx].tick_params(
                        axis="both", which="major", labelsize=FONTSIZE_TICKS
                    )
                    if ax_idx == 1:
                        axes[ax_idx, idx].set_xlabel("Cost", fontsize=FONTSIZE_TITLE)
                    if idx == 0:
                        axes[ax_idx, idx].set_ylabel(
                            "Accuracy", fontsize=FONTSIZE_TITLE
                        )
                    else:
                        axes[ax_idx, idx].set_ylabel(None)
                    axes[ax_idx, idx].set_title(
                        f"{setting.replace('_rn18', '').replace('_vit', '')} | {method}",
                        fontsize=FONTSIZE_TITLE,
                    )

    axes[-1, -1].legend(fontsize=FONTSIZE_LEGEND, loc="lower right", ncol=2)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR_PLOTS / f"plot.pdf")

    # OUTPUT_DIR_TABLES.mkdir(parents=True, exist_ok=True)
    # table_data = load_scores_for_table(ROOT_DIR / "results", filter=filter_fn)
    # table_data["setup"] = table_data.apply(setup_fn, axis=1)
    # table_data["method"] = table_data["exp_name"].apply(method_fn)
    # table_data = table_data[["setting", "method", "setup", "seed", "acc"]]
    # table_data = table_data.sort_values(
    #     by=["setting", "method", "setup", "seed"], ascending=True
    # )
    # methods = [
    #     "FT",
    #     "FT+Ex",
    #     "GDumb",
    #     "ANCL",
    #     "BiC",
    #     "ER",
    #     "EWC",
    #     "LwF",
    #     "LODE",
    #     "SSIL",
    # ]
    # # settings = [
    # #     "CIFAR100x5",
    # #     "CIFAR100x10",
    # #     "CIFAR100x6",
    # #     "CIFAR100x11",
    # #     "ImageNet100x5_rn18",
    # #     "ImageNet100x10_rn18",
    # #     "ImageNet100x5_vit",
    # #     "ImageNet100x10_vit",
    # # ]
    # settings = [
    #     "CIFAR100x20",
    # ]
    # setups = ["Base", "+AC"]
    # seeds = [0, 1, 2]
    #
    # dfs_combined = []
    # for setting in settings:
    #     setting_df = table_data[table_data["setting"] == setting]
    #     outputs_setting = []
    #     for setup in setups:
    #         for seed in sorted(setting_df["seed"].unique().tolist()):
    #             output = {"setting": setting, "setup": setup, "seed": seed}
    #             for method in methods:
    #                 result_data = setting_df[
    #                     (setting_df["method"] == method)
    #                     & (setting_df["setup"] == setup)
    #                     & (setting_df["seed"] == seed)
    #                 ]
    #                 if len(result_data) == 0:
    #                     output[method] = -1
    #                 elif len(result_data) > 1:
    #                     print(
    #                         f"WARNING: More than 1 value for {setting}, {method} {setup}, seed {seed}"
    #                     )
    #                     output[method] = -1
    #                 else:
    #                     output[method] = result_data["acc"].values[0]
    #             outputs_setting.append(output)
    #     df = pd.DataFrame(outputs_setting)
    #     df["Avg"] = df.values[:, 3:].mean(axis=1)
    #     base = df[df["setup"] == "Base"]
    #     ac = df[df["setup"] == "+AC"]
    #     diff = deepcopy(ac)
    #     diff["setup"] = "$\Delta$"
    #     diff.iloc[:, 3:] = np.array(ac.values[:, 3:]) - np.array(base.values[:, 3:])
    #     df = pd.concat([base, ac, diff], axis=0)
    #     df = (
    #         df.groupby(by=["setting", "setup"])
    #         .aggregate({m: ["mean", "std"] for m in methods + ["Avg"]})
    #         .reset_index()
    #     )
    #     for i, method_name in enumerate(methods + ["Avg"]):
    #         method_col_name_mean = df.columns[(i + 1) * 2]
    #         method_col_name_std = df.columns[(i + 1) * 2 + 1]
    #         # Format to string with std and 2 decimals
    #         mean = df[method_col_name_mean].apply(lambda x: f"{float(x):.2f}")
    #         std = df[method_col_name_std].apply(lambda x: f"{float(x):.2f}")
    #         df["_" + method_name] = mean + "\\tiny{$\pm$" + std + "}"
    #
    #     df = df[
    #         [
    #             "setting",
    #             "setup",
    #             "_FT",
    #             "_FT+Ex",
    #             "_GDumb",
    #             "_ANCL",
    #             "_BiC",
    #             "_ER",
    #             "_EWC",
    #             "_LwF",
    #             "_LODE",
    #             "_SSIL",
    #             "_Avg",
    #         ]
    #     ]
    #     df.columns = [
    #         "Setting",
    #         "Setup",
    #         "FT",
    #         "FT+Ex",
    #         "GDumb",
    #         "ANCL",
    #         "BiC",
    #         "ER",
    #         "EWC",
    #         "LwF",
    #         "LODE",
    #         "SSIL",
    #         "Avg",
    #     ]
    #     df = df.sort_values(
    #         by=["Setup"],
    #         key=lambda x: x.map({"Base": 0, "+AC": 1, "$\Delta$": 2}),
    #         ascending=True,
    #     )
    #     df.iloc[-1, 2:] = df.iloc[-1, 2:].apply(
    #         lambda x: x if x.startswith("-") else "+" + x
    #     )
    #     df.to_csv(OUTPUT_DIR_TABLES / f"{setting}.csv", index=False)
    #     dfs_combined.append(df)
    # combined_df = pd.concat(dfs_combined, axis=0)
    # combined_df.to_csv(OUTPUT_DIR_TABLES / "combined.csv", index=False)
