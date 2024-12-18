import random
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

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
SPARSE_AC_NAME = "+6AC"
MEDIUM_AC_NAME = "+10AC"
DENSE_AC_NAME = "+18AC"

CMAP = {
    BASELINE_NAME: "tab:red",
    SPARSE_AC_NAME: "tab:orange",
    MEDIUM_AC_NAME: "tab:blue",
    DENSE_AC_NAME: "tab:green",
}

sns.set_style("whitegrid")

# For paper
FONTSIZE_TITLE = 24
FONTSIZE_TICKS = 22
FONTSIZE_LEGEND = 22
LINEWIDTH = 5
MARKERSIZE = 80

METHODS = {
    "ancl_tw_ex_0": "ANCL",
    # "ancl_tw_ex_2000": "ANCL+Ex,
    "bic": "BiC",
    "ewc": "EWC",
    "er": "ER",
    "der++": "DER++",
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
    # plot.get_figure().savefig(str(save_path))


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
    OUTPUT_DIR_PLOTS = ROOT_DIR / "iclr_data_vgg" / " dynamic_acc_plots"
    OUTPUT_DIR_TABLES = ROOT_DIR / "iclr_data_vgg" / " tables"
    DOWNSAMPLE = True

    def filter_fn(path: Path):
        # if "vit" in str(path):
        #     return False
        if "pretrained_old" in str(path):
            return False
        string_path = str(path)
        # if 'der' not in string_path:
        #     return False
        # if 'medium' in string_path or 'dense' in string_path:
        #     return False

        if "CIFAR100x20" in string_path or "CIFAR100x50" in string_path:
            return False

        return True

    averaged_scores = load_averaged_scores(
        ROOT_DIR / "results_vgg19", downsample=DOWNSAMPLE, filter=filter_fn
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

        # # # TODO uncomment to plot per method with org labels
        # try:
        #     plot_scores(
        #         filtered_scores, save_path, method, setting, keep_org_label=True
        #     )
        # except:
        #     print("Failed plotting for setting", setting, "method", method)

    def setup_fn(x):
        if "vgg19_dense" in x:
            setup = "+18AC"
        elif "vgg19_medium" in x:
            setup = "+10AC"
        elif "vgg19_small" in x:
            setup = "+6AC"
        elif "vgg19" not in x:
            setup = "Base"
        else:
            raise ValueError()
        return setup

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
        elif x.startswith("der++"):
            return "DER++"
        elif x.startswith("ewc"):
            return "EWC"
        elif x.startswith("gdumb"):
            return "GDumb"
        else:
            raise ValueError()

    # for setting in ["CIFAR100x5", "CIFAR100x10", "CIFAR100x6", "CIFAR100x11", "ImageNet100x5_rn18",
    #                 "ImageNet100x10_rn18", "ImageNet100x5_vit", "ImageNet100x10_vit"]:
    for setting in ["CIFAR100x10", "CIFAR100x5"]:
        method_batches = [
            [
                "ANCL",
                "BiC",
                "DER++",
                "LODE",
                "SSIL",
            ],
            ["ER", "EWC", "FT", "FT+Ex", "GDumb", "LwF"],
        ]
        setting_scores = [s for s in averaged_scores if s.metadata.setting == setting]
        method_to_scores = {}
        for score in setting_scores:
            method_name = method_fn(score.metadata.exp_name)
            setup = setup_fn(score.metadata.exp_name)
            method_to_scores[(setup, method_name)] = score

        for batch_idx, method_batch in enumerate(method_batches):
            plt.cla()
            plt.clf()
            plt.close("all")

            if len(method_batch) == 5:
                figsize = 25
            elif len(method_batch) == 6:
                figsize = 30
            else:
                raise ValueError()

            fig, axes = plt.subplots(1, len(method_batch), figsize=(figsize, 4))
            for idx, method in enumerate(method_batch):
                score_base: ScoresStandard = method_to_scores[("Base", method)]
                score_ee_sparse: ScoresEE = method_to_scores[("+6AC", method)]
                score_ee_medium: ScoresEE = method_to_scores[("+10AC", method)]
                score_ee_dense: ScoresEE = method_to_scores[("+18AC", method)]

                for score_ee, AC_NAME in (
                    (score_ee_sparse, SPARSE_AC_NAME),
                    (score_ee_medium, MEDIUM_AC_NAME),
                    (score_ee_dense, DENSE_AC_NAME),
                ):
                    sns.lineplot(
                        x=score_ee.per_th_cost,
                        y=score_ee.per_th_acc,
                        label=AC_NAME,
                        linewidth=LINEWIDTH,
                        zorder=2,
                        color=CMAP[AC_NAME],
                        ax=axes[idx],
                        legend=idx == 5,
                    )
                    step_size = 5
                    marker_x, marker_y = (
                        score_ee.per_th_cost[1::step_size].tolist(),
                        score_ee.per_th_acc[1::step_size].tolist(),
                    )
                    marker_x.append(score_ee.per_th_cost[-1])
                    marker_y.append(score_ee.per_th_acc[-1])
                    axes[idx].scatter(
                        marker_x, marker_y, color=CMAP[AC_NAME], zorder=3, s=MARKERSIZE
                    )
                    axes[idx].fill_between(
                        score_ee.per_th_cost,
                        score_ee.per_th_acc - score_ee.per_th_std,
                        score_ee.per_th_acc + score_ee.per_th_std,
                        alpha=0.2,
                        color=CMAP[AC_NAME],
                    )

                min_cost = min(score_ee_dense.per_th_cost)
                sns.lineplot(
                    x=[min_cost, 1.0],
                    y=[score_base.tag_acc_final, score_base.tag_acc_final],
                    label=BASELINE_NAME,
                    linestyle="dashed",
                    linewidth=LINEWIDTH,
                    zorder=1,
                    color=CMAP[BASELINE_NAME],
                    ax=axes[idx],
                    legend=idx == 5,
                )
                axes[idx].fill_between(
                    [min_cost, 1],
                    score_base.tag_acc_final - score_base.tag_acc_std,
                    score_base.tag_acc_final + score_base.tag_acc_std,
                    alpha=0.2,
                    color=CMAP[BASELINE_NAME],
                )

                # Set fontsize for xticks and yticks
                axes[idx].tick_params(
                    axis="both", which="major", labelsize=FONTSIZE_TICKS
                )
                axes[idx].set_xlabel("Cost", fontsize=FONTSIZE_TITLE)
                if idx == 0:
                    axes[idx].set_ylabel("Accuracy", fontsize=FONTSIZE_TITLE)
                else:
                    axes[idx].set_ylabel(None)
                axes[idx].set_title(
                    f"{setting.replace('_rn18', '').replace('_vit', '')} | {method}",
                    fontsize=FONTSIZE_TITLE,
                )

            handles, labels = axes[-1].get_legend_handles_labels()

            handles = [
                handles[labels.index(BASELINE_NAME)],
                handles[labels.index(MEDIUM_AC_NAME)],
                handles[labels.index(DENSE_AC_NAME)],
            ]
            labels = [BASELINE_NAME, MEDIUM_AC_NAME, DENSE_AC_NAME]
            axes[-1].legend(
                handles, labels, fontsize=FONTSIZE_LEGEND, loc="lower right"
            )

            fig.tight_layout()
            fig.savefig(OUTPUT_DIR_PLOTS / f"{setting}_{batch_idx}.pdf")

    OUTPUT_DIR_TABLES.mkdir(parents=True, exist_ok=True)
    table_data = load_scores_for_table(ROOT_DIR / "results_vgg19", filter=filter_fn)
    table_data["setup"] = table_data["exp_name"].apply(setup_fn)
    table_data["method"] = table_data["exp_name"].apply(method_fn)
    table_data = table_data[["setting", "method", "setup", "seed", "acc"]]
    table_data = table_data.sort_values(
        by=["setting", "method", "setup", "seed"], ascending=True
    )
    methods = [
        "FT",
        "FT+Ex",
        "GDumb",
        "ANCL",
        "BiC",
        "DER++",
        "ER",
        "EWC",
        "LwF",
        "LODE",
        "SSIL",
    ]
    # settings = [
    #     "CIFAR100x5",
    #     "CIFAR100x10",
    #     "CIFAR100x6",
    #     "CIFAR100x11",
    #     "ImageNet100x5_rn18",
    #     "ImageNet100x10_rn18",
    #     "ImageNet100x5_vit",
    #     "ImageNet100x10_vit",
    # ]
    settings = ["CIFAR100x10", "CIFAR100x5"]
    setups = ["Base", "+6AC", "+10AC", "+18AC"]
    seeds = [0, 1, 2]

    dfs_combined = []
    for setting in settings:
        setting_df = table_data[table_data["setting"] == setting]
        outputs_setting = []
        for setup in setups:
            for seed in sorted(setting_df["seed"].unique().tolist()):
                output = {"setting": setting, "setup": setup, "seed": seed}
                for method in methods:
                    result_data = setting_df[
                        (setting_df["method"] == method)
                        & (setting_df["setup"] == setup)
                        & (setting_df["seed"] == seed)
                    ]
                    if len(result_data) == 0:
                        output[method] = -1
                    elif len(result_data) > 1:
                        print(
                            f"WARNING: More than 1 value for {setting}, {method} {setup}, seed {seed}"
                        )
                        output[method] = -1
                    else:
                        output[method] = result_data["acc"].values[0]
                outputs_setting.append(output)
        df = pd.DataFrame(outputs_setting)
        df["Avg"] = df.values[:, 3:].mean(axis=1)

        df = (
            df.groupby(by=["setting", "setup"])
            .aggregate({m: ["mean", "std"] for m in methods + ["Avg"]})
            .reset_index()
        )
        for i, method_name in enumerate(methods + ["Avg"]):
            method_col_name_mean = df.columns[(i + 1) * 2]
            method_col_name_std = df.columns[(i + 1) * 2 + 1]
            # Format to string with std and 2 decimals
            mean = df[method_col_name_mean].apply(lambda x: f"{float(x):.2f}")
            std = df[method_col_name_std].apply(lambda x: f"{float(x):.2f}")
            df["_" + method_name] = mean + "\\tiny{$\pm$" + std + "}"

        df = df[
            [
                "setting",
                "setup",
                "_FT",
                "_FT+Ex",
                "_GDumb",
                "_ANCL",
                "_BiC",
                "_DER++",
                "_ER",
                "_EWC",
                "_LwF",
                "_LODE",
                "_SSIL",
                "_Avg",
            ]
        ]
        df.columns = [
            "Setting",
            "Setup",
            "FT",
            "FT+Ex",
            "GDumb",
            "ANCL",
            "BiC",
            "DER++",
            "ER",
            "EWC",
            "LwF",
            "LODE",
            "SSIL",
            "Avg",
        ]
        df = df.sort_values(
            by=["Setup"],
            key=lambda x: x.map({"Base": 0, "+AC": 1, "$\Delta$": 2}),
            ascending=True,
        )
        df.to_csv(OUTPUT_DIR_TABLES / f"{setting}.csv", index=False)
        dfs_combined.append(df)
    combined_df = pd.concat(dfs_combined, axis=0)
    combined_df.to_csv(OUTPUT_DIR_TABLES / "combined.csv", index=False)
