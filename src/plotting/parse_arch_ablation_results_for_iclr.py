from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from results_utils.load_data import load_averaged_scores, load_scores_for_table, ScoresStandard, ScoresEE

BASELINE_NAME = "Base"
LINEAR_PROBING_NAME = "LP"
STANDARD_NAME = "+AC"
CASCADING_NAME = "+AC+C"
ENSEMBLING_NAME = "+AC+E"

CMAP = {
    BASELINE_NAME: "tab:red",
    LINEAR_PROBING_NAME: "tab:orange",
    STANDARD_NAME: "tab:blue",
    CASCADING_NAME: "tab:gray",
    ENSEMBLING_NAME: "tab:purple",
}

sns.set_style("whitegrid")

# For paper
FONTSIZE_TITLE = 26
FONTSIZE_TICKS = 22
FONTSIZE_LEGEND = 20
LINEWIDTH = 2
MARKERSIZE = 30

# For examining results
# FONTSIZE_TITLE = 12
# FONTSIZE_TICKS = 12
# FONTSIZE_LEGEND = 10
# LINEWIDTH = 1
# MARKERSIZE = 10

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


def plot_scores(scores: list, save_path: Path, method: str, setting: str):
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
            # TODO uncomment later
            if "ensembling" in label:
                label = ENSEMBLING_NAME
                color = CMAP[ENSEMBLING_NAME]
            elif "cascading" in label:
                label = CASCADING_NAME
                color = CMAP[CASCADING_NAME]
            elif "detach" in label:
                label = LINEAR_PROBING_NAME
                color = CMAP[LINEAR_PROBING_NAME]
            else:
                label = STANDARD_NAME
                color = CMAP[STANDARD_NAME]
            plot = sns.lineplot(
                x=x, y=y, label=label, linewidth=LINEWIDTH, zorder=2, color=color
            )

            # TODO comment laeter
            # plot = sns.lineplot(x=x, y=y, label=label, linewidth=LINEWIDTH, zorder=2)

            step_size = 5
            marker_x, marker_y = x[1::step_size].tolist(), y[1::step_size].tolist()
            marker_x.append(x[-1])
            marker_y.append(y[-1])

            # TODO uncomment later
            plt.scatter(marker_x, marker_y, zorder=3, s=MARKERSIZE, color=color)
            plot.fill_between(x, y - err, y + err, alpha=0.2, color=color)

            # TODO comment later
            # plot.fill_between(
            #     x, y - err, y + err, alpha=0.2, color=plot.lines[0].get_color()
            # )
        else:
            x = np.array([min_cost, 1.0])
            y = np.array([score.tag_acc_final, score.tag_acc_final])
            std = np.array([score.tag_acc_std, score.tag_acc_std])
            label = BASELINE_NAME
            color = CMAP[BASELINE_NAME]
            plot = sns.lineplot(
                x=x,
                y=y,
                label=label,
                linestyle="dashed",
                linewidth=LINEWIDTH,
                zorder=1,
                color=color,
            )
            # fill error bars
            plot.fill_between(x, y - std, y + std, alpha=0.2, color=color)

    plot.legend(fontsize=FONTSIZE_LEGEND, loc="lower right")
    handles, labels = plot.get_legend_handles_labels()
    # TODO uncomment later
    handles = [
        handles[labels.index(BASELINE_NAME)],
        handles[labels.index(STANDARD_NAME)],
        handles[labels.index(CASCADING_NAME)],
        handles[labels.index(ENSEMBLING_NAME)],
        # handles[labels.index(LINEAR_PROBING_NAME)],
    ]
    labels = [
        BASELINE_NAME,
        STANDARD_NAME,
        CASCADING_NAME,
        ENSEMBLING_NAME,
        # LINEAR_PROBING_NAME,
    ]
    plot.legend(handles, labels, fontsize=FONTSIZE_LEGEND, loc="lower right")

    # Set fontsize for xticks and yticks
    plot.tick_params(axis="both", which="major", labelsize=FONTSIZE_TICKS)
    plot.set_xlabel("Cost", fontsize=FONTSIZE_TITLE)
    plot.set_ylabel("Accuracy", fontsize=FONTSIZE_TITLE)
    plot.set_title(
        f"{setting.split('_')[0]} | {METHODS[method]}", fontsize=FONTSIZE_TITLE
    )
    plt.tight_layout()
    # plot.get_figure().savefig(str(save_path))
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
    OUTPUT_DIR_PLOTS = ROOT_DIR / "iclr_data_arch_ablation" / "dynamic_acc_plots_arch"
    OUTPUT_DIR_TABLES = ROOT_DIR / "iclr_data_arch_ablation" / "tables"
    DOWNSAMPLE = True

    def filter_fn(path: Path):
        if "CIFAR100x5" not in str(path) and "CIFAR100x10" not in str(path):
            return False
        string_path = str(path)
        if "sparse" in string_path or "dense" in string_path or "detach" in string_path:
            return False
        else:
            return True

    averaged_scores = load_averaged_scores(
        ROOT_DIR / "results", downsample=DOWNSAMPLE, filter=filter_fn
    )
    # averaged_scores = [s for s in averaged_scores if s.early_exit is False]
    method_setting_pairs = []

    for method in sorted(METHODS.keys()):
        method_scores = [
            deepcopy(s) for s in averaged_scores if s.metadata.exp_name.startswith(method)
        ]
        unique_settings = set(s.metadata.setting for s in method_scores)
        for setting in sorted(unique_settings):
            method_setting_pairs.append((method, setting))

    print()
    print(method_setting_pairs)
    print()

    for method, setting in tqdm(method_setting_pairs, desc="Parsing results"):
        filtered_scores = [
            deepcopy(s)
            for s in averaged_scores
            if s.metadata.setting == setting and s.metadata.exp_name.startswith(method)
        ]
        for s in filtered_scores:
            s.metadata.exp_name = s.metadata.exp_name.replace(method + "_", "")

        save_path = OUTPUT_DIR_PLOTS / setting / f"{method}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plot_scores(filtered_scores, save_path, method, setting)

    def setup_fn(x):
        if not x["early_exit"]:
            setup = "Base"
        elif "cascading" in x["exp_name"]:
            setup = "+AC+C"
        elif "ensembling" in x["exp_name"]:
            setup = "+AC+E"
        elif "sdn" in x["exp_name"]:
            setup = "+AC"
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
        elif x.startswith("ewc"):
            return "EWC"
        elif x.startswith("gdumb"):
            return "GDumb"
        else:
            raise ValueError()

    for setting in ["CIFAR100x5", "CIFAR100x10"]:
        method_batches = [['ANCL', "BiC", "ER", "LODE", "SSIL", ], ["EWC", "FT", "FT+Ex", "GDumb", "LwF"]]
        setting_scores = [
            s
            for s in averaged_scores
            if s.metadata.setting == setting
        ]
        method_to_scores = {}
        for score in setting_scores:
            method_name = method_fn(score.metadata.exp_name)
            if score.early_exit is False:
                setup = BASELINE_NAME
            elif 'cascading' in score.metadata.exp_name:
                setup = CASCADING_NAME
            elif 'ensembling' in score.metadata.exp_name:
                setup = ENSEMBLING_NAME
            else:
                setup = STANDARD_NAME
            method_to_scores[(setup, method_name)] = score

        for batch_idx, method_batch in enumerate(method_batches):
            plt.cla()
            plt.clf()
            plt.close("all")

            fig, axes = plt.subplots(1, len(method_batch), figsize=(24, 4))
            for idx, method in enumerate(method_batch):
                score_base: ScoresStandard = method_to_scores[(BASELINE_NAME, method)]
                scores_ee: List[Tuple[str, ScoresEE]] = [
                    (CASCADING_NAME, method_to_scores[(CASCADING_NAME, method)]),
                    (STANDARD_NAME, method_to_scores[(STANDARD_NAME, method)]),
                (ENSEMBLING_NAME, method_to_scores[(ENSEMBLING_NAME, method)])]

                for method_name, score_ee in scores_ee:
                    sns.lineplot(
                        x=score_ee.per_th_cost, y=score_ee.per_th_acc, label=method_name, linewidth=LINEWIDTH, zorder=2,
                        color=CMAP[method_name], ax=axes[idx],
                        legend=idx == 5
                    )
                    step_size = 5
                    marker_x, marker_y = score_ee.per_th_cost[1::step_size].tolist(), score_ee.per_th_acc[
                                                                                      1::step_size].tolist()
                    marker_x.append(score_ee.per_th_cost[-1])
                    marker_y.append(score_ee.per_th_acc[-1])
                    axes[idx].scatter(marker_x, marker_y, color=CMAP[method_name], zorder=3, s=MARKERSIZE)
                    axes[idx].fill_between(score_ee.per_th_cost, score_ee.per_th_acc - score_ee.per_th_std,
                                           score_ee.per_th_acc + score_ee.per_th_std, alpha=0.2, color=CMAP[method_name])

                min_cost = min([score[1].per_th_cost[0] for score in scores_ee])
                sns.lineplot(
                    x=[min_cost, 1.],
                    y=[score_base.tag_acc_final, score_base.tag_acc_final],
                    label=BASELINE_NAME,
                    linestyle="dashed",
                    linewidth=LINEWIDTH,
                    zorder=1,
                    color=CMAP[BASELINE_NAME],
                    ax=axes[idx],
                    legend=idx == 5
                )
                axes[idx].fill_between([min_cost, 1], score_base.tag_acc_final - score_base.tag_acc_std,
                                       score_base.tag_acc_final + score_base.tag_acc_std, alpha=0.2,
                                       color=CMAP[BASELINE_NAME])

                # Set fontsize for xticks and yticks
                axes[idx].tick_params(axis="both", which="major", labelsize=FONTSIZE_TICKS)
                axes[idx].set_xlabel("Cost", fontsize=FONTSIZE_TITLE)
                if idx == 0:
                    axes[idx].set_ylabel("Accuracy", fontsize=FONTSIZE_TITLE)
                else:
                    axes[idx].set_ylabel(None)
                axes[idx].set_title(
                    f"{setting.replace('_rn', '').replace('_vit', '')} | {method}", fontsize=FONTSIZE_TITLE
                )

            try:
                handles, labels = axes[-1].get_legend_handles_labels()

                handles = [
                    handles[labels.index(BASELINE_NAME)],
                    handles[labels.index(STANDARD_NAME)],
                    handles[labels.index(CASCADING_NAME)],
                    handles[labels.index(ENSEMBLING_NAME)]
                ]
                labels = [BASELINE_NAME, STANDARD_NAME, CASCADING_NAME, ENSEMBLING_NAME]
                axes[-1].legend(handles, labels, fontsize=FONTSIZE_LEGEND, loc="lower right")
            except:
                axes[-1].legend(fontsize=FONTSIZE_LEGEND, loc="lower right")

            fig.tight_layout()
            fig.savefig(OUTPUT_DIR_PLOTS / f"{setting}_{batch_idx}.pdf")
            # fig.savefig(OUTPUT_DIR_PLOTS / f"{setting}_{batch_idx}.png")

    OUTPUT_DIR_TABLES.mkdir(parents=True, exist_ok=True)
    table_data = load_scores_for_table(ROOT_DIR / "results", filter=filter_fn)
    table_data["setup"] = table_data.apply(setup_fn, axis=1)
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
        "ER",
        "EWC",
        "LwF",
        "LODE",
        "SSIL",
    ]
    settings = ["CIFAR100x5", "CIFAR100x10"]
    setups = ["Base", "+AC", "+AC+E", "+AC+C"]
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
        base = df[df["setup"] == "Base"]
        ac = df[df["setup"] == "+AC"]
        casc = df[df["setup"] == "+AC+C"]
        ens = df[df["setup"] == "+AC+E"]

        ac.iloc[:, 3:] = np.array(ac.values[:, 3:]) - np.array(base.values[:, 3:])
        casc.iloc[:, 3:] = np.array(casc.values[:, 3:]) - np.array(base.values[:, 3:])
        ens.iloc[:, 3:] = np.array(ens.values[:, 3:]) - np.array(base.values[:, 3:])

        df = pd.concat([ac, casc, ens], axis=0)
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
            "ER",
            "EWC",
            "LwF",
            "LODE",
            "SSIL",
            "Avg",
        ]
        df = df.sort_values(
            by=["Setup"],
            key=lambda x: x.map({"+AC": 0, "+AC+C": 1, "+AC+E": 2}),
            ascending=True,
        )
        df.iloc[0, 2:] = df.iloc[0, 2:].apply(
            lambda x: x if x.startswith("-") else "+" + x
        )
        df.iloc[1, 2:] = df.iloc[1, 2:].apply(
            lambda x: x if x.startswith("-") else "+" + x
        )
        df.iloc[2, 2:] = df.iloc[2, 2:].apply(
            lambda x: x if x.startswith("-") else "+" + x
        )
        df.to_csv(OUTPUT_DIR_TABLES / f"{setting}.csv", index=False)
        dfs_combined.append(df)
    combined_df = pd.concat(dfs_combined, axis=0)
    combined_df.to_csv(OUTPUT_DIR_TABLES / "combined.csv", index=False)
