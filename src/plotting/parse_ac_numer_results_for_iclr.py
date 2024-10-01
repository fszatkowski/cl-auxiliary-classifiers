from copy import deepcopy
from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from results_utils.load_data import load_averaged_scores, load_scores_for_table, ScoresStandard, ScoresEE

BASELINE_NAME = "Base"
AC_NAME_SPARSE = "+3AC"
AC_NAME_BASE = "+6AC"
AC_NAME_DENSE = "+12AC"

CMAP = {
    BASELINE_NAME: "tab:red",
    AC_NAME_SPARSE: "tab:orange",
    AC_NAME_BASE: "tab:blue",
    AC_NAME_DENSE: "tab:green",
}

sns.set_style("whitegrid")

FONTSIZE_TITLE = 26
FONTSIZE_TICKS = 22
FONTSIZE_LEGEND = 20
LINEWIDTH = 2
MARKERSIZE = 30

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
            if "dense" in label:
                label = AC_NAME_DENSE
                color = CMAP[AC_NAME_DENSE]
            elif "sparse" in label:
                label = AC_NAME_SPARSE
                color = CMAP[AC_NAME_SPARSE]
            else:
                label = AC_NAME_BASE
                color = CMAP[AC_NAME_BASE]
            plot = sns.lineplot(
                x=x, y=y, label=label, linewidth=LINEWIDTH, zorder=2, color=color
            )
            step_size = 5
            marker_x, marker_y = x[1::step_size].tolist(), y[1::step_size].tolist()
            marker_x.append(x[-1])
            marker_y.append(y[-1])
            plt.scatter(marker_x, marker_y, color=color, zorder=3, s=MARKERSIZE)

            # fill error bars
            plot.fill_between(x, y - err, y + err, alpha=0.2, color=color)
        else:
            x = np.array([min_cost, 1.0])
            y = np.array([score.tag_acc_final, score.tag_acc_final])
            std = np.array([score.tag_acc_std, score.tag_acc_std])
            label = BASELINE_NAME
            plot = sns.lineplot(
                x=x,
                y=y,
                label=label,
                linestyle="dashed",
                linewidth=LINEWIDTH,
                zorder=1,
                color=CMAP[BASELINE_NAME],
            )
            # fill error bars
            plot.fill_between(x, y - std, y + std, alpha=0.2, color=CMAP[BASELINE_NAME])

    plot.legend(fontsize=FONTSIZE_LEGEND, loc="lower right")
    handles, labels = plot.get_legend_handles_labels()
    handles = [
        handles[labels.index(BASELINE_NAME)],
        handles[labels.index(AC_NAME_SPARSE)],
        handles[labels.index(AC_NAME_BASE)],
        handles[labels.index(AC_NAME_DENSE)],
    ]
    labels = [BASELINE_NAME, AC_NAME_SPARSE, AC_NAME_BASE, AC_NAME_DENSE]
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
            if "dense" in exp_name:
                setup = "+12AC"
            elif "sparse" in exp_name:
                setup = "+3AC"
            else:
                setup = "+6AC"
            outputs.append(
                {
                    "Setting": setting,
                    "Method": METHODS[method],
                    "exp_name": exp_name,
                    "Setup": setup,
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
    OUTPUT_DIR_PLOTS = ROOT_DIR / "iclr_data_ac_setup" / "dynamic_acc_plots_ac"
    OUTPUT_DIR_TABLES = ROOT_DIR / "iclr_data_ac_setup" / "tables"
    DOWNSAMPLE = True

    def filter_fn(path: Path):
        string_path = str(path)
        if "CIFAR100x5" not in string_path and "CIFAR100x10" not in string_path:
            return False
        if (
            "dense" not in string_path
            and "sdn" in string_path
            and "dense" not in string_path
            and not path.parent.parent.parent.name.endswith("sdn")
        ):
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
            s for s in averaged_scores if s.metadata.exp_name.startswith(method)
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
        elif "dense" in x["exp_name"]:
            setup = "+12AC"
        elif "sparse" in x["exp_name"]:
            setup = "+3AC"
        elif "sdn" in x["exp_name"]:
            setup = "+6AC"
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
            elif 'sparse' in score.metadata.exp_name:
                setup = AC_NAME_SPARSE
            elif 'dense' in score.metadata.exp_name:
                setup = AC_NAME_DENSE
            else:
                setup = AC_NAME_BASE
            method_to_scores[(setup, method_name)] = score

        for batch_idx, method_batch in enumerate(method_batches):
            plt.cla()
            plt.clf()
            plt.close("all")

            fig, axes = plt.subplots(1, len(method_batch), figsize=(24, 4))
            for idx, method in enumerate(method_batch):
                score_base: ScoresStandard = method_to_scores[(BASELINE_NAME, method)]
                scores_ee: List[Tuple[str, ScoresEE]] = [
                    (AC_NAME_SPARSE, method_to_scores[(AC_NAME_SPARSE, method)]),
                    (AC_NAME_BASE, method_to_scores[(AC_NAME_BASE, method)]),
                (AC_NAME_DENSE, method_to_scores[(AC_NAME_DENSE, method)])]

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
                    handles[labels.index(AC_NAME_SPARSE)],
                    handles[labels.index(AC_NAME_BASE)],
                    handles[labels.index(AC_NAME_DENSE)]
                ]
                labels = [BASELINE_NAME, AC_NAME_SPARSE, AC_NAME_BASE, AC_NAME_DENSE]
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
    setups = ["Base", "+3AC", "+6AC", "+12AC"]

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
        ac3 = df[df["setup"] == "+3AC"]
        ac6 = df[df["setup"] == "+6AC"]
        ac12 = df[df["setup"] == "+12AC"]

        diff3 = deepcopy(ac3)
        diff3.iloc[:, 3:] = np.array(diff3.values[:, 3:]) - np.array(base.values[:, 3:])

        diff6 = deepcopy(ac6)
        diff6.iloc[:, 3:] = np.array(diff6.values[:, 3:]) - np.array(base.values[:, 3:])

        diff12 = deepcopy(ac12)
        diff12.iloc[:, 3:] = np.array(diff12.values[:, 3:]) - np.array(
            base.values[:, 3:]
        )

        df = pd.concat([diff3, diff6, diff12], axis=0)
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
            key=lambda x: x.map({"+3AC": 0, "+6AC": 1, "+12AC": 2}),
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
