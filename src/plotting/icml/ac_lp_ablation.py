import random
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from plotting.icml.common import *
from results_utils.load_data import (
    ScoresEE,
    ScoresStandard,
    load_averaged_scores,
    load_scores_for_table,
)

SETTINGS = ["CIFAR100x5", "CIFAR100x10"]


PLOT_SETTINGS = [
    ["CIFAR100x5","CIFAR100x10"],
]
PLOT_TITLES = [
    ["CIFAR100x5 | ResNet32", "CIFAR100x10 | ResNet32"],
]

sns.set_style("whitegrid")


def plot_scores(
    scores: list, save_path: Path, method: str, setting: str, keep_org_label=False
):
    save_path.parent.mkdir(parents=True, exist_ok=True)

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
    ROOT_DIR = Path(__file__).parent.parent.parent.parent
    OUTPUT_DIR_PLOTS = ROOT_DIR / "icml_data" / " ablations"
    OUTPUT_DIR_TABLES = ROOT_DIR / "icml_data" / " tables"
    DOWNSAMPLE = True

    def filter_fn(path: Path):
        # if "vit" in str(path):
        #     return False
        if "pretrained_old" in str(path):
            return False
        string_path = str(path)

        if (
             "sparse" in string_path
            or "dense" in string_path
            or "cascading" in string_path
            or "ensembling" in string_path
            or "ex100" in string_path
            or "weighting" in string_path
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

        # # uncomment to plot per method results with original run labels
        # try:
        #     plot_scores(
        #         filtered_scores, save_path, method, setting, keep_org_label=True
        #     )
        # except:
        #     print("Failed plotting for setting", setting, "method", method)

    def setup_fn(x):
        if isinstance(x, str):
            if 'detach' in x:
                setup = "+LP"
            elif 'sdn' in x:
                setup = "+AC"
            else:
                setup = "Base"
        elif x["early_exit"]:
            if 'detach' in x["exp_name"]:
                setup = "+LP"
            else:
                setup = "+AC"
        else:
            setup = "Base"
        return setup

    def method_fn(x):
        if x.startswith("finetuning_ex0"):
            return "FT"
        elif x.startswith("finetuning_ex20"):
            return "FT+Ex"
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
            raise ValueError('Invalid method name: "{}"'.format(x))

    OUTPUT_DIR_PLOTS.mkdir(parents=True, exist_ok=True)

    # OLD PLOTTING CODE
    # for setting in SETTINGS:
    #     method_batches = [
    #         [
    #             "ANCL",
    #             "BiC",
    #             "DER++",
    #             "LODE",
    #             "SSIL",
    #         ],
    #         ["ER", "EWC", "FT", "FT+Ex", "GDumb", "LwF"],
    #     ]
    #     setting_scores = [s for s in averaged_scores if s.metadata.setting == setting]
    #     method_to_scores = {}
    #     for score in setting_scores:
    #         method_name = method_fn(score.metadata.exp_name)
    #         setup = "+AC" if score.early_exit else "Base"
    #         method_to_scores[(setup, method_name)] = score
    #
    #     for batch_idx, method_batch in enumerate(method_batches):
    #         plt.cla()
    #         plt.clf()
    #         plt.close("all")
    #
    #         if len(method_batch) == 5:
    #             figsize = 25
    #         elif len(method_batch) == 6:
    #             figsize = 30
    #         else:
    #             raise ValueError()
    #         fig, axes = plt.subplots(1, len(method_batch), figsize=(figsize, 4))
    #         for idx, method in enumerate(method_batch):
    #             try:
    #                 if isinstance(axes, Iterable):
    #                     ax = axes[idx]
    #                 else:
    #                     ax = axes
    #
    #                 score_base: ScoresStandard = method_to_scores[("Base", method)]
    #                 score_ee: ScoresEE = method_to_scores[("+AC", method)]
    #
    #                 sns.lineplot(
    #                     x=score_ee.per_th_cost,
    #                     y=score_ee.per_th_acc,
    #                     label=AC_NAME,
    #                     linewidth=LINEWIDTH,
    #                     zorder=2,
    #                     color=CMAP[AC_NAME],
    #                     ax=ax,
    #                     legend=idx == 5,
    #                 )
    #                 step_size = 5
    #                 marker_x, marker_y = (
    #                     score_ee.per_th_cost[1::step_size].tolist(),
    #                     score_ee.per_th_acc[1::step_size].tolist(),
    #                 )
    #                 marker_x.append(score_ee.per_th_cost[-1])
    #                 marker_y.append(score_ee.per_th_acc[-1])
    #                 ax.scatter(
    #                     marker_x, marker_y, color=CMAP[AC_NAME], zorder=3, s=MARKERSIZE
    #                 )
    #                 ax.fill_between(
    #                     score_ee.per_th_cost,
    #                     score_ee.per_th_acc - score_ee.per_th_std,
    #                     score_ee.per_th_acc + score_ee.per_th_std,
    #                     alpha=0.2,
    #                     color=CMAP[AC_NAME],
    #                 )
    #
    #                 min_cost = min(score_ee.per_th_cost)
    #                 sns.lineplot(
    #                     x=[min_cost, 1.0],
    #                     y=[score_base.tag_acc_final, score_base.tag_acc_final],
    #                     label=BASELINE_NAME,
    #                     linestyle="dashed",
    #                     linewidth=LINEWIDTH,
    #                     zorder=1,
    #                     color=CMAP[BASELINE_NAME],
    #                     ax=ax,
    #                     legend=idx == 5,
    #                 )
    #                 ax.fill_between(
    #                     [min_cost, 1],
    #                     score_base.tag_acc_final - score_base.tag_acc_std,
    #                     score_base.tag_acc_final + score_base.tag_acc_std,
    #                     alpha=0.2,
    #                     color=CMAP[BASELINE_NAME],
    #                 )
    #
    #                 # Set fontsize for xticks and yticks
    #                 ax.tick_params(axis="both", which="major", labelsize=FONTSIZE_TICKS)
    #                 ax.set_xlabel("Cost", fontsize=FONTSIZE_TITLE)
    #                 if idx == 0:
    #                     ax.set_ylabel("Accuracy", fontsize=FONTSIZE_TITLE)
    #                 else:
    #                     ax.set_ylabel(None)
    #                 ax.set_title(
    #                     f"{setting.replace('_rn18', '').replace('_vit', '')} | {method}",
    #                     fontsize=FONTSIZE_TITLE,
    #                 )
    #             except:
    #                 continue
    #
    #         last_ax = axes[-1] if isinstance(axes, Iterable) else axes
    #         try:
    #             handles, labels = last_ax.get_legend_handles_labels()
    #
    #             handles = [
    #                 handles[labels.index(BASELINE_NAME)],
    #                 handles[labels.index(AC_NAME)],
    #             ]
    #             labels = [BASELINE_NAME, AC_NAME]
    #             last_ax.legend(
    #                 handles, labels, fontsize=FONTSIZE_LEGEND, loc="lower right"
    #             )
    #         except:
    #             last_ax.legend(fontsize=FONTSIZE_LEGEND, loc="lower right")
    #
    #         fig.tight_layout()
    #         fig.savefig(OUTPUT_DIR_PLOTS / f"{setting}_{batch_idx}.pdf")
    #         # fig.savefig(OUTPUT_DIR_PLOTS / f"{setting}_{batch_idx}.png")

    # ------------------------------------
    # NEW VERSION OF THE PLOT
    PLOT_METHODS = [
        ["ANCL"],
        ["BiC"],
        ["FT"],
        ["FT+Ex"],
        ["ER"],
        ["EWC"],
        ["GDumb"],
        ["LODE"],
        ["LwF"],
        ["SSIL"],
    ]
    for batch_idx, method_batch in enumerate(PLOT_METHODS):
        for combined_plot_settings, combined_plot_titles in zip(
            PLOT_SETTINGS, PLOT_TITLES
        ):
            plt.cla()
            plt.clf()
            plt.close("all")
            fig, axes = plt.subplots(1, len(combined_plot_settings), figsize=FIGSIZE_APPENDIX_SINGLE_METHOD_ABLATIONS)

            for ax_idx, (setting, title) in enumerate(
                zip(combined_plot_settings, combined_plot_titles)
            ):
                setting_scores = [
                    s for s in averaged_scores if s.metadata.setting == setting
                ]
                method_to_scores = {}
                for score in setting_scores:
                    method_name = method_fn(score.metadata.exp_name)
                    setup = setup_fn(score.metadata.exp_name)
                    method_to_scores[(setup, method_name)] = score

                # Get all method scores in the setup that have the name in method_batch
                plot_data = [
                    s
                    for (setup, method_name), s in method_to_scores.items()
                    if method_name in method_batch
                ]
                # Create df for the method

                # Get early exit data
                scores_ee = [s for s in plot_data if s.early_exit]
                scores_base = [s for s in plot_data if not s.early_exit]

                early_exit_scores_combined = []
                # Create dataframes from scores
                for score_ee in scores_ee:
                    early_exit_scores_combined.extend(
                        [
                            {
                                "method": method_fn(score_ee.metadata.exp_name),
                                "setup": setup_fn(score_ee.metadata.exp_name),
                                "cost": cost,
                                "acc": acc,
                                "std": std,
                            }
                            for cost, acc, std in zip(
                                score_ee.per_th_cost.tolist(),
                                score_ee.per_th_acc.tolist(),
                                score_ee.per_th_std.tolist(),
                            )
                        ]
                    )
                df_ee = pd.DataFrame(early_exit_scores_combined)
                min_cost = df_ee["cost"].min()
                max_cost = df_ee["cost"].max()

                baseline_scores_combined = []
                for score_base in scores_base:
                    baseline_scores_combined.extend(
                        [
                            {
                                "method": method_fn(score_base.metadata.exp_name),
                                "setup": setup_fn(score_base.metadata.exp_name),
                                "cost": min_cost,
                                "acc": score_base.tag_acc_final,
                                "std": score_base.tag_acc_std,
                            },
                            {
                                "method": method_fn(score_base.metadata.exp_name),
                                "setup": setup_fn(score_base.metadata.exp_name),
                                "cost": max_cost,
                                "acc": score_base.tag_acc_final,
                                "std": score_base.tag_acc_std,
                            },
                        ]
                    )

                df_base = pd.DataFrame(baseline_scores_combined)
                df_combined = pd.concat([df_base, df_ee], axis=0)
                df_combined = df_combined.sort_values(
                    by=["setup", "method", "cost"], ascending=[False, True, True]
                )
                df_combined["label"] = df_combined.apply(
                    lambda x: x["method"]
                    + (" " + x["setup"] if x["setup"] != "Base" else ""),
                    axis=1,
                )
                df_combined["cost"] = df_combined["cost"] * 100
                # Create colormap per method
                palette = {
                    label: METHOD_TO_COLOR[method]
                    for method, label in zip(
                        df_combined["method"], df_combined["label"]
                    )
                }
                # Make style dict that sets "method+AC" or base method to either solid or dashed line
                def style_fn(setup):
                    if setup == "Base":
                        return (0.5, 0.5)
                    elif setup == "+LP":
                        return (1.0, 1.0)
                    elif setup == "+AC":
                        return (1.0, 0.0)
                    else:
                        return False

                # Make style dict that sets "method+AC" or base method to either solid or dashed line
                style_dict = {
                    label: style_fn(setup)
                    for setup, label in zip(df_combined["setup"], df_combined["label"])
                }

                plot = sns.lineplot(
                    df_combined,
                    x="cost",
                    y="acc",
                    hue="label",
                    style="label",
                    dashes=style_dict,
                    palette=palette,
                    linewidth=LINEWIDTH,
                    ax=axes[ax_idx],
                )

                # For non-base plot a mark for first and every next 100 point
                step_size = 10
                df_markers = df_combined[df_combined["setup"] != "Base"]
                for label in df_markers["label"].unique():
                    df_method = df_markers[df_markers["label"] == label]
                    cost_data = df_method["cost"].tolist()[::step_size]
                    acc_data = df_method["acc"].tolist()[::step_size]
                    plot.scatter(
                        cost_data,
                        acc_data,
                        color=palette[label],
                        marker="o",
                        s=MARKERSIZE,
                    )

                # Fill gaps between methods depending on std
                # Use the color of the method from the palette
                # for label, color in palette.items():
                #     df_method = df_combined[df_combined["label"] == label]
                #     plot.fill_between(
                #         df_method["cost"],
                #         df_method["acc"] - df_method["std"],
                #         df_method["acc"] + df_method["std"],
                #         color=palette[label],
                #         alpha=ALPHA,
                #     )

                plot.set_title(title, fontsize=FONTSIZE_TITLE)
                plot.set_xlabel("Inference cost [%]", fontsize=FONTSIZE_LABELS)
                if ax_idx == 0:
                    plot.set_ylabel("Accuracy", fontsize=FONTSIZE_LABELS)
                plot.tick_params(axis="both", which="major", labelsize=FONTSIZE_TICKS)
                if ax_idx != 0:
                    plot.set_ylabel(None)

                if ax_idx == len(combined_plot_settings) - 1:
                    # Set legend fontsize
                    handles, labels = plot.get_legend_handles_labels()
                    # Sort according to label names
                    handles, labels = zip(
                        *sorted(zip(handles, labels), key=lambda t: t[1])
                    )

                    def label_fn(l):
                        if "AC" in l:
                            return "+AC"
                        elif 'LP' in l:
                            return "+LP"
                        else:
                            return l

                    # Change labels to either method name or +AC if it contains +AC
                    labels = [label_fn(l)    for l in labels]
                    # Make handles shorter
                    plot.legend(
                        handles=handles,
                        labels=labels,
                        fontsize=FONTSIZE_LEGEND,
                        ncol=len(method_batch),
                        loc="lower right",
                        title=None,
                        handlelength=1,
                        columnspacing=0.8,
                    )
                else:
                    plot.get_legend().remove()
            plt.tight_layout()
            if "x5" in setting:
                n_tasks = 5
            elif "x10" in setting:
                n_tasks = 10
            else:
                raise ValueError()
            fig.savefig(OUTPUT_DIR_PLOTS / f"lp_{batch_idx}.pdf")

    # ------------------------------------
    # TABLE

    exit()

    OUTPUT_DIR_TABLES.mkdir(parents=True, exist_ok=True)
    table_data = load_scores_for_table(ROOT_DIR / "results", filter=filter_fn)
    table_data["setup"] = table_data.apply(setup_fn, axis=1)
    table_data["method"] = table_data["exp_name"].apply(method_fn)
    table_data = table_data[["setting", "method", "setup", "seed", "acc"]]
    table_data = table_data.sort_values(
        by=["setting", "method", "setup", "seed"], ascending=True
    )
    setups = ["Base", "+AC"]

    dfs_combined = []
    for setting in SETTINGS:
        setting_df = table_data[table_data["setting"] == setting]
        outputs_setting = []
        for setup in setups:
            method_to_scores = defaultdict(list)
            for method in METHODS.values():
                result_data = setting_df[
                    (setting_df["method"] == method) & (setting_df["setup"] == setup)
                ]
                for seed in sorted(result_data["seed"].unique().tolist()):
                    acc = result_data[result_data["seed"] == seed]["acc"]
                    if len(acc) != 1:
                        print(
                            f"Warning: more than one result for {setting}, {setup}, {method}, {seed}: {acc}"
                        )
                    method_to_scores[method].append(acc.values[0])
            num_results = max(len(v) for v in method_to_scores.values())
            if not all(len(v) == num_results for v in method_to_scores.values()):
                raise ValueError("Not all results have the same number of scores")
            for score_idx in range(num_results):
                output = {"setting": setting, "setup": setup}
                for method in METHODS.values():
                    output[method] = method_to_scores[method][score_idx]
                outputs_setting.append(output)
        df = pd.DataFrame(outputs_setting)
        df["Avg"] = df.values[:, 2:].mean(axis=1)
        base = df[df["setup"] == "Base"]
        ac = df[df["setup"] == "+AC"]
        diff = deepcopy(ac)
        diff["setup"] = "$\Delta$"
        diff.iloc[:, 2:] = np.array(ac.values[:, 2:]) - np.array(base.values[:, 2:])
        df = pd.concat([base, ac, diff], axis=0)
        df = (
            df.groupby(by=["setting", "setup"])
            .aggregate({m: ["mean", "std"] for m in list(METHODS.values()) + ["Avg"]})
            .reset_index()
        )
        for i, method_name in enumerate(list(METHODS.values()) + ["Avg"]):
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
        df.iloc[-1, 2:] = df.iloc[-1, 2:].apply(
            lambda x: x if x.startswith("-") else "+" + x
        )
        df.to_csv(OUTPUT_DIR_TABLES / f"{setting}.csv", index=False)
        dfs_combined.append(df)
    combined_df = pd.concat(dfs_combined, axis=0)
    combined_df.to_csv(OUTPUT_DIR_TABLES / "combined_main.csv", index=False)
