import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plotting.common import *
from results_utils.load_data import (
    load_averaged_scores,
)

BASELINE_NAME = "Base"
AC_NAME = "+AC"

PLOT_SETTINGS = [
    ["CIFAR100x5", "ImageNet100x5_vit"],
    ["CIFAR100x10", "ImageNet100x10_vit"],
]
PLOT_TITLES = [
    ["CIFAR100x5 | VGG19", "ImageNet100x5 | ViT-base"],
    ["CIFAR100x10 | VGG19", "ImageNet100x10 | ViT-base"],
]

CMAP = {
    BASELINE_NAME: "tab:red",
    AC_NAME: "tab:blue",
}

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
    OUTPUT_DIR_PLOTS = ROOT_DIR / "icml_data" / " dynamic_acc_plots"
    OUTPUT_DIR_TABLES = ROOT_DIR / "icml_data" / " tables"
    DOWNSAMPLE = True

    def filter_fn_vit(path: Path):
        if "vit" not in str(path):
            return False
        return True

    def filter_fn_vgg(path: Path):
        string_path = str(path)
        if "small" in string_path or "medium" in string_path:
            return False

        return True

    averaged_scores_vit = load_averaged_scores(
        ROOT_DIR / "results", downsample=DOWNSAMPLE, filter=filter_fn_vit
    )
    averaged_scores_vgg = load_averaged_scores(
        ROOT_DIR / "results_vgg19", downsample=DOWNSAMPLE, filter=filter_fn_vgg
    )
    averaged_scores = averaged_scores_vit + averaged_scores_vgg

    # averaged_scores = [s for s in averaged_scores if s.early_exit is False]
    method_setting_pairs = []

    for method in sorted(METHODS.keys()):
        method_scores = [
            s for s in averaged_scores if s.metadata.exp_name.startswith(method)
        ]
        unique_settings = set(s.metadata.setting for s in method_scores)
        for setting in sorted(unique_settings):
            method_setting_pairs.append((method, setting))

    def setup_fn(x):
        if isinstance(x, str):
            if "_ln" in x:
                setup = "+AC"
            elif "_dense" in x:
                setup = "+AC"
            else:
                setup = "Base"
        elif x["early_exit"]:
            if "_dense" in x["exp_name"]:
                setup = "+AC"
            elif "_ln" in x["exp_name"]:
                setup = "+AC"
            else:
                raise ValueError()
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

    # ------------------------------------
    # NEW VERSION OF THE PLOT
    PLOT_METHODS = [
        [
            "ANCL",
            "BiC",
            "LODE",
            "SSIL",
        ],
        ["DER++", "FT", "FT+Ex", "ER", "EWC", "GDumb", "LwF"],
    ]
    for batch_idx, method_batch in enumerate(PLOT_METHODS):
        for combined_plot_settings, combined_plot_titles in zip(
            PLOT_SETTINGS, PLOT_TITLES
        ):
            plt.cla()
            plt.clf()
            plt.close("all")
            fig, axes = plt.subplots(
                1, len(combined_plot_settings), figsize=FIGSIZE_MAIN_PAPER
            )

            for ax_idx, (setting, title) in enumerate(
                zip(combined_plot_settings, combined_plot_titles)
            ):
                setting_scores = [
                    s for s in averaged_scores if s.metadata.setting == setting
                ]
                method_to_scores = {}
                for score in setting_scores:
                    method_name = method_fn(score.metadata.exp_name)
                    setup = "+AC" if score.early_exit else "Base"
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
                style_dict = {
                    label: ((1, 0) if setup == "+AC" else (1.5, 1.5))
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
                    # Change labels to either method name or +AC if it contains +AC
                    labels = [l if not "+AC" in l else "+AC" for l in labels]
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
            fig.savefig(OUTPUT_DIR_PLOTS / f"vgg_vit_{n_tasks}_{batch_idx}.pdf")
            # fig.savefig(OUTPUT_DIR_PLOTS / f"vgg_vit_{n_tasks}_{batch_idx}.png")
