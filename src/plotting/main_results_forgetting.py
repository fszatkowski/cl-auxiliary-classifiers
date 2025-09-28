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

SETTINGS = ["CIFAR100x5", "CIFAR100x10"]


PLOT_SETTINGS = [
    ["CIFAR100x5", "CIFAR100x10"],
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


def compare_fn(averaged_runs):
    output = []
    for run in averaged_runs:
        setting = run.metadata.setting
        name = run.metadata.exp_name
        seeds = run.num_seeds
        if hasattr(run, "per_th_acc"):
            final_acc = run.per_th_acc[-1]
        else:
            final_acc = run.tag_acc_final
        output.append((setting, name, seeds, final_acc))

    output = sorted(output, key=lambda x: (x[0], x[1]), reverse=True)
    for setting, name, seeds, acc in output:
        print(f"{setting} - {name}: {acc} ({seeds} seeds)")


if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent.parent.parent.parent
    OUTPUT_DIR_PLOTS = ROOT_DIR / "icml_data" / " per_task_acc_plots"
    OUTPUT_DIR_TABLES = ROOT_DIR / "icml_data" / " forgetting_tables"
    DOWNSAMPLE = True

    averaged_scores = load_averaged_scores(
        ROOT_DIR / "results_extended_logging",
        downsample=DOWNSAMPLE,
        filter=None,
        per_task_acc=True,
    )
    # # TODO REMOVE
    # averaged_scores = [s for s in averaged_scores if s.num_seeds == 3]

    plot_data = []

    def setup_fn(x):
        if isinstance(x, str):
            if "sdn" in x:
                setup = "+AC"
            else:
                setup = "Base"
        elif x["early_exit"]:
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
        elif x.startswith("joint"):
            return "Joint"
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
    for score in averaged_scores:
        data = []
        setting = score.metadata.setting
        method = method_fn(score.metadata.exp_name)
        setup = setup_fn(score.metadata.exp_name)
        acc_matrix = score.per_task_acc_matrix
        std_matrix = score.per_task_acc_std
        num_tasks = acc_matrix.shape[1]
        for current_task_idx in range(num_tasks):
            for task_data_idx in range(current_task_idx + 1):
                if current_task_idx < task_data_idx:
                    continue

                acc = acc_matrix[current_task_idx, task_data_idx]
                std = std_matrix[current_task_idx, task_data_idx]
                data.append(
                    {
                        "setting": setting,
                        "method": method,
                        "setup": setup,
                        "task_data_idx": task_data_idx,
                        "current_task_idx": current_task_idx,
                        "acc": acc * 100,
                        "std": std * 100,
                    }
                )
        plot_data.extend(data)
    plot_df = pd.DataFrame(plot_data)

    dash_style = {
        "Base": (0.5, 0.5),
        "+AC": (1.0, 0.0),
    }
    methods = METHODS.values()

    plt.cla()
    plt.clf()
    fig, axes = plt.subplots(
        len(SETTINGS), len(methods), figsize=(5 * len(methods), 4 * len(SETTINGS))
    )

    for setting_idx, setting in enumerate(SETTINGS):
        for method_idx, method in enumerate(methods):

            fig_df = plot_df[
                (plot_df["setting"] == setting) & (plot_df["method"] == method)
            ]
            plot = sns.lineplot(
                data=fig_df,
                x="current_task_idx",
                y="acc",
                hue="task_data_idx",
                style="setup",
                markers=True,
                dashes=dash_style,
                linewidth=LINEWIDTH,
                palette="viridis",
                ax=axes[setting_idx, method_idx],
            )

            plot.set_title(f"{setting} | {method}", fontsize=FONTSIZE_TITLE)

            if setting_idx == 1:
                plot.set_xlabel("Current Task", fontsize=FONTSIZE_LABELS)
            else:
                plot.set_xlabel("")
            if method_idx == 0:
                plot.set_ylabel("Accuracy", fontsize=FONTSIZE_LABELS)
            else:
                plot.set_ylabel("")
            plot.tick_params(labelsize=FONTSIZE_TICKS)

            handles, labels = plot.get_legend_handles_labels()
            plot.legend(
                handles[-2:], labels[-2:], fontsize=FONTSIZE_LEGEND, loc="lower left"
            )

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR_PLOTS / f"combined.pdf")
    fig.savefig(OUTPUT_DIR_PLOTS / f"combined.png")

    # PLOT PER AC relative forgetting
    ac_indices = [0, 3, 6]
    per_ac_acc_data = []
    for score in averaged_scores:
        if not score.early_exit:
            continue

        data = []
        setting = score.metadata.setting
        method = method_fn(score.metadata.exp_name)
        setup = setup_fn(score.metadata.exp_name)
        per_ac_per_task_acc_matrix = score.per_ic_per_task_acc
        num_tasks = per_ac_per_task_acc_matrix.shape[0]
        for ac_idx in ac_indices:
            ac_acc_matrix = per_ac_per_task_acc_matrix[:, :, ac_idx]
            ac_forgetting_matrix = np.zeros((num_tasks, num_tasks))
            for current_task_idx in range(num_tasks):
                for task_data_idx in range(current_task_idx + 1):
                    if current_task_idx < task_data_idx:
                        continue
                    ac_forgetting_matrix[current_task_idx, task_data_idx] = (
                        ac_acc_matrix[task_data_idx, task_data_idx]
                        - ac_acc_matrix[current_task_idx, task_data_idx]
                    )
            # Compute relative forgetting as the amount of forgettng divided by the original task accuracy
            acc_relative_forgetting = np.zeros((num_tasks, num_tasks))
            for current_task_idx in range(num_tasks):
                for task_data_idx in range(current_task_idx + 1):
                    if current_task_idx < task_data_idx:
                        continue
                    acc_relative_forgetting[current_task_idx, task_data_idx] = (
                        ac_forgetting_matrix[current_task_idx, task_data_idx]
                        / ac_acc_matrix[task_data_idx, task_data_idx]
                    )
            # Update the list of dicts with the data for each task
            for current_task_idx in range(num_tasks):
                for task_data_idx in range(current_task_idx + 1):
                    data.append(
                        {
                            "setting": setting,
                            "method": method,
                            "setup": setup,
                            "task_data_idx": task_data_idx,
                            "current_task_idx": current_task_idx,
                            "forgetting": ac_forgetting_matrix[
                                current_task_idx, task_data_idx
                            ]
                            * 100,
                            "relative_forgetting": acc_relative_forgetting[
                                current_task_idx, task_data_idx
                            ]
                            * 100,
                            "ac_idx": ac_idx,
                        }
                    )
        per_ac_acc_data.extend(data)
    per_ac_plot_df = pd.DataFrame(per_ac_acc_data)

    for setting in SETTINGS:
        plt.cla()
        plt.clf()
        fig, axes = plt.subplots(
            len(ac_indices),
            len(methods),
            figsize=(5 * len(methods), 4 * len(ac_indices)),
        )
        for ac_enumerator, ac_idx in enumerate(ac_indices):
            for method_idx, method in enumerate(methods):
                ax_idx = ac_enumerator

                fig_df = per_ac_plot_df[
                    (per_ac_plot_df["setting"] == setting)
                    & (per_ac_plot_df["method"] == method)
                    & (per_ac_plot_df["ac_idx"] == ac_idx)
                ]
                plot = sns.lineplot(
                    data=fig_df,
                    x="current_task_idx",
                    y="relative_forgetting",
                    hue="task_data_idx",
                    style="setup",
                    markers=True,
                    dashes=dash_style,
                    linewidth=LINEWIDTH,
                    palette="viridis",
                    ax=axes[ax_idx, method_idx],
                )
                plot.set_title(
                    f"{setting} | {method} | AC {ac_idx}", fontsize=FONTSIZE_TITLE
                )

                if ax_idx == len(ac_indices) - 1:
                    plot.set_xlabel("Current Task", fontsize=FONTSIZE_LABELS)
                else:
                    plot.set_xlabel("")

                if method_idx == 0:
                    plot.set_ylabel("Relative Forgetting", fontsize=FONTSIZE_LABELS)
                else:
                    plot.set_ylabel("")

                plot.set_ylim(-20, 100)
                plot.set_xlim(0, fig_df["current_task_idx"].max())
                plot.tick_params(labelsize=FONTSIZE_TICKS)
                # Delete legend
                plot.get_legend().remove()
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR_PLOTS / f"per_ac_relative_forgetting_{setting}.png")

    # PLOT SEPARETE IMAGE FOR JUST FT
    ft_df = per_ac_plot_df[per_ac_plot_df["method"] == "FT"]

    plt.cla()
    plt.clf()
    fig, axes = plt.subplots(
        len(SETTINGS), len(ac_indices), figsize=(5 * len(ac_indices), 4 * len(SETTINGS))
    )
    for setting_idx, setting in enumerate(SETTINGS):
        for ac_enumerator, ac_idx in enumerate(ac_indices):
            ax_idx = ac_enumerator
            fig_df = ft_df[(ft_df["setting"] == setting) & (ft_df["ac_idx"] == ac_idx)]
            plot = sns.lineplot(
                data=fig_df,
                x="current_task_idx",
                y="relative_forgetting",
                hue="task_data_idx",
                style="setup",
                markers=True,
                dashes=dash_style,
                linewidth=LINEWIDTH,
                palette="viridis",
                ax=axes[setting_idx, ax_idx],
            )
            plot.set_title(f"{setting} | FT | AC {ac_idx}", fontsize=FONTSIZE_TITLE)
            if setting_idx == len(SETTINGS) - 1:
                plot.set_xlabel("Current Task", fontsize=FONTSIZE_LABELS)
            else:
                plot.set_xlabel("")

            if ac_idx == 0:
                plot.set_ylabel("Relative Forgetting", fontsize=FONTSIZE_LABELS)
            else:
                plot.set_ylabel("")

            plot.set_ylim(0, 100)
            plot.set_xlim(0, fig_df["current_task_idx"].max())
            plot.tick_params(labelsize=FONTSIZE_TICKS)
            # Delete legend
            plot.get_legend().remove()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR_PLOTS / f"per_ac_relative_forgetting_FT.png")

    # ------------------------------------
    # TABLE

    OUTPUT_DIR_TABLES.mkdir(parents=True, exist_ok=True)

    parsed_scores = []
    for score in averaged_scores:
        setting = score.metadata.setting
        exp_name = score.metadata.exp_name
        avg_forgetting = score.per_task_forgetting_matrix[-1, :-1].mean()
        forgetting_std = score.per_task_forgetting_std[-1, :-1].mean()
        parsed_scores.append(
            {
                "setting": setting,
                "method": method_fn(exp_name),
                "setup": setup_fn(exp_name),
                "avg_forgetting": avg_forgetting * 100,
                "forgetting_std": forgetting_std * 100,
            }
        )

    table_data = pd.DataFrame(parsed_scores)
    settings = table_data["setting"].unique()
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
    setups = [s for s in table_data["setup"].unique().tolist()]
    settings = sorted(settings, reverse=True)
    setups = sorted(setups, reverse=True)

    parsed_outputs = []
    for setting in settings:
        for setup in setups:
            output = {
                "Setting": setting,
                "Setup": setup,
            }
            vals_to_avg = []
            for method_name in methods:
                score = table_data[
                    (table_data["setting"] == setting)
                    & (table_data["method"] == method_name)
                    & (table_data["setup"] == setup)
                ]
                if len(score) == 0:
                    output[method_name] = None
                elif len(score) == 1:
                    score = score.iloc[0]
                    forgetting = score.avg_forgetting
                    forgetting_std = score.forgetting_std
                    vals_to_avg.append(forgetting)
                    val = f"{forgetting:.2f}$\pm${forgetting_std:.2f}"
                    output[method_name] = val
                elif len(score) > 1:
                    raise ValueError(
                        f"More than one score found for {setting}, {setup}, {method_name}"
                    )

            avg_val = np.mean(vals_to_avg)
            std_val = np.std(vals_to_avg)
            output["Avg"] = f"{avg_val:.2f}$\pm${std_val:.2f}"
            parsed_outputs.append(output)

    final_df = pd.DataFrame(parsed_outputs)
    final_df.to_csv(OUTPUT_DIR_TABLES / "combined_forgetting.csv", index=False)

    md_string = final_df.to_markdown(index=False)
    md_string = md_string.replace("\\tiny{", "").replace("}", "")
    md_string = md_string.replace(" ", "")
    md_output_path = OUTPUT_DIR_TABLES / "combined_forgetting.md"
    with open(md_output_path, "w") as f:
        f.write(md_string)
