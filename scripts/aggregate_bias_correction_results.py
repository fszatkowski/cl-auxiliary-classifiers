from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

sns.set_style("whitegrid")

if __name__ == "__main__":
    root_name = Path(__file__).parent.parent / 'bias_correction'
    setting_dirs = sorted(list(root_name.glob("*/*")))

    palette = {
        'baseline': 'tab:orange',
        'tlc': 'tab:red',
        'lctsgd': 'tab:blue',
    }

    for d in tqdm(setting_dirs):
        costs = defaultdict(list)
        accs = defaultdict(list)
        stat_files = list(d.rglob("*/eval_results/*.pt"))
        for file in stat_files:
            method = file.name.split('.pt')[0]
            data = torch.load(file)
            costs[method].append(data["cost_per_th"])
            accs[method].append(data["acc_per_th"])

        mean_costs = {method: torch.mean(torch.stack(costs[method], dim=0), dim=0) for method in costs.keys()}
        mean_accs = {method: torch.mean(torch.stack(accs[method], dim=0), dim=0) for method in accs.keys()}
        std_accs = {method: torch.std(torch.stack(accs[method], dim=0), dim=0) for method in accs.keys()}

        outputs = []
        for method in mean_costs:
            for cost, acc, std in zip(mean_costs[method], mean_accs[method], std_accs[method]):
                outputs.append({"method": method, "cost": float(cost), "acc": float(acc), "std": float(std)})
        df = pd.DataFrame(outputs)

        plt.figure()
        plt.cla()
        plt.clf()
        sns.lineplot(x="cost", y="acc", hue="method", data=df, palette=palette)
        # Plot std
        for method in mean_costs:
            y_1 = mean_accs[method] - std_accs[method]
            y_2 = mean_accs[method] + std_accs[method]
            plt.fill_between(mean_costs[method], y_1, y_2, alpha=0.2, color=palette[method])
        plt.savefig(str(d / "comparision.png"))
