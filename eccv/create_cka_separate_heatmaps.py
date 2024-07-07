from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import PIL.Image
import seaborn as sns
import torch
from tqdm import tqdm

# This was only done for presentation on Sapienza

if __name__ == "__main__":
    root = Path(__file__).parent
    data_paths = sorted(list((root / "cka").rglob("*.pt")))
    output_dir = root / "heatmaps"
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped_data = {}
    for path in tqdm(data_paths):
        exp_name = path.parent.parent.parent.name
        dataset = path.parent.parent.parent.parent.name
        data = torch.load(path, map_location="cpu")
        grouped_data[dataset, exp_name] = data
        # for feat_idx in range(7):
        #     heatmap_data = data[feat_idx].cpu().numpy()
        #     output_path = output_dir / f'{dataset}_{exp_name}_ic{feat_idx}.png'
        #     plt.figure()
        #     plt.clf()
        #     plt.cla()
        #     heatmap = sns.heatmap(heatmap_data, vmin=0, vmax=1, annot=True, cbar=False, fmt='.2f')
        #     heatmap.set_title(f'{dataset}_{exp_name}_ic{feat_idx}')
        #     plt.savefig(output_path)

        # # merge per-ic figures vertically
        # output_path = output_dir / f'merged_{dataset}_{exp_name}.png'
        # plt.figure()
        # plt.clf()
        # plt.cla()
        # imgs = [PIL.Image.open(output_dir / f'{dataset}_{exp_name}_ic{feat_idx}.png') for feat_idx in range(7)]
        # widths, heights = zip(*(i.size for i in imgs))
        # total_width = sum(widths)
        # max_height = max(heights)
        # new_im = PIL.Image.new('RGB', (total_width, max_height))
        # x_offset = 0
        # for im in imgs:
        #     new_im.paste(im, (x_offset, 0))
        #     x_offset += im.size[0]
        # new_im.save(output_path)

    titles = {
        ("CIFAR100x10", "finetuning_ex0_ee"): "CKA, CIFAR100x10, early exits",
        ("CIFAR100x10", "finetuning_ex0_ee_detach"): "CKA, CIFAR100x10, standard",
        ("CIFAR100x5", "finetuning_ex0_ee"): "CKA, CIFAR100x5, early exits",
        ("CIFAR100x5", "finetuning_ex0_ee_detach"): "CKA, CIFAR100x5, standard",
    }

    datasets = sorted(list(set([d for d, e in grouped_data])))
    for dataset in datasets:
        dataset_data = {
            (d, e): grouped_data[d, e] for d, e in grouped_data if d == dataset
        }
        exp_names = reversed(sorted(list(set([e for d, e in dataset_data]))))

        for i, exp_name in enumerate(exp_names):

            plt.cla()
            plt.clf()
            plt.figure()

            exp_data = grouped_data[(dataset, exp_name)]
            non_zero_entries = (exp_data != 0).float().sum(dim=1)
            non_zero_entries[:, 0] = 1
            means = exp_data.sum(dim=1) / non_zero_entries
            means = means[:, 1:]
            heatmap = sns.heatmap(
                means, vmin=0, vmax=1, annot=True, cbar=False, fmt=".2f"
            )
            heatmap.set_title(titles[(dataset, exp_name)])
            heatmap.set_xlabel("Task idx")
            # heatmap.set_ylabel('IC idx')
            # Increment the numbers in axis labels by 1
            heatmap.set_xticklabels([i + 1 for i in list(range(means.shape[1]))])
            heatmap.set_yticklabels(
                [
                    "layer1.block3",
                    "layer1.block5",
                    "layer2.block2",
                    "layer2.block4",
                    "layer3.block1",
                    "layer3.block3",
                    "final",
                ],
                rotation=30,
                va="top",
            )

            # Set fontsize for all labels and ticks
            for ax in [heatmap]:
                for item in (
                    [ax.title, ax.xaxis.label, ax.yaxis.label]
                    + ax.get_xticklabels()
                    + ax.get_yticklabels()
                ):
                    item.set_fontsize(12)
            plt.tight_layout()
            plt.savefig(str(output_dir / f"{dataset}_{i}.png"))
            plt.savefig(str(output_dir / f"{dataset}_{i}.pdf"))
