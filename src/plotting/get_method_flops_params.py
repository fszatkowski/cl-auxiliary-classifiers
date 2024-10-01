from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torchvision.models import resnet18, VisionTransformer, vit_b_16
from tqdm import tqdm

from networks import resnet32
from networks.network import LLL_Net
from plotting.utils import decode_path


def get_flops(path):
    results_dir = list(path.rglob("ee_eval.npy"))
    data = np.load(results_dir[0], allow_pickle=True).item()["avg"]
    baseline_cost = data["baseline_cost"] / 1e6
    ee_model_cost = data["exit_costs"][-1] / 1e6
    print(f"{path} baseline_cost: {baseline_cost}M, ee_model_cost: {ee_model_cost}M")


if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent.parent.parent
    print(f"ROOT_DIR: {ROOT_DIR}")

    ft_cifar100x5_dir_3ac = (
        ROOT_DIR
        / "results"
        / "CIFAR100x5"
        / "finetuning_ex0_cifar100_resnet32_sparse"
        / "seed0"
    )
    get_flops(ft_cifar100x5_dir_3ac)

    ft_cifar100x5_dir_6ac = (
        ROOT_DIR
        / "results"
        / "CIFAR100x5"
        / "finetuning_ex0_cifar100_resnet32_sdn"
        / "seed0"
    )
    get_flops(ft_cifar100x5_dir_6ac)

    ft_cifar100x5_dir_12ac = (
        ROOT_DIR
        / "results"
        / "CIFAR100x5"
        / "finetuning_ex0_cifar100_resnet32_dense"
        / "seed0"
    )
    get_flops(ft_cifar100x5_dir_12ac)

    ft_cifar100x5_dir_12ac = (
        ROOT_DIR
        / "results"
        / "CIFAR100x5"
        / "finetuning_ex0_cifar100_resnet32_sdn_cascading"
        / "seed0"
    )
    get_flops(ft_cifar100x5_dir_12ac)

    ft_cifar100x5_dir_12ac = (
        ROOT_DIR
        / "results"
        / "CIFAR100x5"
        / "finetuning_ex0_cifar100_resnet32_sdn_ensembling"
        / "seed0"
    )
    get_flops(ft_cifar100x5_dir_12ac)

    ft_cifar100x5_dir_3ac = (
        ROOT_DIR
        / "results"
        / "CIFAR100x10"
        / "finetuning_ex0_cifar100_resnet32_sparse"
        / "seed0"
    )
    get_flops(ft_cifar100x5_dir_3ac)

    ft_cifar100x5_dir_6ac = (
        ROOT_DIR
        / "results"
        / "CIFAR100x10"
        / "finetuning_ex0_cifar100_resnet32_sdn"
        / "seed0"
    )
    get_flops(ft_cifar100x5_dir_6ac)

    ft_cifar100x5_dir_12ac = (
        ROOT_DIR
        / "results"
        / "CIFAR100x10"
        / "finetuning_ex0_cifar100_resnet32_dense"
        / "seed0"
    )
    get_flops(ft_cifar100x5_dir_12ac)

    ft_cifar100x5_dir_6ac = (
        ROOT_DIR
        / "results"
        / "CIFAR100x10"
        / "finetuning_ex0_cifar100_resnet32_sdn_cascading"
        / "seed0"
    )
    get_flops(ft_cifar100x5_dir_6ac)

    ft_cifar100x5_dir_12ac = (
        ROOT_DIR
        / "results"
        / "CIFAR100x10"
        / "finetuning_ex0_cifar100_resnet32_sdn_ensembling"
        / "seed0"
    )
    get_flops(ft_cifar100x5_dir_12ac)

    print()

    outputs = {}
    for ic_config in [
        None,
        "cifar100_resnet32_sparse",
        "cifar100_resnet32_sdn",
        "cifar100_resnet32_dense",
        "cifar100_resnet32_sdn_cascading",
        "cifar100_resnet32_sdn_ensembling",
    ]:
        net = LLL_Net(
            resnet32(),
            remove_existing_head=True,
            ic_config=ic_config,
        )
        for i in range(10):
            net.add_head(10)

        outputs[(10, ic_config)] = sum(p.numel() for p in net.parameters()) / 1e6

        net = LLL_Net(
            resnet32(),
            remove_existing_head=True,
            ic_config=ic_config,
        )
        for i in range(5):
            net.add_head(20)

        outputs[(5, ic_config)] = sum(p.numel() for p in net.parameters()) / 1e6

    for k, v in outputs.items():
        print(k, v)

    print()
    ft_rn18 = (
        ROOT_DIR
        / "results"
        / "ImageNet100x5_rn18"
        / "finetuning_ex0_imagenet100_resnet18_sdn"
        / "seed0"
    )
    get_flops(ft_rn18)

    ft_rn18 = (
        ROOT_DIR
        / "results"
        / "ImageNet100x10_rn18"
        / "finetuning_ex0_imagenet100_resnet18_sdn"
        / "seed0"
    )
    get_flops(ft_rn18)

    outputs = {}
    for ic_config in [None, "imagenet100_resnet18_sdn"]:
        rn18 = resnet18()
        rn18.head_var = "fc"
        net = LLL_Net(
            rn18,
            remove_existing_head=True,
            ic_config=ic_config,
        )
        for i in range(10):
            net.add_head(10)

        outputs[(10, ic_config)] = sum(p.numel() for p in net.parameters()) / 1e6

        rn18 = resnet18()
        rn18.head_var = "fc"
        net = LLL_Net(
            rn18,
            remove_existing_head=True,
            ic_config=ic_config,
        )
        for i in range(5):
            net.add_head(20)

        outputs[(5, ic_config)] = sum(p.numel() for p in net.parameters()) / 1e6

    for k, v in outputs.items():
        print(k, v)

    outputs = {}
    for ic_config in [None, "imagenet100_vit_ln"]:
        vit = vit_b_16()
        vit.head_var = "heads"
        net = LLL_Net(
            vit,
            remove_existing_head=True,
            ic_config=ic_config,
        )
        for i in range(10):
            net.add_head(10)

        outputs[(10, ic_config)] = sum(p.numel() for p in net.parameters()) / 1e6

        vit = vit_b_16()
        vit.head_var = "heads"
        net = LLL_Net(
            vit,
            remove_existing_head=True,
            ic_config=ic_config,
        )
        for i in range(5):
            net.add_head(20)

        outputs[(5, ic_config)] = sum(p.numel() for p in net.parameters()) / 1e6

    for k, v in outputs.items():
        print(k, v)
