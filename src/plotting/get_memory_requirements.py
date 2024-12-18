import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import torch
from torchvision.models import resnet18, vit_b_16

from networks import resnet32, set_tvmodel_head_var, tvmodels, vgg19_bn_cifar
from networks.ic_configs import CONFIGS
from networks.network import LLL_Net

if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent.parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int)
    args = parser.parse_args()

    model_config_name = [
        ("resnet32", resnet32, "cifar100_resnet32_sparse", "ResNet32(+3AC)", True),
        ("resnet32", resnet32, "cifar100_resnet32_sdn", "ResNet32(+6AC)", True),
        ("resnet32", resnet32, "cifar100_resnet32_dense", "ResNet32(+12AC)", True),
        ("resnet18", resnet18, "imagenet100_resnet18_sdn", "ResNet18(+6AC)", True),
        (
            "vgg19_bn_cifar",
            vgg19_bn_cifar,
            "cifar100_vgg19_medium",
            "VGG19(+10AC)",
            True,
        ),
        (
            "vgg19_bn_cifar",
            vgg19_bn_cifar,
            "cifar100_vgg19_dense",
            "VGG19(+18AC)",
            True,
        ),
        ("vit_b_16", vit_b_16, "imagenet100_vit_base", "ViT-base(+11AC)", True),
        ("resnet32", resnet32, "cifar100_resnet32_sparse", "ResNet32(base)", False),
        ("resnet32", resnet32, "cifar100_resnet32_sdn", "ResNet32(base)", False),
        ("resnet32", resnet32, "cifar100_resnet32_dense", "ResNet32(base)", False),
        ("resnet18", resnet18, "imagenet100_resnet18_sdn", "ResNet18(base)", False),
        (
            "vgg19_bn_cifar",
            vgg19_bn_cifar,
            "cifar100_vgg19_medium",
            "VGG19(base)",
            False,
        ),
        (
            "vgg19_bn_cifar",
            vgg19_bn_cifar,
            "cifar100_vgg19_dense",
            "VGG19(base)",
            False,
        ),
        ("vit_b_16", vit_b_16, "imagenet100_vit_base", "ViT-base(base)", False),
    ]

    with torch.no_grad():
        model_name, model, ic_config, key, use_ic = model_config_name[args.idx]
        _model = model()
        if model_name in tvmodels:
            set_tvmodel_head_var(_model)
        if use_ic:
            net = LLL_Net(
                model=_model,
                remove_existing_head=True,
                ic_config=ic_config,
            )
        else:
            net = LLL_Net(
                model=_model,
                remove_existing_head=True,
            )

        for n_tasks in range(10):
            net.add_head(10)
        net = net.to("cuda")
        net.eval()

        x = torch.rand([1] + CONFIGS[ic_config]["input_size"]).to("cuda")
        torch.cuda.synchronize()
        net(x)
        memory = torch.cuda.max_memory_allocated(device=None)
        torch.cuda.reset_peak_memory_stats(device=None)

        params = sum(p.numel() for p in net.parameters())

        output = {
            "model": key,
            "params": str(round(params / 10**6, 2)) + "M",
            "memory": str(round(memory / 10**6, 2)) + "M",
        }

    output_dir = Path(ROOT_DIR) / "memory_analysis"
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(output_dir / f"{args.idx}.json", "w+") as f:
        json.dump(output, f)
