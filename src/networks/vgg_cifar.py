"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""

"""VGG11/13/16/19 in Pytorch."""

import torch
import torch.nn as nn

__all__ = [
    "vgg11_bn_cifar",
    "vgg13_bn_cifar",
    "vgg16_bn_cifar",
    "vgg19_bn_cifar",
]

cfg = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):

    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features

        self.fc1 = nn.Linear(512, 4096)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.fc2 = nn.Linear(4096, 4096)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout()
        self.fc = nn.Linear(4096, num_class)

        self.head_var = "fc"

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        output = self.relu1(output)
        output = self.dropout1(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output = self.dropout2(output)
        output = self.fc(output)

        return output


def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)


def vgg11_bn_cifar(pretrained=False, **kwargs):
    return VGG(make_layers(cfg["A"], batch_norm=True))


def vgg13_bn_cifar(pretrained=False, **kwargs):
    return VGG(make_layers(cfg["B"], batch_norm=True))


def vgg16_bn_cifar(pretrained=False, **kwargs):
    return VGG(make_layers(cfg["D"], batch_norm=True))


def vgg19_bn_cifar(pretrained=False, **kwargs):
    return VGG(make_layers(cfg["E"], batch_norm=True))
