from functools import partial

import torch
from torch import nn

__all__ = [
    "convnext_base_cifar",
]


class LayerNormChannels(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.transpose(1, -1)
        x = self.norm(x)
        x = x.transpose(-1, 1)
        return x


class Residual(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.residual = nn.Sequential(*layers)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.after_skip = nn.Identity()

    def forward(self, x):
        result = x + self.gamma * self.residual(x)
        result = self.after_skip(result)
        return result


class ConvNeXtBlock(Residual):
    def __init__(self, channels, kernel_size, mult=4, p_drop=0.0):
        padding = (kernel_size - 1) // 2
        hidden_channels = channels * mult
        super().__init__(
            nn.Conv2d(
                channels, channels, kernel_size, padding=padding, groups=channels
            ),
            LayerNormChannels(channels),
            nn.Conv2d(channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, 1),
            nn.Dropout(p_drop),
        )


class DownsampleBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__(
            LayerNormChannels(in_channels),
            nn.Conv2d(in_channels, out_channels, stride, stride=stride),
        )


class Stage(nn.Sequential):
    def __init__(self, in_channels, out_channels, num_blocks, kernel_size, p_drop=0.0):
        layers = (
            []
            if in_channels == out_channels
            else [DownsampleBlock(in_channels, out_channels)]
        )
        layers += [
            ConvNeXtBlock(out_channels, kernel_size, p_drop=p_drop)
            for _ in range(num_blocks)
        ]
        super().__init__(*layers)


class ConvNeXtBody(nn.Sequential):
    def __init__(
        self, in_channels, channel_list, num_blocks_list, kernel_size, p_drop=0.0
    ):
        layers = []
        for out_channels, num_blocks in zip(channel_list, num_blocks_list):
            layers.append(
                Stage(in_channels, out_channels, num_blocks, kernel_size, p_drop)
            )
            in_channels = out_channels
        super().__init__(*layers)


class Stem(nn.Sequential):
    def __init__(self, in_channels, out_channels, patch_size):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, patch_size, stride=patch_size),
            LayerNormChannels(out_channels),
        )


# class Head(nn.Sequential):
#     def __init__(self, in_channels, classes):
#         super().__init__(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.LayerNorm(in_channels),
#             nn.Linear(in_channels, classes),
#         )


class ConvNeXt(nn.Module):
    def __init__(
        self,
        classes,
        channel_list,
        num_blocks_list,
        kernel_size,
        patch_size,
        in_channels=3,
        res_p_drop=0.0,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.stem = Stem(in_channels, channel_list[0], patch_size)
        self.features = ConvNeXtBody(
            channel_list[0], channel_list, num_blocks_list, kernel_size, res_p_drop
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(channel_list[-1]),
            nn.Linear(channel_list[-1], classes),
        )
        self.head_var = "classifier"
        self.reset_parameters()

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.zeros_(m.bias)
            elif isinstance(m, Residual):
                nn.init.zeros_(m.gamma)

    def separate_parameters(self):
        parameters_decay = set()
        parameters_no_decay = set()
        modules_weight_decay = (nn.Linear, nn.Conv2d)
        modules_no_weight_decay = (nn.LayerNorm, nn.BatchNorm2d)

        for m_name, m in self.named_modules():
            for param_name, param in m.named_parameters():
                full_param_name = f"{m_name}.{param_name}" if m_name else param_name

                if isinstance(m, modules_no_weight_decay):
                    parameters_no_decay.add(full_param_name)
                elif param_name.endswith("bias"):
                    parameters_no_decay.add(full_param_name)
                elif isinstance(m, Residual) and param_name.endswith("gamma"):
                    parameters_no_decay.add(full_param_name)
                elif isinstance(m, modules_weight_decay):
                    parameters_decay.add(full_param_name)

        # # sanity check
        # assert len(parameters_decay & parameters_no_decay) == 0
        # assert len(parameters_decay) + len(parameters_no_decay) == len(list(model.parameters()))

        return parameters_decay, parameters_no_decay


convnext_base_cifar = partial(
    ConvNeXt,
    classes=1000,
    channel_list=[64, 128, 256, 512],
    num_blocks_list=[2, 2, 2, 2],
    kernel_size=7,
    patch_size=1,
    res_p_drop=0.0,
)
