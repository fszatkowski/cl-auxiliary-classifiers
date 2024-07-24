import math
from typing import Tuple

import torch
from torch import nn


def create_ic(ic_type: str, ic_input_size: Tuple[int, ...], num_outputs: int):
    if ic_type == "standard_conv":
        return StandardConvHead(ic_input_size, num_outputs)
    elif ic_type == "standard_cascading_conv":
        return StandardCascadingConvHead(ic_input_size, num_outputs)
    elif ic_type == "adaptive_standard_conv":
        return AdaptiveStandardConvHead(ic_input_size, num_outputs, 2)
    elif ic_type == "adaptive_standard_conv_4x":
        return AdaptiveStandardConvHead(ic_input_size, num_outputs, 4)
    elif ic_type == "adaptive_standard_conv_8x":
        return AdaptiveStandardConvHead(ic_input_size, num_outputs, 8)
    elif ic_type == "downsample_4x_conv":
        return DownsampleConvHead(ic_input_size, num_outputs, downsample_factor=4)
    elif ic_type == "downsample_8x_conv":
        return DownsampleConvHead(ic_input_size, num_outputs, downsample_factor=8)
    elif ic_type == "downsample_16x_conv":
        return DownsampleConvHead(ic_input_size, num_outputs, downsample_factor=16)
    elif ic_type == "standard_fc":
        if isinstance(ic_input_size, int):
            num_ic_features = ic_input_size
        else:
            num_ic_features = math.prod(ic_input_size)
        return nn.Linear(num_ic_features, num_outputs)
    elif ic_type == "cascading_fc":
        return CascadingLinear(ic_input_size, num_outputs)
    elif ic_type == "standard_transformer":
        return StandardTransformerHead(ic_input_size, num_outputs)
    elif ic_type == "ln_transformer":
        return LNTransformerHead(ic_input_size, num_outputs)
    else:
        raise NotImplementedError()


class AdaptiveStandardConvHead(nn.Module):
    def __init__(
        self, input_features: Tuple[int, ...], num_classes: int, downsample_factor: int
    ):
        super().__init__()
        _, c, h, w = input_features
        h_new = math.ceil(h / downsample_factor)
        w_new = math.ceil(w / downsample_factor)
        self.maxpool = nn.AdaptiveMaxPool2d((h_new, w_new))
        self.avgpool = nn.AdaptiveAvgPool2d((h_new, w_new))
        self.alpha = nn.Parameter(torch.rand(1))
        self.classifier = nn.Linear(c * h_new * w_new, num_classes)

    def forward(self, x, return_features=False):
        pool_output = self.alpha * self.maxpool(x) + (1 - self.alpha) * self.avgpool(x)
        cls_output = self.classifier(pool_output.view(pool_output.size(0), -1))
        if return_features:
            return cls_output, pool_output
        else:
            return cls_output


class AdaptiveDownsampleConvHead(nn.Module):
    def __init__(self, input_features: Tuple[int, ...], num_classes: int):
        super().__init__()
        _, c, h, w = input_features
        h_new = math.ceil(h / 2)
        w_new = math.ceil(w / 2)
        self.maxpool = nn.AdaptiveMaxPool2d((h_new, w_new))
        self.avgpool = nn.AdaptiveAvgPool2d((h_new, w_new))
        self.alpha = nn.Parameter(torch.rand(1))
        self.classifier = nn.Linear(c * h_new * w_new, num_classes)

    def forward(self, x, return_features=False):
        pool_output = self.alpha * self.maxpool(x) + (1 - self.alpha) * self.avgpool(x)
        cls_output = self.classifier(pool_output.view(pool_output.size(0), -1))
        if return_features:
            return cls_output, pool_output
        else:
            return cls_output


class DownsampleConvHead(nn.Module):
    def __init__(
        self, input_features: Tuple[int, ...], num_classes: int, downsample_factor: int
    ):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.alpha = nn.Parameter(torch.rand(1))
        num_input_features = math.prod(input_features)
        self.downsample_conv = nn.Conv2d(
            input_features[-3],
            input_features[-3] // downsample_factor,
            kernel_size=(1, 1),
        )
        self.classifier = nn.Linear(
            num_input_features // (downsample_factor * 4), num_classes
        )

    def forward(self, x, return_features=False):
        pool_output = self.alpha * self.maxpool(x) + (1 - self.alpha) * self.avgpool(x)
        pool_output = self.downsample_conv(pool_output)
        cls_output = self.classifier(pool_output.view(pool_output.size(0), -1))
        if return_features:
            return cls_output, pool_output
        else:
            return cls_output


class StandardConvHead(nn.Module):
    def __init__(self, input_features: Tuple[int, ...], num_classes: int):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.alpha = nn.Parameter(torch.rand(1))
        num_input_features = math.prod(input_features)
        self.classifier = nn.Linear(num_input_features // 4, num_classes)

    def forward(self, x, return_features=False):
        pool_output = self.alpha * self.maxpool(x) + (1 - self.alpha) * self.avgpool(x)
        cls_output = self.classifier(pool_output.view(pool_output.size(0), -1))
        if return_features:
            return cls_output, pool_output
        else:
            return cls_output


class StandardCascadingConvHead(nn.Module):
    def __init__(self, input_features: Tuple[int, ...], num_classes: int):
        # Convolutional head similar to one in ZTW paper, but without ensembling
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.alpha = nn.Parameter(torch.rand(1))
        num_input_features = math.prod(input_features)
        self.classifier = nn.Linear(num_classes + num_input_features // 4, num_classes)

    def forward(self, x, prev_out, return_features=False):
        pool_output = self.alpha * self.maxpool(x) + (1 - self.alpha) * self.avgpool(x)
        cls_input = pool_output.view(pool_output.size(0), -1)
        cls_input = torch.cat([cls_input, prev_out.view(prev_out.size(0), -1)], dim=1)
        cls_output = self.classifier(cls_input)
        if return_features:
            return cls_output, pool_output
        else:
            return cls_output


class CascadingLinear(nn.Module):
    def __init__(self, input_features: Tuple[int, ...], num_classes: int):
        super().__init__()
        if isinstance(input_features, int):
            num_input_features = input_features
        else:
            num_input_features = math.prod(input_features)
        self.classifier = nn.Linear(num_classes + num_input_features, num_classes)

    def forward(self, x, prev_out, return_features=False):
        cls_input = torch.cat([x, prev_out.view(prev_out.size(0), -1)], dim=1)
        cls_output = self.classifier(cls_input)
        if return_features:
            return cls_output, cls_input
        else:
            return cls_output


class StandardTransformerHead(nn.Module):
    def __init__(self, input_features: Tuple[int, ...], num_classes: int):
        # Transformer head with CLS token pooling, layer norm and classifier
        super().__init__()
        assert (
            len(input_features) == 3
        )  # assert [batch_size,num_tokens, hidden_dim] shape
        input_features = input_features[2]
        self.classifier = nn.Linear(input_features, num_classes)

    def forward(self, x, return_features=False):
        assert len(x.shape) == 3  # Assert [batch_size, hidden_dim, num_tokens] shape
        cls_input = x[:, 0, :]  # Only use CLS token features for classification
        cls_output = self.classifier(cls_input)
        if return_features:
            return cls_output, cls_input
        else:
            return cls_output


class LNTransformerHead(nn.Module):
    def __init__(self, input_features: Tuple[int, ...], num_classes: int):
        # Transformer head with CLS token pooling, layer norm and classifier
        super().__init__()
        assert (
            len(input_features) == 3
        )  # assert [batch_size,num_tokens, hidden_dim] shape
        input_features = input_features[2]
        self.ln = nn.LayerNorm(input_features)
        self.classifier = nn.Linear(input_features, num_classes)

    def forward(self, x, return_features=False):
        assert len(x.shape) == 3  # Assert [batch_size, hidden_dim, num_tokens] shape
        cls_input = self.ln(
            x[:, 0, :]
        )  # Only use CLS token features for classification
        cls_output = self.classifier(cls_input)
        if return_features:
            return cls_output, cls_input
        else:
            return cls_output


def forward(self, x, prev_out, return_features=False):
    pool_output = self.alpha * self.maxpool(x) + (1 - self.alpha) * self.avgpool(x)
    cls_input = pool_output.view(pool_output.size(0), -1)
    cls_input = torch.cat([cls_input, prev_out.view(prev_out.size(0), -1)], dim=1)
    cls_output = self.classifier(cls_input)
    if return_features:
        return cls_output, pool_output
    else:
        return cls_output


def get_sdn_weights(current_epoch, total_epochs, n_ics):
    if n_ics == 6:
        final_weights = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
    else:
        raise NotImplementedError(
            f"Cannot provide SDN weights for the network with {n_ics} ICs"
        )

    start_val = 0.01
    current_weights = [
        start_val + (current_epoch / (total_epochs - 1)) * (final_weight - start_val)
        for final_weight in final_weights
    ]
    current_weights += [1.0]
    return current_weights


def get_alt_sdn_weights(current_epoch, total_epochs):
    starting_weights = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    current_weights = [
        starting_weight + (current_epoch / (total_epochs - 1)) * (1 - starting_weight)
        for starting_weight in starting_weights
    ]
    return current_weights


class RegisterForwardHook:
    def __init__(self, mode: str):
        self.mode = mode
        self.output = None

    def __call__(self, module, input, output):
        if self.mode == "output":
            self.output = output
        elif self.mode == "input":
            if isinstance(input, tuple):
                # Handle cases like ViT where hook input might be a tuple
                input = input[0]
            self.output = input
        else:
            raise NotImplementedError('Hook mode should be either "output" or "input"')


def register_intermediate_layer_hooks(model, layers, hook_placements):
    assert len(layers) == len(hook_placements), (
        "Must provide the same number of layers and hook placements,"
        "but received {} layers and {} hooks".format(len(layers), len(hook_placements))
    )

    hooks = []
    modules = [(name, module) for name, module in model.named_modules()]
    for layer_name, hook_placement in zip(layers, hook_placements):
        module_found = False
        for module_name, module in modules:
            if module_name == layer_name:
                hook = RegisterForwardHook(mode=hook_placement)
                module.register_forward_hook(hook)
                hooks.append(hook)
                print(f"Attaching IC to the layer {layer_name}...")
                module_found = True
                break
        if not module_found:
            raise ValueError(f"Could not find module {layer_name} to attach the IC.")
    return hooks
