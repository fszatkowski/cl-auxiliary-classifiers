from copy import deepcopy

import torch
from torch import nn

from networks.ic_conifgs import CONFIGS
from networks.ic_utils import (
    create_ic,
    get_sdn_weights,
    register_intermediate_layer_hooks,
)


class LLL_Net(nn.Module):
    """Basic class for implementing networks"""

    def __init__(
        self,
        model,
        remove_existing_head=False,
        ic_config=None,
        ic_type=None,
        ic_layers=None,
        hook_placements=None,
        input_size=None,
        ic_weighting="sdn",
        detach_ics=False,
    ):
        head_var = model.head_var
        assert type(head_var) == str
        assert not remove_existing_head or hasattr(
            model, head_var
        ), "Given model does not have a variable called {}".format(head_var)
        assert not remove_existing_head or type(getattr(model, head_var)) in [
            nn.Sequential,
            nn.Linear,
        ], "Given model's head {} does is not an instance of nn.Sequential or nn.Linear".format(
            head_var
        )
        super(LLL_Net, self).__init__()

        self.model = model
        last_layer = getattr(self.model, head_var)

        if remove_existing_head:
            if type(last_layer) == nn.Sequential:
                self.out_size = last_layer[-1].in_features
                # strips off last linear layer of classifier
                del last_layer[-1]
            elif type(last_layer) == nn.Linear:
                self.out_size = last_layer.in_features
                # converts last layer into identity
                # setattr(self.model, head_var, nn.Identity())
                # WARNING: this is for when pytorch version is <1.2
                setattr(self.model, head_var, nn.Sequential())
        else:
            self.out_size = last_layer.out_features

        if ic_config is not None:
            assert (
                ic_config in CONFIGS
            ), f"IC config {ic_config} not found. Please provide one of {CONFIGS.keys()}"
            ic_config = CONFIGS[ic_config]
            ic_type = ic_config["ic_type"]
            ic_layers = ic_config["ic_layers"]
            hook_placements = ic_config["hook_placements"]
            input_size = ic_config["input_size"]
            ic_weighting = ic_config["ic_weighting"]
            detach_ics = ic_config["detach_ics"]

        if ic_layers is None:
            ic_layers = []
        self.ic_layers = ic_layers
        if len(self.ic_layers) > 0:
            assert len(ic_layers) == len(hook_placements), (
                f"Expected the same number of IC layers and hook placements, "
                f"but got {len(ic_layers)} and {len(hook_placements)}"
            )
            assert len(ic_type) == len(hook_placements) + 1
            # For early exits, create heads list per each IC
            self.ic_type = ic_type
            self.heads = nn.ModuleList()
            self.ic_input_sizes = []
            self.intermediate_layer_hooks = register_intermediate_layer_hooks(
                model, ic_layers, hook_placements
            )
            assert len(self.intermediate_layer_hooks) == len(ic_layers)

            model.eval()
            with torch.no_grad():
                x = torch.rand(1, *input_size)
                model(x)
            for hook in self.intermediate_layer_hooks:
                self.heads.append(nn.ModuleList())
                ic_input_size = hook.output.shape
                self.ic_input_sizes.append(ic_input_size)
            self.heads.append(nn.ModuleList())
            assert len(self.heads) == len(ic_layers) + 1
        else:
            self.heads = nn.ModuleList()

        self.ic_weighting = ic_weighting
        self.detach_ics = detach_ics
        self.exit_layer_idx = None
        self.task_cls = []
        self.task_offset = []
        self._initialize_weights()

    def add_head(self, num_outputs):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets
        """
        if len(self.ic_layers) == 0:
            self.heads.append(nn.Linear(self.out_size, num_outputs))
        else:
            for i in range(len(self.heads) - 1):
                if "ensembling" in self.ic_type[i]:
                    num_prev_heads = i
                else:
                    num_prev_heads = None
                self.heads[i].append(
                    create_ic(
                        self.ic_type[i],
                        self.ic_input_sizes[i],
                        num_outputs,
                        num_prev_heads,
                    )
                )
            if "ensembling" in self.ic_type[-1]:
                num_prev_heads = len(self.heads) - 1
            else:
                num_prev_heads = None
            self.heads[-1].append(
                create_ic(self.ic_type[-1], self.out_size, num_outputs, num_prev_heads)
            )

        # we re-compute instead of append in case an approach makes changes to the heads
        if len(self.ic_layers) == 0:
            final_heads = self.heads
        else:
            final_heads = self.heads[-1]
        self.task_cls = torch.tensor(list(self.task_cls) + [num_outputs])
        self.task_offset = torch.cat(
            [torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]]
        )

    def forward(self, x, return_features=False):
        """Applies the forward pass

        Simplification to work on multi-head only -- returns all head outputs in a list
        Args:
            x (tensor): input images
            return_features (bool): return the representations before the heads
        """
        if len(self.ic_layers) == 0:
            x = self.model(x)
            assert len(self.heads) > 0, "Cannot access any head"
            y = []
            for head in self.heads:
                y.append(head(x))
            if return_features:
                return y, x
            else:
                return y
        else:
            features = []
            outputs = []

            final_features = self.model(x)
            for i, feature_hook in enumerate(self.intermediate_layer_hooks):
                intermediate_output = feature_hook.output
                features.append(intermediate_output)
            features.append(final_features)

            assert len(features) == len(self.ic_layers) + 1

            for ic_idx in range(len(features)):
                ic_heads = self.heads[ic_idx]
                ic_features = features[ic_idx]

                head_outputs = []
                # For linear probing classifier detach the heads
                if self.detach_ics and ic_idx != len(self.heads) - 1:
                    ic_features = ic_features.detach()

                for head_idx in range(len(ic_heads)):
                    task_head = ic_heads[head_idx]
                    if "cascading" in self.ic_type[ic_idx]:
                        prev_output = outputs[ic_idx - 1][head_idx]
                        head_outputs.append(
                            task_head(ic_features, prev_output.detach())
                        )
                    elif "ensembling" in self.ic_type[ic_idx]:
                        prev_output = [o[head_idx] for o in outputs[:ic_idx]]
                        head_outputs.append(
                            task_head(ic_features, [o.detach() for o in prev_output])
                        )
                    else:
                        head_outputs.append(task_head(ic_features))

                outputs.append(head_outputs)
            assert len(outputs) == len(self.ic_layers) + 1

            if self.exit_layer_idx == -1:
                # for baseline no-ee net profiling
                if return_features:
                    return outputs[-1], features[-1]
                else:
                    return outputs[-1]
            elif self.exit_layer_idx is not None:
                # for ee cost profiling
                if return_features:
                    return (
                        outputs[: self.exit_layer_idx + 1],
                        features[: self.exit_layer_idx + 1],
                    )
                else:
                    return outputs[: self.exit_layer_idx + 1]
            else:
                # standard case
                if return_features:
                    return outputs, features
                else:
                    return outputs

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        """Freeze all parameters from the main model, but not the heads"""
        for param in self.model.parameters():
            param.requires_grad = False

    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def unfreeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad = True

    def unfreeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = True

    def freeze_last_head(self):
        for param in self.heads[-1].parameters():
            param.requires_grad = False

    def unfreeze_last_head(self):
        for param in self.heads[-1].parameters():
            param.requires_grad = True

    def _initialize_weights(self):
        """Initialize weights using different strategies"""
        # TODO: add different initialization strategies
        pass

    def is_early_exit(self):
        return len(self.ic_layers) > 0

    def get_ic_weights(self, current_epoch, max_epochs):
        if self.ic_weighting == "sdn":
            weights = get_sdn_weights(
                current_epoch, max_epochs, n_ics=len(self.ic_layers)
            )
        elif self.ic_weighting == "uniform":
            weights = [1.0] * (len(self.ic_layers) + 1)
        elif self.ic_weighting == "proportional":
            step = 1 / (len(self.ic_layers) + 1)
            weights = [step * (i + 1) for i in range(len(self.ic_layers) + 1)]
        else:
            raise NotImplementedError("Unknown IC weighting: " + self.ic_weighting)

        assert len(weights) == len(self.ic_layers) + 1
        return weights

    def set_exit_layer(self, layer_idx: int):
        self.exit_layer_idx = layer_idx
