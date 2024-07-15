import time
from argparse import ArgumentParser
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.exemplars_dataset import ExemplarsDataset

from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    """Class implementing the Bias Correction (BiC) approach described in
    http://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Large_Scale_Incremental_Learning_CVPR_2019_paper.pdf
    Original code available at https://github.com/wuyuebupt/LargeScaleIncrementalLearning
    """

    def __init__(
        self,
        model,
        device,
        nepochs=100,
        optimizer_name="sgd",
        lr=0.05,
        lr_min=1e-4,
        lr_factor=3,
        lr_patience=5,
        clipgrad=10000,
        momentum=0,
        wd=0,
        multi_softmax=False,
        fix_bn=False,
        eval_on_train=False,
        select_best_model_by_val_loss=True,
        logger=None,
        exemplars_dataset=None,
        scheduler_milestones=None,
        scheduler_name="multistep",
        val_exemplar_percentage=0.1,
        num_bias_epochs=200,
        T=2,
        lamb=-1,
        lamb_warmup=None,
    ):
        # Sec. 6.1. CIFAR-100: 2,000 exemplars, ImageNet-1000: 20,000 exemplars, Celeb-10000: 50,000 exemplars
        # Sec. 6.2. weight decay for CIFAR-100 is 0.0002, for ImageNet-1000 and Celeb-10000 is 0.0001
        super(Appr, self).__init__(
            model=model,
            device=device,
            nepochs=nepochs,
            optimizer_name=optimizer_name,
            lr=lr,
            lr_min=lr_min,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            clipgrad=clipgrad,
            momentum=momentum,
            wd=wd,
            multi_softmax=multi_softmax,
            fix_bn=fix_bn,
            eval_on_train=eval_on_train,
            select_best_model_by_val_loss=select_best_model_by_val_loss,
            logger=logger,
            exemplars_dataset=exemplars_dataset,
            scheduler_name=scheduler_name,
            scheduler_milestones=scheduler_milestones,
        )
        self.val_percentage = val_exemplar_percentage
        self.bias_epochs = num_bias_epochs
        self.model_old = None
        self.T = T
        self.lamb = lamb
        self.lamb_warmup = (
            lamb_warmup  # TODO get rid of this warmup later if it does not help
        )
        if self.model.is_early_exit():
            self.bias_layers = torch.nn.ModuleList(
                [torch.nn.ModuleList([]) for _ in range(len(self.model.ic_layers) + 1)]
            )
        else:
            self.bias_layers = torch.nn.ModuleList([])
        self.x_valid_exemplars = []
        self.y_valid_exemplars = []

        if self.exemplars_dataset.max_num_exemplars != 0:
            self.num_exemplars = self.exemplars_dataset.max_num_exemplars
        elif self.exemplars_dataset.max_num_exemplars_per_class != 0:
            self.num_exemplars_per_class = (
                self.exemplars_dataset.max_num_exemplars_per_class
            )

        have_exemplars = (
            self.exemplars_dataset.max_num_exemplars
            + self.exemplars_dataset.max_num_exemplars_per_class
        )
        assert have_exemplars > 0, "Error: BiC needs exemplars."

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Sec. 3. "lambda is set to n / (n+m)" where n=num_old_classes and m=num_new_classes - so lambda is not a param
        # To use the original, set lamb=-1, otherwise, we allow to use specific lambda for the distillation loss
        parser.add_argument(
            "--lamb",
            default=-1,
            type=float,
            required=False,
            help="Forgetting-intransigence trade-off (default=%(default)s)",
        )
        parser.add_argument(
            "--lamb-warmup",
            default=None,
            type=int,
            required=False,
            help="Number of linear warmup steps for KD lamba (default=%(default)s)",
        )
        # Sec. 6.2. "The temperature scalar T in Eq. 1 is set to 2 by following [13,2]."
        parser.add_argument(
            "--T",
            default=2,
            type=int,
            required=False,
            help="Temperature scaling (default=%(default)s)",
        )
        # Sec. 6.1. "the ratio of train/validation split on the exemplars is 9:1 for CIFAR-100 and ImageNet-1000"
        parser.add_argument(
            "--val-exemplar-percentage",
            default=0.1,
            type=float,
            required=False,
            help="Percentage of exemplars that will be used for validation (default=%(default)s)",
        )
        # In the original code they define epochs_per_eval=100 and epoch_val_times=2, making a total of 200 bias epochs
        parser.add_argument(
            "--num-bias-epochs",
            default=200,
            type=int,
            required=False,
            help="Number of epochs for training bias (default=%(default)s)",
        )
        return parser.parse_known_args(args)

    def bias_forward(self, outputs):
        """Utility function --- inspired by https://github.com/sairin1202/BIC"""
        bic_outputs = []
        if self.model.is_early_exit():
            for ic_idx in range(len(outputs)):
                ic_bic_outputs = []
                ic_outputs = outputs[ic_idx]
                for m in range(len(ic_outputs)):
                    ic_bic_outputs.append(self.bias_layers[ic_idx][m](ic_outputs[m]))
                bic_outputs.append(ic_bic_outputs)
        else:
            for m in range(len(outputs)):
                bic_outputs.append(self.bias_layers[m](outputs[m]))
        return bic_outputs

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop
        Some parts could go into self.pre_train_process() or self.post_train_process(), but we leave it for readability
        """
        # add a bias layer for the new classes

        if self.model.is_early_exit():
            for bias_head_list in self.bias_layers:
                bias_head_list.append(BiasLayer().to(self.device, non_blocking=True))
        else:
            self.bias_layers.append(BiasLayer().to(self.device, non_blocking=True))

        # STAGE 0: EXEMPLAR MANAGEMENT -- select subset of validation to use in Stage 2 -- val_old, val_new (Fig.2)
        print("Stage 0: Select exemplars from validation")
        clock0 = time.time()

        # number of classes and proto samples per class
        num_cls = sum(self.model.task_cls)
        num_old_cls = sum(self.model.task_cls[:t])
        if self.exemplars_dataset.max_num_exemplars != 0:
            num_exemplars_per_class = int(np.floor(self.num_exemplars / num_cls))
            num_val_ex_cls = int(np.ceil(self.val_percentage * num_exemplars_per_class))
            num_trn_ex_cls = num_exemplars_per_class - num_val_ex_cls
            # Reset max_num_exemplars
            self.exemplars_dataset.max_num_exemplars = (num_trn_ex_cls * num_cls).item()
        elif self.exemplars_dataset.max_num_exemplars_per_class != 0:
            num_val_ex_cls = int(
                np.ceil(self.val_percentage * self.num_exemplars_per_class)
            )
            num_trn_ex_cls = self.num_exemplars_per_class - num_val_ex_cls
            # Reset max_num_exemplars
            self.exemplars_dataset.max_num_exemplars_per_class = num_trn_ex_cls

        # Remove extra exemplars from previous classes -- val_old
        if t > 0:
            if self.exemplars_dataset.max_num_exemplars != 0:
                num_exemplars_per_class = int(
                    np.floor(self.num_exemplars / num_old_cls)
                )
                num_old_ex_cls = int(
                    np.ceil(self.val_percentage * num_exemplars_per_class)
                )
                for cls in range(num_old_cls):
                    assert len(self.y_valid_exemplars[cls]) == num_old_ex_cls
                    self.x_valid_exemplars[cls] = self.x_valid_exemplars[cls][
                        :num_val_ex_cls
                    ]
                    self.y_valid_exemplars[cls] = self.y_valid_exemplars[cls][
                        :num_val_ex_cls
                    ]

        # Add new exemplars for current classes -- val_new
        non_selected = []
        for curr_cls in range(num_old_cls, num_cls):
            self.x_valid_exemplars.append([])
            self.y_valid_exemplars.append([])
            # get all indices from current class
            cls_ind = np.where(np.asarray(trn_loader.dataset.labels) == curr_cls)[0]
            assert len(cls_ind) > 0, "No samples to choose from for class {:d}".format(
                curr_cls
            )
            assert num_val_ex_cls <= len(
                cls_ind
            ), "Not enough samples to store for class {:d}".format(curr_cls)
            # add samples to the exemplar list
            self.x_valid_exemplars[curr_cls] = [
                trn_loader.dataset.images[idx] for idx in cls_ind[:num_val_ex_cls]
            ]
            self.y_valid_exemplars[curr_cls] = [
                trn_loader.dataset.labels[idx] for idx in cls_ind[:num_val_ex_cls]
            ]
            non_selected.extend(cls_ind[num_val_ex_cls:])
        # remove selected samples from the validation data used during training
        trn_loader.dataset.images = [
            trn_loader.dataset.images[idx] for idx in non_selected
        ]
        trn_loader.dataset.labels = [
            trn_loader.dataset.labels[idx] for idx in non_selected
        ]
        clock1 = time.time()
        print(
            " > Selected {:d} validation exemplars, time={:5.1f}s".format(
                sum([len(elem) for elem in self.y_valid_exemplars]), clock1 - clock0
            )
        )

        # make copy to keep the type of dataset for Stage 2 -- not efficient
        bic_val_dataset = deepcopy(trn_loader.dataset)
        # add exemplars to train_loader -- train_new + train_old (Fig.2)
        if t > 0:
            trn_loader = torch.utils.data.DataLoader(
                trn_loader.dataset + self.exemplars_dataset,
                batch_size=trn_loader.batch_size,
                shuffle=True,
                num_workers=trn_loader.num_workers,
                pin_memory=trn_loader.pin_memory,
            )

        # STAGE 1: DISTILLATION
        print("Stage 1: Training model with distillation")
        super().train_loop(t, trn_loader, val_loader)
        # From LwF: Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

        # STAGE 2: BIAS CORRECTION
        if t > 0:
            print("Stage 2: Training bias correction layers")
            # Fill bic_val_loader with validation protoset
            if isinstance(bic_val_dataset.images, list):
                bic_val_dataset.images = sum(self.x_valid_exemplars, [])
            else:
                bic_val_dataset.images = np.vstack(self.x_valid_exemplars)
            bic_val_dataset.labels = sum(self.y_valid_exemplars, [])
            bic_val_loader = DataLoader(
                bic_val_dataset,
                batch_size=trn_loader.batch_size,
                shuffle=True,
                num_workers=trn_loader.num_workers,
                pin_memory=trn_loader.pin_memory,
            )

            # bias optimization on validation
            self.model.eval()
            # Allow to learn the alpha and beta for the current task
            if self.model.is_early_exit():
                for ic_biases in self.bias_layers:
                    ic_biases[t].alpha.requires_grad = True
                    ic_biases[t].beta.requires_grad = True

                params = []
                for ic_biases in self.bias_layers:
                    params += list(ic_biases[t].parameters())
            else:
                self.bias_layers[t].alpha.requires_grad = True
                self.bias_layers[t].beta.requires_grad = True

                params = list(self.bias_layers[t].parameters())

            # In their code is specified that momentum is always 0.9
            bic_optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.9)
            # Loop epochs
            for e in range(self.bias_epochs):
                # Train bias correction layers
                clock0 = time.time()

                if self.model.is_early_exit():
                    total_loss, total_acc = np.zeros(len(self.bias_layers)), np.zeros(
                        len(self.bias_layers)
                    )
                else:
                    total_loss, total_acc = 0, 0

                for inputs, targets in bic_val_loader:
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

                    # Forward current model
                    with torch.no_grad():
                        outputs = self.model(inputs)

                    if self.model.is_early_exit():
                        with torch.no_grad():
                            old_cls_outs = self.bias_forward([o[:t] for o in outputs])
                        new_cls_outs = [
                            self.bias_layers[ic_idx][t](outputs[ic_idx][t])
                            for ic_idx in range(len(outputs))
                        ]
                        pred_all_classes = [
                            torch.cat(
                                [
                                    torch.cat(old_cls_outs[ic_idx], dim=1),
                                    new_cls_outs[ic_idx],
                                ],
                                dim=1,
                            )
                            for ic_idx in range(len(outputs))
                        ]
                        loss = 0
                        for ic_idx in range(len(outputs)):
                            ic_loss = torch.nn.functional.cross_entropy(
                                pred_all_classes[ic_idx], targets
                            )
                            ic_loss += 0.1 * (
                                (self.bias_layers[ic_idx][t].beta[0] ** 2) / 2
                            )
                            total_loss[ic_idx] += ic_loss.item() * len(targets)
                            total_acc[ic_idx] += (
                                (
                                    (
                                        pred_all_classes[ic_idx].argmax(1) == targets
                                    ).float()
                                )
                                .sum()
                                .item()
                            )
                            loss += ic_loss
                    else:
                        with torch.no_grad():
                            old_cls_outs = self.bias_forward(outputs[:t])
                        new_cls_outs = self.bias_layers[t](outputs[t])
                        pred_all_classes = torch.cat(
                            [torch.cat(old_cls_outs, dim=1), new_cls_outs], dim=1
                        )
                        # Eqs. 4-5: outputs from previous tasks are not modified (any alpha or beta from those is fixed),
                        #           only alpha and beta from the new task is learned. No temperature scaling used.
                        loss = torch.nn.functional.cross_entropy(
                            pred_all_classes, targets
                        )
                        # However, in their code, they apply a 0.1 * L2-loss to the gamma variable (beta in the paper)
                        loss += 0.1 * ((self.bias_layers[t].beta[0] ** 2) / 2)
                        # Log
                        total_loss += loss.item() * len(targets)
                        total_acc += (
                            ((pred_all_classes.argmax(1) == targets).float())
                            .sum()
                            .item()
                        )
                    # Backward
                    bic_optimizer.zero_grad()
                    loss.backward()
                    bic_optimizer.step()
                clock1 = time.time()
                # reducing the amount of verbose
                if (e + 1) % (self.bias_epochs / 4) == 0:
                    if self.model.is_early_exit():
                        loss_placeholder = (
                            total_loss / len(bic_val_loader.dataset.labels)
                        ).tolist()
                        acc_placeholder = (
                            100 * total_acc / len(bic_val_loader.dataset.labels)
                        ).tolist()
                        loss_placeholder = [round(l, 4) for l in loss_placeholder]
                        acc_placeholder = [round(a, 4) for a in acc_placeholder]
                    else:
                        loss_placeholder = total_loss / len(
                            bic_val_loader.dataset.labels
                        )
                        acc_placeholder = (
                            100 * total_acc / len(bic_val_loader.dataset.labels)
                        )
                        loss_placeholder = round(loss_placeholder, 4)
                        acc_placeholder = round(acc_placeholder, 4)

                    print(
                        "| Epoch {:3d}, time={:5.1f}s | Train: loss={}, TAg acc={}% |".format(
                            e + 1,
                            clock1 - clock0,
                            loss_placeholder,
                            acc_placeholder,
                        )
                    )
            # Fix alpha and beta after learning them
            if self.model.is_early_exit():
                for ic_biases in self.bias_layers:
                    ic_biases[t].alpha.requires_grad = False
                    ic_biases[t].beta.requires_grad = False
            else:
                self.bias_layers[t].alpha.requires_grad = False
                self.bias_layers[t].beta.requires_grad = False

        # Print all alpha and beta values
        # for task in range(t + 1):
        #     print(
        #         "Stage 2: BiC training for Task {:d}: alpha={:.5f}, beta={:.5f}".format(
        #             task,
        #             self.bias_layers[task].alpha.item(),
        #             self.bias_layers[task].beta.item(),
        #         )
        #     )

        # STAGE 3: EXEMPLAR MANAGEMENT
        self.exemplars_dataset.collect_exemplars(
            self.model, trn_loader, val_loader.dataset.transform
        )

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            # Forward old model
            targets_old = None
            if t > 0:
                targets_old = self.model_old(images.to(self.device, non_blocking=True))
                targets_old = self.bias_forward(targets_old)  # apply bias correction
            # Forward current model
            outputs = self.model(images.to(self.device, non_blocking=True))
            outputs = self.bias_forward(outputs)  # apply bias correction
            loss = self.criterion(
                t, outputs, targets.to(self.device, non_blocking=True), targets_old
            )

            if self.model.is_early_exit():
                for ic_idx, ic_loss in enumerate(loss):
                    assert not torch.isnan(ic_loss), f"Loss is NaN at {ic_idx}"
                loss = sum(loss)
            else:
                assert not torch.isnan(loss), "Loss is NaN"

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

    def eval(self, t, val_loader, features_save_dir=None):
        """Contains the evaluation code"""
        with torch.no_grad():
            if self.model.is_early_exit():
                total_loss, total_acc_taw, total_acc_tag, total_num = (
                    np.zeros((len(self.model.ic_layers) + 1,)),
                    np.zeros((len(self.model.ic_layers) + 1,)),
                    np.zeros((len(self.model.ic_layers) + 1,)),
                    np.zeros((len(self.model.ic_layers) + 1,)),
                )
            else:
                total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0

            self.model.eval()
            for images, targets in val_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                # Forward old model
                targets_old = None
                if t > 0:
                    targets_old = self.model_old(images)
                    targets_old = self.bias_forward(
                        targets_old
                    )  # apply bias correction
                # Forward current model
                outputs = self.model(images)
                outputs = self.bias_forward(outputs)  # apply bias correction
                loss = self.criterion(
                    t, outputs, targets.to(self.device, non_blocking=True), targets_old
                )

                if self.model.is_early_exit():
                    for i, ic_outputs in enumerate(outputs):
                        hits_taw, hits_tag = self.calculate_metrics(ic_outputs, targets)
                        # Log
                        total_loss += loss[i].item() * len(targets)
                        total_acc_taw[i] += hits_taw.sum().item()
                        total_acc_tag[i] += hits_tag.sum().item()
                        total_num += len(targets)
                else:
                    hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                    # Log
                    total_loss += loss.item() * len(targets)
                    total_acc_taw += hits_taw.sum().item()
                    total_acc_tag += hits_tag.sum().item()
                    total_num += len(targets)

            if self.model.is_early_exit():
                total_num //= len(self.model.ic_layers) + 1

        return (
            total_loss / total_num,
            total_acc_taw / total_num,
            total_acc_tag / total_num,
        )

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce

    def criterion(self, t, outputs, targets, targets_old):
        """Returns the loss value"""
        if self.model.is_early_exit():
            ic_weights = self.model.get_ic_weights(
                current_epoch=self.current_epoch, max_epochs=self.nepochs
            )
            losses = []

            for ic_idx in range(len(outputs)):
                loss_dist = 0
                if t > 0:
                    loss_dist += self.cross_entropy(
                        torch.cat(outputs[ic_idx][:t], dim=1),
                        torch.cat(targets_old[ic_idx][:t], dim=1),
                        exp=1.0 / self.T,
                    )
                # trade-off - the lambda from the paper if lamb=-1
                if self.lamb == -1:
                    lamb = (
                        self.model.task_cls[:t].sum().float()
                        / self.model.task_cls.sum()
                    ).to(self.device, non_blocking=True)
                    loss = (1.0 - lamb) * torch.nn.functional.cross_entropy(
                        torch.cat(outputs[ic_idx], dim=1), targets
                    ) + lamb * loss_dist
                else:
                    if self.lamb_warmup is not None:
                        lamb = (
                            min(1, (self.current_epoch + 1) / self.lamb_warmup)
                            * self.lamb
                        )
                    else:
                        lamb = self.lamb
                    loss = (
                        torch.nn.functional.cross_entropy(
                            torch.cat(outputs[ic_idx], dim=1), targets
                        )
                        + lamb * loss_dist
                    )
                losses.append(loss * ic_weights[ic_idx])
            return losses

        else:
            # Knowledge distillation loss for all previous tasks
            loss_dist = 0
            if t > 0:
                loss_dist += self.cross_entropy(
                    torch.cat(outputs[:t], dim=1),
                    torch.cat(targets_old[:t], dim=1),
                    exp=1.0 / self.T,
                )
            # trade-off - the lambda from the paper if lamb=-1
            if self.lamb == -1:
                lamb = (
                    self.model.task_cls[:t].sum().float() / self.model.task_cls.sum()
                ).to(self.device, non_blocking=True)
                return (1.0 - lamb) * torch.nn.functional.cross_entropy(
                    torch.cat(outputs, dim=1), targets
                ) + lamb * loss_dist
            else:
                if self.lamb_warmup is not None:
                    lamb = (
                        min(1, (self.current_epoch + 1) / self.lamb_warmup) * self.lamb
                    )
                else:
                    lamb = self.lamb
                return (
                    torch.nn.functional.cross_entropy(
                        torch.cat(outputs, dim=1), targets
                    )
                    + lamb * loss_dist
                )

    def ee_net(self, exit_layer: Optional[int] = None) -> torch.nn.Module:
        # early exit network with all CL logic implemented for inference, for profiling
        tmp_model = deepcopy(self.model)
        tmp_model.set_exit_layer(exit_layer)
        bic_model = BiCModelWrapper(tmp_model, self.bias_layers)
        return bic_model


class BiCModelWrapper(torch.nn.Module):
    def __init__(self, model, bias_layers):
        super().__init__()
        self.model = model
        self.bias_layers = bias_layers

    def forward(self, x):
        outputs = self.model(x)
        if self.model.exit_layer_idx == -1:
            # This is the case when model does not use early exits
            final_bic_layers = self.bias_layers[-1]
            return [
                bias_layer(output)
                for bias_layer, output in zip(final_bic_layers, outputs)
            ]
        else:
            # This is the case when model uses early exits sequentially
            bic_outputs = []
            for ic_idx, ic_output in enumerate(outputs):
                ic_bic_outputs = []
                for output, bias_layer in zip(ic_output, self.bias_layers[ic_idx]):
                    ic_bic_outputs.append(bias_layer(output))
                bic_outputs.append(ic_bic_outputs)
            return bic_outputs


class BiasLayer(torch.nn.Module):
    """Bias layers with alpha and beta parameters"""

    def __init__(self):
        super(BiasLayer, self).__init__()
        # Initialize alpha and beta with requires_grad=False and only set to True during Stage 2
        self.alpha = torch.nn.Parameter(torch.ones(1, requires_grad=False))
        self.beta = torch.nn.Parameter(torch.zeros(1, requires_grad=False))

    def forward(self, x):
        return self.alpha * x + self.beta
