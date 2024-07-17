from argparse import ArgumentParser

import torch
import torch.nn.functional as F

from datasets.exemplars_dataset import ExemplarsDataset

from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    """
    Loss Decoupling for Task-Agnostic Continual Learning
    This code implements ER-based version of LODE
    NeurIPS2023 - https://openreview.net/pdf?id=9Oi3YxIBSa
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
            all_outputs=False,
            const=1.0,
            ro=1.0,
    ):
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
        self.memory_loader = None
        self.all_out = all_outputs

        self.c = const
        self.ro = ro

        have_exemplars = (
                self.exemplars_dataset.max_num_exemplars
                + self.exemplars_dataset.max_num_exemplars_per_class
        )
        assert (
            have_exemplars
        ), "Warning: LODE is expected to use exemplars. Check documentation."

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument(
            "--all-outputs",
            action="store_true",
            required=False,
            help="Allow all weights related to all outputs to be modified (default=%(default)s)",
        )
        parser.add_argument(
            "--const",
            type=float,
            default=1.0,
            help="C hyperparamter from Eq. 5 in the paper (default=%(default)s)",
        )
        parser.add_argument(
            "--ro",
            type=float,
            default=0.1,
            help="ro hyperparamter from Eq. 5 in the paper (default=%(default)s)",
        )
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if len(self.exemplars_dataset) == 0 and not self.all_out:
            # No exemplars case
            base_params = list(self.model.model.parameters())
            if self.model.is_early_exit():
                head_params = [
                    p
                    for ic_heads in self.model.heads
                    for p in ic_heads[-1].parameters()
                ]
            else:
                head_params = list(self.model.heads[-1].parameters())
            params = base_params + head_params
        else:
            params = list(self.model.parameters())

        if self.optimizer_name == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.lr,
                weight_decay=self.wd,
                momentum=self.momentum,
            )
        elif self.optimizer_name == "adamw":
            return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.wd)
        else:
            raise NotImplementedError(f"Unknown optimizer: {self.optimizer_name}")

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add exemplars to train_loader; this is necessary to correctly update exemplars after each task
        if len(self.exemplars_dataset) > 0 and t > 0:
            tmp_loader = torch.utils.data.DataLoader(
                trn_loader.dataset + self.exemplars_dataset,
                batch_size=trn_loader.batch_size,
                shuffle=True,
                num_workers=trn_loader.num_workers,
                pin_memory=trn_loader.pin_memory,
            )

            org_batch_size = trn_loader.batch_size
            # Lower the batch size for replay
            # TODO check if it is correct to make the batch size half
            trn_loader = torch.utils.data.DataLoader(
                trn_loader.dataset,
                batch_size=org_batch_size // 2,
                shuffle=True,
                num_workers=trn_loader.num_workers,
                pin_memory=trn_loader.pin_memory,
                drop_last=True,
            )
            self.memory_loader = torch.utils.data.DataLoader(
                self.exemplars_dataset,
                batch_size=org_batch_size // 2,
                shuffle=True,
                num_workers=trn_loader.num_workers,
                pin_memory=trn_loader.pin_memory,
                drop_last=True,
            )
        else:
            tmp_loader = trn_loader

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        # import pdb; pdb.set_trace()
        self.exemplars_dataset.collect_exemplars(
            self.model, tmp_loader, val_loader.dataset.transform
        )

    def train_epoch(self, t, trn_loader):
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        if t > 0:
            memory_iterator = iter(self.memory_loader)

        for new_images, new_targets in trn_loader:
            new_images = new_images.to(self.device, non_blocking=True)
            new_targets = new_targets.to(self.device, non_blocking=True)

            if t > 0:
                # Compute LODE loss for tasks after the first
                try:
                    replay_images, replay_targets = next(memory_iterator)
                except StopIteration:
                    memory_iterator = iter(self.memory_loader)
                    replay_images, replay_targets = next(memory_iterator)
                replay_images = replay_images.to(self.device, non_blocking=True)
                replay_targets = replay_targets.to(self.device, non_blocking=True)

                new_outputs = self.model(new_images)
                replay_outputs = self.model(replay_images)

                loss = self.criterion(
                    t, new_outputs, new_targets, replay_outputs, replay_targets
                )

            else:
                # Compute standard cross-entropy for the first task
                outputs = self.model(new_images)
                loss = self.criterion(t, outputs, new_targets)

            # Forward current model
            if self.model.is_early_exit():
                loss = sum(loss)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

    # Added for LODE: loss decoupling for task agnostic continual learning
    def logmeanexp_previous(self, x, classes1, classes2, dim=None):
        """Stable computation of log(mean(exp(x))"""
        if dim is None:
            x, dim = x.view(-1), 0
        x_max, _ = torch.max(x, dim)
        x_max = x_max.detach()
        old_pre = torch.logsumexp(x[:, classes1], dim=1)
        new_pre = torch.logsumexp(x[:, classes2], dim=1)
        pre = torch.stack([old_pre, new_pre], dim=-1)
        return pre

    def inter_cls(self, logits, y, classes1, classes2):
        logits = torch.cat(logits, dim=1)
        inter_logits = self.logmeanexp_previous(logits, classes1, classes2, dim=-1)
        inter_y = torch.ones_like(y)
        return F.cross_entropy(inter_logits, inter_y, reduction="none")

    def single_head_lode_loss(
            self,
            t,
            outputs_new,
            targets_new,
            outputs_old,
            targets_old,
            classes_new,
            classes_old,
    ):
        ce_new = F.cross_entropy(outputs_new[t], targets_new - self.model.task_offset[t])

        new_inter_cls = self.inter_cls(
            outputs_new,
            targets_new,
            classes_old,
            classes_new,
        )

        # This is ER version of LODE, so replay loss is simple cross entropy on old exemplars
        l_rep = F.cross_entropy(
            torch.cat(outputs_old, dim=1),
            targets_old,
        )
        # TODO does it match eq. 4 exactly?
        total_loss = self.c * ce_new + self.ro * (len(classes_new) / len(classes_old)) * new_inter_cls.mean() + l_rep
        return total_loss

    def criterion(self, t, outputs, targets, outputs_old=None, targets_old=None):
        if outputs_old is not None and targets_old is not None:
            outputs_new, targets_new = outputs, targets

            n_old_classes = self.model.task_offset[-1]
            if self.model.is_early_exit():
                n_new_classes = self.model.heads[-1][-1].out_features
            else:
                n_new_classes = self.model.heads[-1].out_features
            old_classes = torch.arange(n_old_classes)
            new_classes = torch.arange(n_old_classes, n_old_classes + n_new_classes)

            if self.model.is_early_exit():
                ic_weights = self.model.get_ic_weights(
                    current_epoch=self.current_epoch, max_epochs=self.nepochs
                )
                loss = []
                for ic_idx in range(len(ic_weights)):
                    ic_loss = self.single_head_lode_loss(
                        t,
                        outputs_new[ic_idx],
                        targets_new,
                        outputs_old[ic_idx],
                        targets_old,
                        new_classes,
                        old_classes,
                    )
                    loss.append(ic_weights[ic_idx] * ic_loss)
                return loss
            else:
                return self.single_head_lode_loss(
                    t,
                    outputs_new,
                    targets_new,
                    outputs_old,
                    targets_old,
                    new_classes,
                    old_classes,
                )
        else:
            return self._standard_cross_entropy_loss(t, outputs, targets)

    def _standard_cross_entropy_loss(self, t, outputs, targets):
        if self.model.is_early_exit():
            ic_weights = self.model.get_ic_weights(
                current_epoch=self.current_epoch, max_epochs=self.nepochs
            )
            loss = []
            for ic_outputs, ic_weight in zip(outputs, ic_weights):
                if self.all_out or len(self.exemplars_dataset) > 0:
                    loss.append(
                        ic_weight
                        * torch.nn.functional.cross_entropy(
                            torch.cat(ic_outputs, dim=1), targets
                        )
                    )
                else:
                    loss.append(
                        ic_weight
                        * torch.nn.functional.cross_entropy(
                            ic_outputs[t], targets - self.model.task_offset[t]
                        )
                    )
            assert len(loss) == len(self.model.ic_layers) + 1
        else:
            """Returns the loss value"""
            if self.all_out or len(self.exemplars_dataset) > 0:
                loss = torch.nn.functional.cross_entropy(
                    torch.cat(outputs, dim=1), targets
                )
            else:
                loss = torch.nn.functional.cross_entropy(
                    outputs[t], targets - self.model.task_offset[t]
                )
        return loss
