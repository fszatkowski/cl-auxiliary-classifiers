from argparse import ArgumentParser
from copy import deepcopy

import numpy as np
import torch

from datasets.exemplars_dataset import ExemplarsDataset

from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    """Class implementing A-LwF approach from
    Achieving a Better Stability-Plasticity Trade-off via Auxiliary Networks in Continual Learning
    (CVPR 2023)"""

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
        scheduler_name="multistep",
        scheduler_milestones=None,
        lamb=1,
        lamb_a=1,
        T=2,
        taskwise_kd=False,
        debug_loss=False,
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
        self.model_old = None
        self.model_aux = None
        self.lamb = lamb
        self.lamb_a = lamb_a
        self.T = T
        self.taskwise_kd = taskwise_kd

        self.debug_loss = debug_loss

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Page 5: "lambda is a loss balance weight, set to 1 for most our experiments. Making lambda larger will favor
        # the old task performance over the new task’s, so we can obtain a old-task-new-task performance line by
        # changing lambda."
        parser.add_argument(
            "--lamb",
            default=10,
            type=float,
            required=False,
            help="Forgetting-intransigence trade-off (default=%(default)s)",
        )
        parser.add_argument(
            "--lamb-a",
            default=1,
            type=float,
            required=False,
            help="Forgetting-intransigence trade-off (default=%(default)s)",
        )
        # Page 5: "We use T=2 according to a grid search on a held out set, which aligns with the authors’
        #  recommendations." -- Using a higher value for T produces a softer probability distribution over classes.
        parser.add_argument(
            "--T",
            default=2,
            type=int,
            required=False,
            help="Temperature scaling (default=%(default)s)",
        )
        parser.add_argument(
            "--taskwise-kd",
            default=False,
            action="store_true",
            required=False,
            help="If set, will use task-wise KD loss as defined in SSIL. (default=%(default)s)",
        )

        parser.add_argument(
            "--debug-loss",
            default=False,
            action="store_true",
            required=False,
            help="If set, will log partial losses during training. (default=%(default)s)",
        )
        return parser.parse_known_args(args)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        if t > 0:
            print("=" * 108)
            print("Training of Auxiliary Network")
            print("=" * 108)
            # Args for the new trainer
            new_trainer_args = dict(
                nepochs=self.nepochs,
                optimizer_name=self.optimizer_name,
                lr=self.lr,
                lr_min=self.lr_min,
                lr_factor=self.lr_factor,
                lr_patience=self.lr_patience,
                clipgrad=self.clipgrad,
                momentum=0.9,
                wd=5e-4,
                multi_softmax=self.multi_softmax,
                eval_on_train=self.eval_on_train,
                select_best_model_by_val_loss=self.select_best_model_by_val_loss,
                logger=self.logger,
                exemplars_dataset=self.exemplars_dataset,
                scheduler_name=self.scheduler_name,
                scheduler_milestones=self.scheduler_milestones,
            )
            self.model_aux = deepcopy(self.model)
            # Train auxiliary model on current dataset
            new_trainer = NewTaskTrainer(
                self.model_aux, self.device, **new_trainer_args
            )
            # New trained always trains on current dataset without exemplars
            new_trainer.train_loop(t, trn_loader, val_loader)
            self.model_aux.eval()
            self.model_aux.freeze_all()

            # add exemplars to train_loader
            if len(self.exemplars_dataset) > 0:
                trn_loader = torch.utils.data.DataLoader(
                    trn_loader.dataset + self.exemplars_dataset,
                    batch_size=trn_loader.batch_size,
                    shuffle=True,
                    num_workers=trn_loader.num_workers,
                    pin_memory=trn_loader.pin_memory,
                )

        print("=" * 108)
        print("Training of Main Network")
        print("=" * 108)
        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(
            self.model, trn_loader, val_loader.dataset.transform
        )

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        # Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Forward old model and auxiliary model
            targets_old = None
            targets_aux = None
            if t > 0:
                targets_old = self.model_old(images)
                targets_aux = self.model_aux(images)
            # Forward current model
            outputs = self.model(images)

            if self.debug_loss:
                loss_ce, loss_kd, loss_kd_a, loss = self.criterion(
                    t,
                    outputs,
                    targets,
                    targets_old,
                    targets_aux,
                    return_partial_losses=True,
                )
                self._log_partial_losses(t, loss_ce, loss_kd, loss_kd_a, loss)
            else:
                loss = self.criterion(
                    t,
                    outputs,
                    targets,
                    targets_old,
                    targets_aux,
                    return_partial_losses=False,
                )

            if self.model.is_early_exit():
                loss = sum(loss)
            assert not torch.isnan(loss), "Loss is NaN"

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

    def eval(
        self, t, val_loader, save_logits=False, save_features=False, save_dir=None
    ):
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
            if self.model_old is not None:
                self.model_old.eval()
            if self.model_aux is not None:
                self.model_aux.eval()

            for batch_idx, (images, targets) in enumerate(val_loader):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # Forward old model and auxiliary model
                targets_old = None
                targets_aux = None
                if t > 0:
                    targets_old = self.model_old(
                        images.to(self.device, non_blocking=True)
                    )
                    targets_aux = self.model_aux(
                        images.to(self.device, non_blocking=True)
                    )
                # Forward current model
                outputs, features = self.model(images, return_features=True)

                if save_dir is not None:
                    task_save_dir = save_dir / f"t_{t}"
                    task_save_dir.mkdir(parents=True, exist_ok=True)

                    save_dict = {"targets": targets}
                    if save_logits:
                        save_dict["logits"] = outputs
                    if save_features:
                        save_dict["features"] = features
                    torch.save(save_dict, task_save_dir / f"{batch_idx}.pt")

                loss = self.criterion(
                    t,
                    outputs,
                    targets.to(self.device, non_blocking=True),
                    targets_old,
                    targets_aux,
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

    def joint_kd_loss(self, t, outputs, outputs_old):
        kd_outputs, kd_outputs_old = torch.cat(outputs[:t], dim=1), torch.cat(
            outputs_old[:t], dim=1
        )
        return self.cross_entropy(kd_outputs, kd_outputs_old, exp=1.0 / self.T)

    def taskwise_kd_loss(self, t, outputs, outputs_old):
        loss_kd = torch.zeros(t).to(self.device, non_blocking=True)
        for _t in range(t):
            soft_target = torch.nn.functional.softmax(outputs_old[_t] / self.T, dim=1)
            output_log = torch.nn.functional.log_softmax(outputs[_t] / self.T, dim=1)
            loss_kd[_t] = torch.nn.functional.kl_div(
                output_log, soft_target, reduction="batchmean"
            ) * (self.T**2)
        loss_kd = loss_kd.sum()
        return loss_kd

    def kd_loss(self, t, outputs, outputs_old):
        if self.taskwise_kd:
            return self.taskwise_kd_loss(t, outputs, outputs_old)
        else:
            return self.joint_kd_loss(t, outputs, outputs_old)

    def criterion(
        self,
        t,
        outputs,
        targets,
        targets_old=None,
        targets_aux=None,
        return_partial_losses=False,
    ):
        """Returns the loss value"""
        if self.model.is_early_exit():
            ic_weights = self.model.get_ic_weights(
                current_epoch=self.current_epoch, max_epochs=self.nepochs
            )

            losses_ce = []
            losses_kd = []
            losses_kd_a = []
            losses_total = []

            for i in range(len(outputs)):
                ic_outputs = outputs[i]

                # Cross entropy
                if len(self.exemplars_dataset) > 0:
                    loss_ce = torch.nn.functional.cross_entropy(
                        torch.cat(ic_outputs, dim=1), targets
                    )
                else:
                    loss_ce = torch.nn.functional.cross_entropy(
                        ic_outputs[t], targets - self.model.task_offset[t]
                    )

                # Standard distillation
                if t > 0 and targets_old is not None:
                    loss_kd = self.kd_loss(t, ic_outputs, targets_old[i])
                else:
                    loss_kd = 0

                # Aux net distillation
                if t > 0 and targets_aux is not None:
                    loss_kd_a = self.cross_entropy(
                        ic_outputs[t],
                        targets_aux[i][t] - self.model.task_offset[t],
                        exp=1.0 / self.T,
                    )
                else:
                    loss_kd_a = 0

                loss_ce = loss_ce * ic_weights[i]
                loss_kd = loss_kd * ic_weights[i]
                loss_kd_a = loss_kd_a * ic_weights[i]

                losses_ce.append(loss_ce)
                losses_kd.append(loss_kd)
                losses_kd_a.append(loss_kd_a)
                losses_total.append(
                    loss_ce + self.lamb * loss_kd + self.lamb_a * loss_kd_a
                )

            if return_partial_losses:
                return losses_ce, losses_kd, losses_kd_a, losses_total
            else:
                return losses_total
        else:
            # Current cross-entropy loss -- with exemplars use all heads
            if len(self.exemplars_dataset) > 0:
                loss_ce = torch.nn.functional.cross_entropy(
                    torch.cat(outputs, dim=1), targets
                )
            else:
                loss_ce = torch.nn.functional.cross_entropy(
                    outputs[t], targets - self.model.task_offset[t]
                )

            if t > 0:
                # Knowledge distillation loss for all previous tasks on old(previous) network
                loss_kd = self.kd_loss(t, outputs, targets_old)

                # Auxiliary KD
                # Knowledge distillation loss for current task on new network
                loss_kd_a = self.cross_entropy(
                    outputs[t],
                    targets_aux[t] - self.model.task_offset[t],
                    exp=1.0 / self.T,
                )
            else:
                loss_kd, loss_kd_a = 0, 0

            if return_partial_losses:
                return (
                    loss_ce,
                    loss_kd,
                    loss_kd_a,
                    loss_ce + self.lamb * loss_kd + self.lamb_a * loss_kd_a,
                )
            else:
                return loss_ce + self.lamb * loss_kd + self.lamb_a * loss_kd_a

    def _log_partial_losses(self, t, loss_ce, loss_kd, loss_kd_a, loss):
        if self.model.is_early_exit():
            for ic_idx, (loss_kd_, loss_kd_a_, loss_ce_, loss_) in enumerate(
                zip(loss_kd[:-1], loss_kd_a[:-1], loss_ce[:-1], loss[:-1])
            ):
                self.logger.log_scalar(
                    task=None,
                    iter=None,
                    name="loss_kd",
                    group=f"debug_ic_{ic_idx}_t{t}",
                    value=float(loss_kd_[-1]),
                )
                self.logger.log_scalar(
                    task=None,
                    iter=None,
                    name="loss_kd_a",
                    group=f"debug_ic_{ic_idx}_t{t}",
                    value=float(loss_kd_a[-1]),
                )
                self.logger.log_scalar(
                    task=None,
                    iter=None,
                    name="loss_ce",
                    group=f"debug_ic_{ic_idx}_t{t}",
                    value=float(loss_ce_[-1]),
                )
                self.logger.log_scalar(
                    task=None,
                    iter=None,
                    name="loss_total",
                    group=f"debug_ic_{ic_idx}_t{t}",
                    value=float(loss_[-1]),
                )

            self.logger.log_scalar(
                task=None,
                iter=None,
                name="loss_kd",
                group=f"debug_t{t}",
                value=float(loss_kd[-1]),
            )
            self.logger.log_scalar(
                task=None,
                iter=None,
                name="loss_kd_a",
                group=f"debug_t{t}",
                value=float(loss_kd_a[-1]),
            )
            self.logger.log_scalar(
                task=None,
                iter=None,
                name="loss_ce",
                group=f"debug_t{t}",
                value=float(loss_ce[-1]),
            )
            self.logger.log_scalar(
                task=None,
                iter=None,
                name="loss_total",
                group=f"debug_t{t}",
                value=float(loss[-1]),
            )
        else:
            self.logger.log_scalar(
                task=None,
                iter=None,
                name="loss_kd",
                group=f"debug_t{t}",
                value=float(loss_kd),
            )
            self.logger.log_scalar(
                task=None,
                iter=None,
                name="loss_kd_a",
                group=f"debug_t{t}",
                value=float(loss_kd_a),
            )
            self.logger.log_scalar(
                task=None,
                iter=None,
                name="loss_ce",
                group=f"debug_t{t}",
                value=float(loss_ce),
            )
            self.logger.log_scalar(
                task=None,
                iter=None,
                name="loss_total",
                group=f"debug_t{t}",
                value=float(loss),
            )


class NewTaskTrainer(Inc_Learning_Appr):
    def __init__(
        self,
        model,
        device,
        nepochs=160,
        optimizer_name="sgd",
        lr=0.05,
        lr_min=1e-4,
        lr_factor=3,
        lr_patience=5,
        clipgrad=10000,
        momentum=0.9,
        wd=5e-4,
        multi_softmax=False,
        fix_bn=False,
        eval_on_train=False,
        select_best_model_by_val_loss=True,
        logger=None,
        exemplars_dataset=None,
        scheduler_name="multistep",
        scheduler_milestones=None,
    ):
        super(NewTaskTrainer, self).__init__(
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
