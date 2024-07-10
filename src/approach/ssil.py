import warnings
from argparse import ArgumentParser
from copy import deepcopy

import torch
import torch.nn.functional as F

from datasets.exemplars_dataset import ExemplarsDataset

from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    """Class implementing the SS-IL : Separated Softmax for Incremental Learning approach
    described in:
    https://openaccess.thecvf.com/content/ICCV2021/papers/Ahn_SS-IL_Separated_Softmax_for_Incremental_Learning_ICCV_2021_paper.pdf

    Code: https://github.com/hongjoon0805/SS-IL-Official/blob/master/trainer/ssil.py
    """

    def __init__(
        self,
        model,
        device,
        nepochs=100,
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
        lamb=1,
        T=2,
        replay_batch_size=32,
    ):
        super(Appr, self).__init__(
            model,
            device,
            nepochs,
            lr,
            lr_min,
            lr_factor,
            lr_patience,
            clipgrad,
            momentum,
            wd,
            multi_softmax,
            fix_bn,
            eval_on_train,
            select_best_model_by_val_loss,
            logger,
            exemplars_dataset,
            scheduler_milestones,
        )
        self.model_old = None
        self.lamb = lamb
        self.T = T
        self.replay_batch_size = replay_batch_size

        self.loss = torch.nn.CrossEntropyLoss(reduction="sum")

        # SSIL is expected to be used with exemplars. If needed to be used without exemplars, overwrite here the
        # `_get_optimizer` function with the one in LwF and update the criterion
        have_exemplars = (
            self.exemplars_dataset.max_num_exemplars
            + self.exemplars_dataset.max_num_exemplars_per_class
        )
        if not have_exemplars:
            warnings.warn(
                "Warning: SS-IL is expected to use exemplars. Check documentation."
            )

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument(
            "--lamb",
            default=1,
            type=float,
            required=False,
            help="Forgetting-intransigence trade-off (default=%(default)s)",
        )
        parser.add_argument(
            "--T",
            default=2,
            type=int,
            required=False,
            help="Temperature scaling (default=%(default)s)",
        )
        parser.add_argument(
            "--replay-batch-size",
            default=32,
            type=int,
            required=False,
            help="Replay batch size (default=%(default)s)",
        )

        return parser.parse_known_args(args)

    def _get_optimizer(self):
        params = self.model.parameters()
        return torch.optim.SGD(
            params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum
        )

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        exemplar_selection_loader = torch.utils.data.DataLoader(
            trn_loader.dataset + self.exemplars_dataset,
            batch_size=trn_loader.batch_size,
            shuffle=True,
            num_workers=trn_loader.num_workers,
            pin_memory=trn_loader.pin_memory,
        )
        self.exemplars_dataset.collect_exemplars(
            self.model, exemplar_selection_loader, val_loader.dataset.transform
        )

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        # Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def train_epoch(self, t, trn_loader):
        if t > 0:
            exemplar_loader = torch.utils.data.DataLoader(
                self.exemplars_dataset,
                batch_size=self.replay_batch_size,
                shuffle=True,
                num_workers=trn_loader.num_workers,
                pin_memory=trn_loader.pin_memory,
                drop_last=True,
            )
            exemplar_iterator = iter(exemplar_loader)

        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        for samples in trn_loader:
            if t > 0:
                data, target = samples
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                try:
                    data_r, target_r = next(exemplar_iterator)
                except StopIteration:
                    exemplar_iterator = iter(exemplar_loader)
                    data_r, target_r = next(exemplar_iterator)
                data_r = data_r.to(self.device, non_blocking=True)
                target_r = target_r.to(self.device, non_blocking=True)

                # Forward old model
                data = torch.cat((data, data_r))
                targets_old = self.model_old(data)
            else:
                data, target = samples
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                target_r = None
                targets_old = None
            # Forward current model
            outputs = self.model(data)
            loss = self.criterion(t, outputs, target, target_r, targets_old)
            if self.model.is_early_exit():
                loss = sum(loss)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def criterion(self, t, outputs, target, target_r=None, outputs_old=None):
        batch_size = len(target)
        replay_size = len(target_r) if target_r is not None else 0

        if self.model.is_early_exit():
            ic_weights = self.model.get_ic_weights(
                current_epoch=self.current_epoch, max_epochs=self.nepochs
            )
            losses_kd = []
            losses_ce = []
            losses_total = []

            for ic_idx in range(len(outputs)):
                outputs_ic = outputs[ic_idx]

                ic_loss_KD = 0
                ic_loss_CE_curr = self.loss(
                    outputs_ic[t][:batch_size], target - self.model.task_offset[t]
                )
                if t > 0 and target_r is not None:
                    outputs_old_ic = outputs_old[ic_idx]

                    ic_prev = torch.cat(
                        [
                            o[batch_size : batch_size + replay_size]
                            for o in outputs_ic[:t]
                        ],
                        dim=1,
                    )
                    ic_loss_CE_prev = self.loss(ic_prev, target_r)
                    ic_loss_CE = (ic_loss_CE_curr + ic_loss_CE_prev) / (
                        batch_size + replay_size
                    )

                    # KD loss
                    ic_loss_KD = torch.zeros(t).to(self.device, non_blocking=True)
                    for _t in range(t):
                        soft_target = F.softmax(outputs_old_ic[_t] / self.T, dim=1)
                        output_log = F.log_softmax(outputs_ic[_t] / self.T, dim=1)
                        ic_loss_KD[_t] = F.kl_div(
                            output_log, soft_target, reduction="batchmean"
                        ) * (self.T**2)
                    ic_loss_KD = ic_loss_KD.sum()
                else:
                    ic_loss_CE = ic_loss_CE_curr / batch_size

                ic_loss_CE = ic_loss_CE * ic_weights[ic_idx]
                ic_loss_KD = ic_loss_KD * ic_weights[ic_idx]

                losses_kd.append(ic_loss_KD)
                losses_ce.append(ic_loss_CE)
                losses_total.append(ic_loss_CE + self.lamb * ic_loss_KD)
            return losses_total
        else:
            loss_KD = 0
            loss_CE_curr = self.loss(
                outputs[t][:batch_size], target - self.model.task_offset[t]
            )

            if t > 0 and target_r is not None:
                prev = torch.cat(
                    [o[batch_size : batch_size + replay_size] for o in outputs[:t]],
                    dim=1,
                )
                loss_CE_prev = self.loss(prev, target_r)
                loss_CE = (loss_CE_curr + loss_CE_prev) / (batch_size + replay_size)

                # loss_KD
                loss_KD = torch.zeros(t).to(self.device, non_blocking=True)
                for _t in range(t):
                    soft_target = F.softmax(outputs_old[_t] / self.T, dim=1)
                    output_log = F.log_softmax(outputs[_t] / self.T, dim=1)
                    loss_KD[_t] = F.kl_div(
                        output_log, soft_target, reduction="batchmean"
                    ) * (self.T**2)
                loss_KD = loss_KD.sum()
            else:
                loss_CE = loss_CE_curr / batch_size

            return loss_CE + self.lamb * loss_KD
