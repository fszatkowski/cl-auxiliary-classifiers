from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader, Dataset

from datasets.exemplars_dataset import ExemplarsDataset

from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    """Class implementing the joint baseline"""

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
        freeze_after=-1,
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
        self.trn_datasets = []
        self.val_datasets = []
        self.freeze_after = freeze_after

        have_exemplars = (
            self.exemplars_dataset.max_num_exemplars
            + self.exemplars_dataset.max_num_exemplars_per_class
        )
        assert (
            have_exemplars == 0
        ), "Warning: Joint does not use exemplars. Comment this line to force it."

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument(
            "--freeze-after",
            default=-1,
            type=int,
            required=False,
            help="Freeze model except heads after the specified task"
            "(-1: normal Incremental Joint Training, no freeze) (default=%(default)s)",
        )
        return parser.parse_known_args(args)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        if self.freeze_after > -1 and t >= self.freeze_after:
            self.model.freeze_all()
            if self.model.is_early_exit():
                for cls_heads in self.model.heads:
                    for task_head in cls_heads:
                        for param in task_head.parameters():
                            param.requires_grad = True
            else:
                for head in self.model.heads:
                    for param in head.parameters():
                        param.requires_grad = True

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add new datasets to existing cumulative ones
        self.trn_datasets.append(trn_loader.dataset)
        self.val_datasets.append(val_loader.dataset)
        trn_dset = JointDataset(self.trn_datasets)
        val_dset = JointDataset(self.val_datasets)
        trn_loader = DataLoader(
            trn_dset,
            batch_size=trn_loader.batch_size,
            shuffle=True,
            num_workers=trn_loader.num_workers,
            pin_memory=trn_loader.pin_memory,
        )
        val_loader = DataLoader(
            val_dset,
            batch_size=val_loader.batch_size,
            shuffle=False,
            num_workers=val_loader.num_workers,
            pin_memory=val_loader.pin_memory,
        )
        # continue training as usual
        super().train_loop(t, trn_loader, val_loader)

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        if self.freeze_after < 0 or t <= self.freeze_after:
            self.model.train()
            if self.fix_bn and t > 0:
                self.model.freeze_bn()
        else:
            self.model.eval()
            for head in self.model.heads:
                head.train()
        for images, targets in trn_loader:
            # Forward current model
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            outputs = self.model(images)
            loss = self.criterion(t, outputs, targets)
            if self.model.is_early_exit():
                loss = sum(loss)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        if self.model.is_early_exit():
            ic_weights = self.model.get_ic_weights(
                current_epoch=self.current_epoch, max_epochs=self.nepochs
            )
            loss = []
            for ic_outputs, ic_weight in zip(outputs, ic_weights):
                loss.append(
                    ic_weight
                    * torch.nn.functional.cross_entropy(
                        torch.cat(ic_outputs, dim=1), targets
                    )
                )
            return loss
        else:
            return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)


class JointDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset accumulates each task dataset incrementally"""

    def __init__(self, datasets):
        self.datasets = datasets
        self._len = sum([len(d) for d in self.datasets])

    def __len__(self):
        "Denotes the total number of samples"
        return self._len

    def __getitem__(self, index):
        for d in self.datasets:
            if len(d) <= index:
                index -= len(d)
            else:
                x, y = d[index]
                return x, y
