from argparse import ArgumentParser

import torch

from datasets.exemplars_dataset import ExemplarsDataset

from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    """Class implementing the finetuning baseline"""

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
        self.all_out = all_outputs
        self.memory_loader = None

        have_exemplars = (
            self.exemplars_dataset.max_num_exemplars
            + self.exemplars_dataset.max_num_exemplars_per_class
        )
        assert (
            have_exemplars
        ), "Warning: ER is expected to use exemplars. Check documentation."

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
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
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
            exemplars_batch_size = org_batch_size // 2
            while exemplars_batch_size > len(self.exemplars_dataset):
                exemplars_batch_size = exemplars_batch_size // 2
            self.memory_loader = torch.utils.data.DataLoader(
                self.exemplars_dataset,
                batch_size=exemplars_batch_size,
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
        self.exemplars_dataset.collect_exemplars(
            self.model, tmp_loader, val_loader.dataset.transform
        )

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        if t > 0:
            memory_iterator = iter(self.memory_loader)

        for images, targets in trn_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            if t > 0:
                try:
                    replay_images, replay_targets = next(memory_iterator)
                except StopIteration:
                    memory_iterator = iter(self.memory_loader)
                    replay_images, replay_targets = next(memory_iterator)
                replay_images = replay_images.to(self.device, non_blocking=True)
                replay_targets = replay_targets.to(self.device, non_blocking=True)
                images = torch.cat((images, replay_images), dim=0)
                targets = torch.cat((targets, replay_targets), dim=0)

            # Forward current model
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
            assert len(loss) == len(self.model.ic_layers) + 1
        else:
            """Returns the loss value"""
            loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return loss
