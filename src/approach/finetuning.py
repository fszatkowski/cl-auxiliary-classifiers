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

        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(
                trn_loader.dataset + self.exemplars_dataset,
                batch_size=trn_loader.batch_size,
                shuffle=True,
                num_workers=trn_loader.num_workers,
                pin_memory=trn_loader.pin_memory,
            )

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(
            self.model, trn_loader, val_loader.dataset.transform
        )

    def criterion(self, t, outputs, targets):
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
