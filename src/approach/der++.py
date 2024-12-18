import importlib
import time
from argparse import ArgumentParser

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.exemplars_dataset import ExemplarsDataset
from datasets.memory_dataset import MemoryDataset

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
        alpha=0.5,
        beta=0.5,
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
        self.der_dataset = DERExemplarsDatasetWrapper(device=self.device)
        self.memory_loader = None

        self.alpha = alpha
        self.beta = beta

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
        parser.add_argument(
            "--alpha",
            type=float,
            default=0.5,
            help="Alpha from DER++ algorithm (default=%(default)s)",
        )
        parser.add_argument(
            "--beta",
            type=float,
            default=0.5,
            help="Beta from DER++ algorithm (default=%(default)s)",
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
            trn_loader = torch.utils.data.DataLoader(
                trn_loader.dataset,
                batch_size=org_batch_size // 2,
                shuffle=True,
                num_workers=trn_loader.num_workers,
                pin_memory=trn_loader.pin_memory,
                drop_last=True,
            )

            new_outputs = self.model.task_cls[-1]
            self.der_dataset.mask_new_outputs(new_outputs)
            exemplars_batch_size = org_batch_size // 2
            while exemplars_batch_size > len(self.exemplars_dataset):
                exemplars_batch_size = exemplars_batch_size // 2
            self.memory_loader = torch.utils.data.DataLoader(
                self.der_dataset,
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
        self.der_dataset.update(
            self.model, self.exemplars_dataset, val_loader.dataset.transform
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
                    replay_images, _, _, replay_targets = next(memory_iterator)
                except StopIteration:
                    memory_iterator = iter(self.memory_loader)
                    replay_images, _, _, replay_targets = next(memory_iterator)

                try:
                    logit_replay_images, logit_replay_targets, logit_replay_masks, _ = (
                        next(memory_iterator)
                    )
                except StopIteration:
                    memory_iterator = iter(self.memory_loader)
                    logit_replay_images, logit_replay_targets, logit_replay_masks, _ = (
                        next(memory_iterator)
                    )

                replay_images = replay_images.to(self.device, non_blocking=True)
                replay_targets = replay_targets.to(self.device, non_blocking=True)

                logit_replay_images = logit_replay_images.to(
                    self.device, non_blocking=True
                )
                logit_replay_targets = logit_replay_targets.to(
                    self.device, non_blocking=True
                )
                logit_replay_masks = logit_replay_masks.to(
                    self.device, non_blocking=True
                )

                outputs = self.model(images)
                replay_outputs = self.model(replay_images)
                logit_replay_outputs = self.model(logit_replay_images)

                loss = self.criterion(
                    t,
                    outputs,
                    targets,
                    replay_outputs,
                    replay_targets,
                    logit_replay_outputs,
                    logit_replay_targets,
                    logit_replay_masks,
                )
                if self.model.is_early_exit():
                    loss = sum(loss)
            else:
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

    def criterion(
        self,
        t,
        outputs,
        targets,
        replay_outputs=None,
        replay_targets=None,
        logit_replay_outputs=None,
        logit_replay_targets=None,
        logit_replay_mask=None,
    ):
        if self.model.is_early_exit():
            ic_weights = self.model.get_ic_weights(
                current_epoch=self.current_epoch, max_epochs=self.nepochs
            )
            loss = []
            for ic_idx in range(len(ic_weights)):
                ic_outputs = outputs[ic_idx]
                ic_weight = ic_weights[ic_idx]

                loss_cur = torch.nn.functional.cross_entropy(
                    torch.cat(ic_outputs, dim=1), targets
                )

                if logit_replay_outputs is not None:
                    ic_logit_replay_outputs = logit_replay_outputs[ic_idx]
                    ic_logit_replay_targets = logit_replay_targets[:, ic_idx]
                    loss_der = torch.nn.functional.mse_loss(
                        torch.cat(ic_logit_replay_outputs, dim=1),
                        ic_logit_replay_targets,
                        reduction="none",
                    )
                    loss_der = torch.sum(loss_der * logit_replay_mask) / torch.sum(
                        logit_replay_mask
                    )
                else:
                    loss_der = 0

                if replay_outputs is not None:
                    ic_replay_outputs = replay_outputs[ic_idx]
                    loss_replay = torch.nn.functional.cross_entropy(
                        torch.cat(ic_replay_outputs, dim=1), replay_targets
                    )
                else:
                    loss_replay = 0

                ic_loss = ic_weight * (
                    loss_cur + self.alpha * loss_der + self.beta * loss_replay
                )
                loss.append(ic_loss)

            assert len(loss) == len(self.model.ic_layers) + 1
        else:
            """Returns the loss value"""
            loss_cur = torch.nn.functional.cross_entropy(
                torch.cat(outputs, dim=1), targets
            )

            if logit_replay_outputs is not None:
                loss_der = torch.nn.functional.mse_loss(
                    torch.cat(logit_replay_outputs, dim=1),
                    logit_replay_targets,
                    reduction="none",
                )
                loss_der = torch.sum(loss_der * logit_replay_mask) / torch.sum(
                    logit_replay_mask
                )
            else:
                loss_der = 0

            if replay_outputs is not None:
                loss_replay = torch.nn.functional.cross_entropy(
                    torch.cat(replay_outputs, dim=1), replay_targets
                )
            else:
                loss_replay = 0

            loss = loss_cur + self.alpha * loss_der + self.beta * loss_replay

        return loss


class DERExemplarsDatasetWrapper(Dataset):
    def __init__(self, device):
        self.device = device

        self.images = []
        self.logits = []
        self.logit_masks = []
        self.labels = []

        self.transform = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """Generates one sample of data"""
        image = Image.fromarray(self.images[idx])
        image = self.transform(image)
        logits = self.logits[idx]
        logits_mask = self.logit_masks[idx]
        target = self.labels[idx]
        return image, logits, logits_mask, target

    def _list_idx(self, image, target):
        eq = [np.all((image == ref_img)) for ref_img in self.images]
        if any(eq):
            idx = eq.index(True)
        else:
            idx = -1
        return idx

    def update(self, model, exemplars_dataset, val_transform):
        self.transform = exemplars_dataset.transform
        model.eval()

        torch.cuda.synchronize()
        time_start = time.time()

        updated_images = []
        updated_logits = []
        updated_masks = []
        updated_labels = []

        mask_size = max(exemplars_dataset.labels) + 1
        replaced_samples = 0
        with torch.no_grad():
            for image, label in zip(exemplars_dataset.images, exemplars_dataset.labels):
                idx = self._list_idx(image, label)
                if idx >= 0:
                    updated_images.append(self.images[idx])
                    updated_logits.append(self.logits[idx])
                    mask = self.logit_masks[idx]
                    mask = torch.cat(
                        (mask, torch.zeros(mask_size - mask.shape[0])), dim=0
                    )
                    updated_masks.append(mask)
                    updated_labels.append(self.labels[idx])
                else:
                    tensor_image = val_transform(Image.fromarray(image))
                    tensor_image = tensor_image.to(self.device).unsqueeze(0)
                    if model.is_early_exit():
                        per_ic_logits = model(tensor_image)
                        per_ic_logits = [
                            torch.cat(ic_logits, dim=1).squeeze()
                            for ic_logits in per_ic_logits
                        ]
                        logits = torch.stack(per_ic_logits, dim=0).cpu()
                        mask = torch.ones((logits.shape[1],))
                    else:
                        logits = torch.cat(model(tensor_image), dim=1).squeeze(0).cpu()
                        mask = torch.ones((logits.shape[0],))
                    updated_images.append(image)
                    updated_logits.append(logits)
                    updated_masks.append(mask)
                    updated_labels.append(label)
                    replaced_samples += 1
        assert (
            len(updated_images)
            == len(updated_logits)
            == len(updated_masks)
            == len(updated_labels)
        )
        assert len(updated_images) == len(exemplars_dataset)

        self.images = updated_images
        self.logits = updated_logits
        self.logit_masks = updated_masks
        self.labels = updated_labels

        torch.cuda.synchronize()
        time_end = time.time()
        print(
            f"Updated DER dataset replacing {replaced_samples} exemplars. Time spent: {time_end - time_start}[s]"
        )

    def mask_new_outputs(self, num_outputs):
        for idx in range(len(self.logits)):
            org_logits = self.logits[idx]
            org_logits_mask = self.logit_masks[idx]
            pad = torch.zeros((num_outputs,))

            if len(org_logits.shape) == 2:
                # early exit
                num_ic = org_logits.shape[0]
                padded_logits = torch.cat(
                    (org_logits, torch.zeros((num_ic, num_outputs))), dim=1
                )
            else:
                padded_logits = torch.cat((org_logits, pad), dim=0)
            padded_mask = torch.cat((org_logits_mask, pad), dim=0)

            self.logits[idx] = padded_logits
            self.logit_masks[idx] = padded_mask
