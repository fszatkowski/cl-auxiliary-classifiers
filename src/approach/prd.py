from argparse import ArgumentParser
from copy import deepcopy
from typing import Any, Callable, Dict, List

import kornia
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from approach.incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class SupConLoss(nn.Module):
    def __init__(self, temperature, device, base_temperature=0.07, contrast_mode="all"):
        super(SupConLoss, self).__init__()

        self.temperature = temperature
        self.base_temperature = base_temperature
        self.contrast_mode = contrast_mode

        self.device = device

    def forward(self, features, labels=None):
        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        else:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown contrast mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class Prototypes(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        n_classes_per_task: int,
        n_tasks: int,
    ):
        super(Prototypes, self).__init__()

        self.heads = self._create_prototypes(
            dim_in=feat_dim,
            n_classes=n_classes_per_task,
            n_heads=n_tasks,
        )

    def _create_prototypes(
        self, dim_in: int, n_classes: int, n_heads: int
    ) -> torch.nn.ModuleDict:
        layers = {}
        for t in range(n_heads):
            layers[str(t)] = nn.Linear(dim_in, n_classes, bias=False)

        return nn.ModuleDict(layers)

    def forward(self, x: torch.FloatTensor, task_id: int) -> torch.FloatTensor:
        out = self.heads[str(task_id)](x)
        return out


def add_linear(dim_in, dim_out, batch_norm, relu):
    layers = []
    layers.append(nn.Linear(dim_in, dim_out))
    if batch_norm:
        layers.append(nn.BatchNorm1d(dim_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class ProjectionMLP(nn.Module):
    def __init__(self, dim_in, hidden_dim, feat_dim, batch_norm, num_layers):
        super(ProjectionMLP, self).__init__()

        self.layers = self._make_layers(
            dim_in, hidden_dim, feat_dim, batch_norm, num_layers
        )

    def _make_layers(self, dim_in, hidden_dim, feat_dim, batch_norm, num_layers):
        layers = []
        layers.append(add_linear(dim_in, hidden_dim, batch_norm=batch_norm, relu=True))

        for _ in range(num_layers - 2):
            layers.append(
                add_linear(hidden_dim, hidden_dim, batch_norm=batch_norm, relu=True)
            )

        layers.append(add_linear(hidden_dim, feat_dim, batch_norm=False, relu=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Appr(Inc_Learning_Appr):
    """
    PRD method from "Prototype-Sample Relation Distillation: Towards Replay-Free Continual Learning"
    paper: https://proceedings.mlr.press/v202/asadi23a/asadi23a.pdf
    original code: https://github.com/naderAsadi/Probing-Continual-Learning/blob/main/methods/repe.py

    TODO as implemented here it is not working too well - maybe we need to optimize or drop this method
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
        scheduler_name="multistep",
        scheduler_milestones=None,
        projection_head_output_size: int = 128,
        projection_head_hidden_size: int = 512,
        projection_head_num_layers: int = 3,
        feature_dim: int = 128,
        n_classes_per_task: int = None,
        n_tasks: int = None,
        prototypes_lr: float = 0.01,
        prototypes_coef: float = 2.0,
        supcon_temperature: float = 0.1,
        distill_coef: float = 4.0,
        distill_temp: float = 1.0,
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
        # TODO validate if this works well and where does the normalization come from
        self.train_tf = tfs = nn.Sequential(
            kornia.augmentation.RandomCrop(size=(32, 32), padding=4, fill=-1),
            kornia.augmentation.RandomHorizontalFlip(),
            kornia.augmentation.ColorJitter(0.4, 0.4, 0.4, p=0.8),
            kornia.augmentation.RandomGrayscale(p=0.2),
        )

        self.n_classes_per_task = n_classes_per_task
        # TODO handle early exits for prototypes and projection heads
        self.prototypes = Prototypes(
            feat_dim=feature_dim,
            n_classes_per_task=n_classes_per_task,
            n_tasks=n_tasks,
        ).to(self.device)
        self.prototypes_lr = prototypes_lr
        self.projection_head = ProjectionMLP(
            dim_in=feature_dim,
            feat_dim=projection_head_output_size,
            hidden_dim=projection_head_hidden_size,
            batch_norm=False,
            num_layers=projection_head_num_layers,
        )
        self.supcon_loss = SupConLoss(
            temperature=supcon_temperature, device=self.device
        )
        self.prototypes.to(self.device)
        self.projection_head.to(self.device)

        self.first_task_id = 0
        self.distill_temp = distill_temp
        self.distill_coef = distill_coef
        self.prototypes_coef = prototypes_coef

        self.prev_model = None
        self.prev_prototypes = None

        if len(self.exemplars_dataset) != 0:
            raise ValueError("PRD is meant to be used without exemplars")

    def _get_optimizer(self):
        # TODO handle early exits
        base_params = list(self.model.parameters()) + list(
            self.projection_head.parameters()
        )
        prototypes_params = list(self.prototypes.parameters())

        params = [
            {"params": base_params},
            {
                "params": prototypes_params,
                "lr": self.prototypes_lr,
                "momentum": 0.0,
                "weight_decay": 0.0,
            },
        ]

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

    @staticmethod
    def extra_parser(args):
        # TODO
        parser = ArgumentParser()
        parser.add_argument(
            "--projection-head-output-size",
            default=128,
            type=int,
            required=False,
            help="Output size for projection layer",
        )
        parser.add_argument(
            "--projection-head-hidden-size",
            default=512,
            type=int,
            required=False,
            help="Hidden size of the projection layer",
        )
        parser.add_argument(
            "--projection-head-num-layers",
            default=3,
            type=int,
            required=False,
            help="Number of hidden layers in the projection head",
        )
        parser.add_argument(
            "--feature-dim",
            default=128,
            type=int,
            required=False,
            help="Dimensionality of the feature extractor",
        )

        # TODO can this be replaced?
        parser.add_argument(
            "--n-classes-per-task",
            default=10,
            type=int,
            required=False,
            help="Number of classes per task",
        )
        parser.add_argument(
            "--n-tasks",
            default=10,
            type=int,
            required=False,
            help="Number of tasks",
        )

        parser.add_argument(
            "--prototypes-lr",
            default=0.01,
            type=float,
            required=False,
            help="LR for prototypes",
        )
        parser.add_argument(
            "--prototypes-coef",
            default=2.0,
            type=float,
            required=False,
            help="Weight for prototypes loss",
        )
        parser.add_argument(
            "--supcon-temperature",
            default=0.1,
            type=float,
            required=False,
            help="Temperature for SupCon loss for contrastive learning",
        )
        parser.add_argument(
            "--distill-coef",
            default=4.0,
            type=float,
            required=False,
            help="Relation distillation weight",
        )
        parser.add_argument(
            "--distill-temp",
            default=1.0,
            type=float,
            required=False,
            help="Temperature for relational distillation loss",
        )
        return parser.parse_known_args(args)

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    def adapt_train_loader(self, trn_loader, val_loader):
        # Since we are using contrastive learning the augmentations have to be manually done per each image
        # Basicially we need to strip the train loader from all the augmentations since the train loop uses custom augs anyway
        trn_loader.dataset.transform = val_loader.dataset.transform

    def train_loop(self, t, trn_loader, val_loader):
        self.adapt_train_loader(trn_loader, val_loader)

        if t > 0:
            trn_loader = torch.utils.data.DataLoader(
                trn_loader.dataset + self.exemplars_dataset,
                batch_size=trn_loader.batch_size,
                shuffle=True,
                num_workers=trn_loader.num_workers,
                pin_memory=trn_loader.pin_memory,
            )

        super().train_loop(t, trn_loader, val_loader)

        self.exemplars_dataset.collect_exemplars(
            self.model, trn_loader, val_loader.dataset.transform
        )

    def train_epoch(self, t, trn_loader):
        self.model.train()
        self.projection_head.train()
        self.prototypes.train()

        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        for images, targets in trn_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Forward current model
            supcon_loss, loss_p, loss_d = self.compute_loss(t, images, targets)
            if self.model.is_early_exit():
                # TODO
                loss = sum(loss)
            else:
                self.logger.log_scalar(
                    task=None,
                    iter=None,
                    name="loss_supcon",
                    group=f"debug_t{t}",
                    value=float(supcon_loss),
                )
                self.logger.log_scalar(
                    task=None,
                    iter=None,
                    name="loss_p",
                    group=f"debug_t{t}",
                    value=float(loss_p),
                )
                self.logger.log_scalar(
                    task=None,
                    iter=None,
                    name="loss_d",
                    group=f"debug_t{t}",
                    value=float(loss_d),
                )
                loss = (
                    supcon_loss
                    + self.prototypes_coef * loss_p
                    + self.distill_coef * loss_d
                )
                self.logger.log_scalar(
                    task=None,
                    iter=None,
                    name="total_loss",
                    group=f"debug_t{t}",
                    value=float(loss),
                )
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            params_to_clip = (
                list(self.model.parameters())
                + list(self.projection_head.parameters())
                + list(self.prototypes.parameters())
            )
            torch.nn.utils.clip_grad_norm_(params_to_clip, self.clipgrad)
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def eval(self, t, val_loader, features_save_dir=None):
        with torch.inference_mode():
            if self.model.is_early_exit():
                # TODO
                total_loss, total_acc_taw, total_acc_tag, total_num = (
                    np.zeros((len(self.model.ic_layers) + 1,)),
                    np.zeros((len(self.model.ic_layers) + 1,)),
                    np.zeros((len(self.model.ic_layers) + 1,)),
                    np.zeros((len(self.model.ic_layers) + 1,)),
                )
            else:
                total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            self.projection_head.eval()
            self.prototypes.eval()

            for images, targets in val_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                # compute loss and predictions
                supcon_loss, loss_p, loss_d = self.compute_loss(t, images, targets)
                loss = supcon_loss + loss_p + loss_d
                dists = self.get_distances(images, t)
                offset = self.model.task_offset[t]
                pred = dists[t].argmin(1)
                hits_taw = (
                    pred + offset == targets.to(self.device, non_blocking=True)
                ).float()
                # Task-Agnostic Multi-Head
                pred = torch.cat(dists, dim=1).argmin(1)
                hits_tag = (pred == targets.to(self.device, non_blocking=True)).float()
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)

            return (
                total_loss / total_num,
                total_acc_taw / total_num,
                total_acc_tag / total_num,
            )

    def post_train_process(self, t, trn_loader):
        self.prev_model = deepcopy(self.model)
        self.prev_model.freeze_all()
        # self.prev_model.eval()

        self.prev_prototypes = deepcopy(self.prototypes)
        # self.prev_prototypes.eval()

    def _distillation_loss(
        self, current_out: torch.FloatTensor, prev_out: torch.FloatTensor
    ) -> torch.FloatTensor:

        log_p = torch.log_softmax(current_out / self.distill_temp, dim=1)  # student
        q = torch.softmax(prev_out / self.distill_temp, dim=1)  # teacher
        result = torch.nn.KLDivLoss(reduction="batchmean")(log_p, q)
        return result

    def relation_distillation_loss(
        self, features: torch.FloatTensor, data: torch.FloatTensor, t: int
    ) -> torch.FloatTensor:

        if self.prev_model is None:
            return 0.0

        with torch.inference_mode():
            _, old_features = self.prev_model(data, return_features=True)

        with torch.inference_mode():
            old_model_preds = self._get_scores(
                old_features, prototypes=self.prev_prototypes, t=t
            )
        new_model_preds = self._get_scores(features, prototypes=self.prototypes, t=t)

        dist_loss = 0
        for task_id in range(t + 1):
            dist_loss += self._distillation_loss(
                current_out=new_model_preds[task_id],
                prev_out=old_model_preds[task_id].clone(),
            )

        return dist_loss

    def get_distances(
        self, images: torch.FloatTensor, t: int
    ) -> List[torch.FloatTensor]:
        _, features = self.model(images, return_features=True)
        outputs = []
        for head in list(self.prototypes.heads.values())[: t + 1]:
            prototypes = head.weight.data.clone()
            prototypes = F.normalize(prototypes, dim=1, p=2)
            features = F.normalize(features, dim=1, p=2)  # pass through projection head
            output = F.linear(input=features, weight=prototypes)
            outputs.append(output)
        return outputs

    def _get_scores(
        self, features: torch.FloatTensor, prototypes: Prototypes, t: int
    ) -> torch.FloatTensor:
        scores = []
        for _t in range(t + 1):
            nobout = F.linear(features, prototypes.heads[str(t)].weight)
            wnorm = torch.norm(prototypes.heads[str(t)].weight, dim=1, p=2)
            nobout = nobout / wnorm
            scores.append(nobout)
        return scores

    def linear_loss(
        self,
        features: torch.FloatTensor,
        labels: torch.Tensor,
        t: int,
        lam: int = 1,
    ) -> torch.FloatTensor:

        if lam == 0:
            features = features.detach().clone()  # [0:labels.size(0)]

        nobout = F.linear(features, self.prototypes.heads[str(t)].weight)
        wnorm = torch.norm(self.prototypes.heads[str(t)].weight, dim=1, p=2)
        nobout = nobout / wnorm
        feat_norm = torch.norm(features, dim=1, p=2)

        if not t == 0:
            labels -= t * self.n_classes_per_task  # shift targets
        indecies = labels.unsqueeze(1)
        out = nobout.gather(1, indecies).squeeze()
        out = out / feat_norm
        loss = sum(1 - out) / out.size(0)

        return loss

    def _prototypes_contrast_loss(self, task_id: int):
        # anchor = self.prototypes.heads[str(task_id)].weight.
        contrast_prot = []
        for key, head in self.prototypes.heads.items():
            if int(key) < task_id:
                contrast_prot.append(deepcopy(head).weight.data)

        if len(contrast_prot) == 0:
            return 0.0

        contrast_prot = F.normalize(torch.cat(contrast_prot, dim=-1), dim=1, p=2)
        anchors = F.normalize(
            self.prototypes.heads[str(task_id)].weight.data, dim=1, p=2
        )

        logits = torch.div(
            torch.matmul(anchors.T, contrast_prot), self.args.supcon_temperature
        )
        log_prob = torch.log(torch.exp(logits).sum(1))
        loss = -log_prob.sum() / log_prob.size(0)

        return loss

    def compute_loss(self, t, images, targets):
        x1, x2 = self.train_tf(images), self.train_tf(images)
        aug_data = torch.cat((x1, x2), dim=0)
        bsz = images.shape[0]

        _, features = self.model(aug_data, return_features=True)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)

        # SupCon Loss
        proj_features = self.projection_head(features)
        proj_features = F.normalize(proj_features, dim=1)  # normalize embedding
        proj_f1, proj_f2 = torch.split(proj_features, [bsz, bsz], dim=0)
        proj_features = torch.cat([proj_f1.unsqueeze(1), proj_f2.unsqueeze(1)], dim=1)
        supcon_loss = self.supcon_loss(proj_features, labels=targets)

        # Distillation loss
        loss_d = self.relation_distillation_loss(features, data=aug_data, t=t)

        # Prorotypes loss
        loss_p = self.linear_loss(
            features.detach().clone(),
            labels=targets.repeat(2),
            t=t,
        )

        return supcon_loss, loss_p, loss_d
