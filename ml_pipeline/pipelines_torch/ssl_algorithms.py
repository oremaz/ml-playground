"""Reusable semi-supervised learning algorithms for classification models."""
from __future__ import annotations

from copy import deepcopy
from math import cos, pi
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor
AugmentFn = Optional[Callable[[Tensor], Tensor]]


def cosine_rampup(cur_step: int, rampup: int) -> float:
    """Cosine ramp-up used to gradually scale unsupervised losses."""
    if rampup <= 0:
        return 1.0
    cur_step = max(0, min(cur_step, rampup))
    return float(0.5 - 0.5 * cos(pi * cur_step / rampup))


class SemiSupervisedClassifier(nn.Module):
    """Base wrapper that injects optional weak/strong augmentations."""

    def __init__(self, backbone: nn.Module, *, weak: AugmentFn = None, strong: AugmentFn = None) -> None:
        super().__init__()
        self.backbone = backbone
        self._weak_aug = weak
        self._strong_aug = strong

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.backbone(x)

    def _weak(self, x: Tensor) -> Tensor:
        if self._weak_aug is None:
            return x
        return self._weak_aug(x)

    def _strong(self, x: Tensor) -> Tensor:
        if self._strong_aug is None:
            return x
        return self._strong_aug(x)

    def ssl_loss(
        self,
        batch_l: Tuple[Tensor, Tensor],
        batch_u: Tuple[Tensor, Optional[Tensor]],
        step: int,
    ) -> Tuple[Tensor, Tensor, Dict[str, float]]:
        raise NotImplementedError


class PseudoLabel(SemiSupervisedClassifier):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        threshold: float = 0.95,
        unsup_weight: float = 1.0,
        rampup: int = 5,
        weak: AugmentFn = None,
        strong: AugmentFn = None,
    ) -> None:
        super().__init__(backbone, weak=weak, strong=strong)
        self.threshold = threshold
        self.unsup_weight = unsup_weight
        self.rampup = rampup

    def ssl_loss(self, batch_l, batch_u, step):
        x_l, y_l = batch_l
        x_u, _ = batch_u
        logits_l = self(x_l)
        loss_sup = F.cross_entropy(logits_l, y_l)

        with torch.no_grad():
            logits_w = self(self._weak(x_u))
            probs = torch.softmax(logits_w, dim=1)
            confidence, pseudo = probs.max(dim=1)
            mask = (confidence >= self.threshold).float()

        weight = self.unsup_weight * cosine_rampup(step, self.rampup)
        logs = {"mask_rate": mask.mean().item(), "weight": weight}
        if mask.sum() == 0:
            return loss_sup, torch.tensor(0.0, device=loss_sup.device), logs

        logits_s = self(self._strong(x_u))
        loss_unsup = (F.cross_entropy(logits_s, pseudo, reduction="none") * mask).mean()
        return loss_sup, weight * loss_unsup, logs


class PiModel(SemiSupervisedClassifier):
    """Π-model consistency regularization (unlabeled data only).

    Note: the original paper applies consistency to both labeled and unlabeled
    data. This implementation applies it to unlabeled data only, which is a
    common simplification that avoids double-counting labeled samples.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        unsup_weight: float = 1.0,
        rampup: int = 5,
        temperature: float = 1.0,
        weak: AugmentFn = None,
        strong: AugmentFn = None,
    ) -> None:
        super().__init__(backbone, weak=weak, strong=strong)
        self.unsup_weight = unsup_weight
        self.rampup = rampup
        self.temperature = temperature

    def ssl_loss(self, batch_l, batch_u, step):
        x_l, y_l = batch_l
        x_u, _ = batch_u
        logits_l = self(x_l)
        loss_sup = F.cross_entropy(logits_l, y_l)

        probs_w = torch.softmax(self(self._weak(x_u)) / self.temperature, dim=1)
        probs_s = torch.softmax(self(self._strong(x_u)) / self.temperature, dim=1)
        loss_unsup = F.mse_loss(probs_w, probs_s)
        weight = self.unsup_weight * cosine_rampup(step, self.rampup)
        return loss_sup, weight * loss_unsup, {"weight": weight}


class MeanTeacher(SemiSupervisedClassifier):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        ema_decay: float = 0.999,
        unsup_weight: float = 1.0,
        rampup: int = 5,
        temperature: float = 1.0,
        weak: AugmentFn = None,
        strong: AugmentFn = None,
    ) -> None:
        super().__init__(backbone, weak=weak, strong=strong)
        self.teacher = deepcopy(backbone).eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self.ema_decay = ema_decay
        self.unsup_weight = unsup_weight
        self.rampup = rampup
        self.temperature = temperature

    def ssl_loss(self, batch_l, batch_u, step):
        x_l, y_l = batch_l
        x_u, _ = batch_u
        logits_l = self(x_l)
        loss_sup = F.cross_entropy(logits_l, y_l)

        with torch.no_grad():
            teacher_logits = self.teacher(self._weak(x_u))
            teacher_probs = torch.softmax(teacher_logits / self.temperature, dim=1)

        student_probs = torch.softmax(self(self._strong(x_u)) / self.temperature, dim=1)
        loss_unsup = F.mse_loss(student_probs, teacher_probs)
        weight = self.unsup_weight * cosine_rampup(step, self.rampup)
        logs = {"weight": weight}
        return loss_sup, weight * loss_unsup, logs

    @torch.no_grad()
    def update_teacher(self) -> None:
        for teacher_param, student_param in zip(self.teacher.parameters(), self.backbone.parameters()):
            teacher_param.data.mul_(self.ema_decay).add_(student_param.data, alpha=1 - self.ema_decay)
        for teacher_buf, student_buf in zip(self.teacher.buffers(), self.backbone.buffers()):
            teacher_buf.data.copy_(self.ema_decay * teacher_buf.data + (1 - self.ema_decay) * student_buf.data)
