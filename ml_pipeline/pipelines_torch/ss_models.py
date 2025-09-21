"""Semi-supervised learning utilities for tabular models."""
from __future__ import annotations

from copy import deepcopy
from math import cos, pi
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cosine_rampup(step: int, rampup: int) -> float:
    if rampup <= 0:
        return 1.0
    step = max(0, min(step, rampup))
    return float(0.5 - 0.5 * cos(pi * step / rampup))


class SemiSupervisedTabular(nn.Module):
    """Generic semi-supervised wrapper for tabular models."""

    def __init__(
        self,
        base_model: nn.Module,
        num_classes: int,
        *,
        use_mean_teacher: bool = True,
        ema_decay: float = 0.999,
        threshold: float = 0.95,
        unsup_weight: float = 1.0,
        rampup: int = 5,
        noise_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.base = base_model
        self.num_classes = num_classes
        self.use_mean_teacher = use_mean_teacher
        self.ema_decay = ema_decay
        self.threshold = threshold
        self.unsup_weight = unsup_weight
        self.rampup = rampup
        self.noise_std = noise_std
        if self.use_mean_teacher:
            self.teacher = deepcopy(base_model).eval()
            for param in self.teacher.parameters():
                param.requires_grad_(False)
        else:
            self.teacher = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.base(x)

    def _augment(self, x: torch.Tensor, std: float) -> torch.Tensor:
        if std <= 0:
            return x
        return x + std * torch.randn_like(x)

    def _teacher_predict(self, x: torch.Tensor) -> torch.Tensor:
        if self.teacher is not None:
            return self.teacher(x)
        return self.base(x)

    def step(
        self,
        batch_l: Tuple[torch.Tensor, torch.Tensor],
        batch_u: Tuple[torch.Tensor, torch.Tensor | None],
        epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        x_l, y_l = batch_l
        x_u, _ = batch_u

        logits_l = self(self._augment(x_l, 0.0))
        loss_sup = F.cross_entropy(logits_l, y_l)

        with torch.no_grad():
            teacher_logits = self._teacher_predict(self._augment(x_u, 0.0))
            probs = torch.softmax(teacher_logits, dim=1)
            confidence, pseudo = probs.max(dim=1)
            mask = (confidence >= self.threshold).float()

        logits_u = self(self._augment(x_u, self.noise_std))
        loss_unsup = (F.cross_entropy(logits_u, pseudo, reduction="none") * mask).mean()
        weight = self.unsup_weight * _cosine_rampup(epoch, self.rampup)
        total_loss = loss_sup + weight * loss_unsup
        logs = {"mask_rate": mask.mean().item(), "weight": weight}
        return total_loss, logs

    @torch.no_grad()
    def post_step(self) -> None:
        if self.teacher is None:
            return
        for teacher_param, student_param in zip(self.teacher.parameters(), self.base.parameters()):
            teacher_param.data.mul_(self.ema_decay).add_(student_param.data, alpha=1 - self.ema_decay)
