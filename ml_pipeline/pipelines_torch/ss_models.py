"""Semi-supervised learning utilities for tabular models."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .ssl_algorithms import MeanTeacher, PseudoLabel


class SemiSupervisedTabular(nn.Module):
    """Wrap tabular backbones with reusable SSL algorithms."""

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
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_mean_teacher = use_mean_teacher
        self.noise_std = noise_std

        if self.use_mean_teacher:
            self.ssl_model = MeanTeacher(
                base_model,
                ema_decay=ema_decay,
                unsup_weight=unsup_weight,
                rampup=rampup,
                temperature=temperature,
                weak=self._weak,
                strong=self._strong,
            )
        else:
            self.ssl_model = PseudoLabel(
                base_model,
                threshold=threshold,
                unsup_weight=unsup_weight,
                rampup=rampup,
                weak=self._weak,
                strong=self._strong,
            )

        self.base = self.ssl_model.backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.ssl_model(x)

    def _weak(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _strong(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise_std <= 0:
            return x
        return x + self.noise_std * torch.randn_like(x)

    def step(
        self,
        batch_l: Tuple[torch.Tensor, torch.Tensor],
        batch_u: Tuple[torch.Tensor, torch.Tensor | None],
        epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_sup, loss_unsup, logs = self.ssl_model.ssl_loss(batch_l, batch_u, epoch)
        total_loss = loss_sup + loss_unsup
        return total_loss, logs

    @torch.no_grad()
    def post_step(self) -> None:
        if hasattr(self.ssl_model, "update_teacher"):
            self.ssl_model.update_teacher()
