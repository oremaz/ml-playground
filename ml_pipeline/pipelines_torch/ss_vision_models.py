"""Semi-supervised vision algorithms with optional hooks to official research repositories."""
from __future__ import annotations

from copy import deepcopy
from math import cos, pi
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from third_party import load_class, load_function


def _cosine_rampup(cur_step: int, rampup: int) -> float:
    if rampup <= 0:
        return 1.0
    cur_step = max(0, min(cur_step, rampup))
    return float(0.5 - 0.5 * cos(pi * cur_step / rampup))


class SSLImageClassifier(nn.Module):
    """Base wrapper exposing a torch.nn.Module compatible interface."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.backbone(x)

    def ssl_loss(
        self,
        batch_l: Tuple[torch.Tensor, torch.Tensor],
        batch_u: Tuple[torch.Tensor, torch.Tensor | None],
        epoch: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        raise NotImplementedError

    @torch.no_grad()
    def _weak(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @torch.no_grad()
    def _strong(self, x: torch.Tensor) -> torch.Tensor:
        return x


class PseudoLabel(SSLImageClassifier):
    def __init__(self, backbone: nn.Module, *, threshold: float = 0.95, unsup_weight: float = 1.0, rampup: int = 5) -> None:
        super().__init__(backbone)
        self.threshold = threshold
        self.unsup_weight = unsup_weight
        self.rampup = rampup

    def ssl_loss(self, batch_l, batch_u, epoch):
        x_l, y_l = batch_l
        x_u, _ = batch_u
        logits_l = self(x_l)
        loss_sup = F.cross_entropy(logits_l, y_l)

        with torch.no_grad():
            logits_w = self(self._weak(x_u))
            probs = torch.softmax(logits_w, dim=1)
            confidence, pseudo = probs.max(dim=1)
            mask = (confidence >= self.threshold).float()

        logits_s = self(self._strong(x_u))
        loss_unsup = (F.cross_entropy(logits_s, pseudo, reduction="none") * mask).mean()
        weight = self.unsup_weight * _cosine_rampup(epoch, self.rampup)
        return loss_sup, weight * loss_unsup, {"mask_rate": mask.mean().item(), "weight": weight}


class PiModel(SSLImageClassifier):
    def __init__(self, backbone: nn.Module, *, unsup_weight: float = 1.0, rampup: int = 5, temperature: float = 1.0) -> None:
        super().__init__(backbone)
        self.unsup_weight = unsup_weight
        self.rampup = rampup
        self.temperature = temperature

    def ssl_loss(self, batch_l, batch_u, epoch):
        x_l, y_l = batch_l
        x_u, _ = batch_u
        logits_l = self(x_l)
        loss_sup = F.cross_entropy(logits_l, y_l)

        probs_w = torch.softmax(self(self._weak(x_u)) / self.temperature, dim=1)
        probs_s = torch.softmax(self(self._strong(x_u)) / self.temperature, dim=1)
        loss_unsup = F.mse_loss(probs_w, probs_s)
        weight = self.unsup_weight * _cosine_rampup(epoch, self.rampup)
        return loss_sup, weight * loss_unsup, {"weight": weight}


class MeanTeacher(SSLImageClassifier):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        ema_decay: float = 0.999,
        unsup_weight: float = 1.0,
        rampup: int = 5,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(backbone)
        self.teacher = deepcopy(backbone).eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self.ema_decay = ema_decay
        self.unsup_weight = unsup_weight
        self.rampup = rampup
        self.temperature = temperature

    def ssl_loss(self, batch_l, batch_u, epoch):
        x_l, y_l = batch_l
        x_u, _ = batch_u
        logits_l = self(x_l)
        loss_sup = F.cross_entropy(logits_l, y_l)

        with torch.no_grad():
            teacher_logits = self.teacher(self._weak(x_u))
            teacher_probs = torch.softmax(teacher_logits / self.temperature, dim=1)

        student_probs = torch.softmax(self(self._strong(x_u)) / self.temperature, dim=1)
        loss_unsup = F.mse_loss(student_probs, teacher_probs)
        weight = self.unsup_weight * _cosine_rampup(epoch, self.rampup)
        logs = {"weight": weight, "_update_teacher": True}
        return loss_sup, weight * loss_unsup, logs

    @torch.no_grad()
    def update_teacher(self) -> None:
        for teacher_param, student_param in zip(self.teacher.parameters(), self.backbone.parameters()):
            teacher_param.data.mul_(self.ema_decay).add_(student_param.data, alpha=1 - self.ema_decay)


class STUCSSIC(SSLImageClassifier):
    """Self-adaptive threshold + unreliable sample contrastive learning (Zhang et al., 2024)."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        *,
        base_threshold: float = 0.95,
        min_threshold: float = 0.6,
        projection_dim: int = 128,
        temperature: float = 0.5,
        unsup_weight: float = 1.0,
        rampup: int = 10,
    ) -> None:
        super().__init__(backbone)
        self.num_classes = num_classes
        self.base_threshold = base_threshold
        self.min_threshold = min_threshold
        self.temperature = temperature
        self.unsup_weight = unsup_weight
        self.rampup = rampup
        self.projector = nn.Linear(num_classes, projection_dim)
        self.register_buffer("class_thresholds", torch.full((num_classes,), base_threshold))

    def _update_thresholds(self, epoch: int) -> None:
        relax = _cosine_rampup(epoch, self.rampup)
        updated = self.min_threshold + (self.base_threshold - self.min_threshold) * (1.0 - relax)
        self.class_thresholds.copy_(torch.full_like(self.class_thresholds, updated))

    def ssl_loss(self, batch_l, batch_u, epoch):
        self._update_thresholds(epoch)
        x_l, y_l = batch_l
        x_u, _ = batch_u
        logits_l = self(x_l)
        loss_sup = F.cross_entropy(logits_l, y_l)

        with torch.no_grad():
            logits_w = self(self._weak(x_u))
            probs_w = torch.softmax(logits_w, dim=1)
            confidence, pseudo = probs_w.max(dim=1)
            thresholds = self.class_thresholds[pseudo]
            reliable = (confidence >= thresholds).float()

        logits_s = self(self._strong(x_u))
        loss_ce = (F.cross_entropy(logits_s, pseudo, reduction="none") * reliable).mean()

        unreliable_idx = (reliable < 0.5).nonzero(as_tuple=True)[0]
        loss_con = torch.tensor(0.0, device=x_u.device)
        if unreliable_idx.numel() > 0:
            proj_w = F.normalize(self.projector(probs_w[unreliable_idx]), dim=1)
            proj_s = F.normalize(self.projector(torch.softmax(logits_s[unreliable_idx], dim=1)), dim=1)
            similarity = proj_w @ proj_s.t() / self.temperature
            targets = torch.arange(similarity.size(0), device=similarity.device)
            loss_con = F.cross_entropy(similarity, targets)

        weight = self.unsup_weight * _cosine_rampup(epoch, self.rampup)
        return loss_sup, weight * (loss_ce + loss_con), {
            "reliable_rate": reliable.mean().item(),
            "weight": weight,
        }


class CDMADDebiased(SSLImageClassifier):
    """Class-distribution-mismatch-aware debiasing (CVPR 2024, official repo integration)."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        *,
        threshold: float = 0.95,
        unsup_weight: float = 1.0,
        rampup: int = 8,
        repo_path: Optional[str] = None,
        env_var: str = "CDMAD_REPO",
        module_candidates: Optional[Tuple[str, ...]] = (
            "cdmad", "src.cdmad", "cdmad.core", "cdmad.models"
        ),
    ) -> None:
        super().__init__(backbone)
        self.threshold = threshold
        self.unsup_weight = unsup_weight
        self.rampup = rampup
        self.register_buffer("bias_offset", torch.zeros(num_classes))
        self._refiner = None
        self._bias_estimator = None
        try:
            self._refiner = load_function(
                "CDMAD",
                "refine_logits",
                repo_path=repo_path,
                env_var=env_var,
                module_candidates=module_candidates,
            )
        except (ImportError, FileNotFoundError):
            try:
                ref_class = load_class(
                    "CDMAD",
                    "CDMAD",
                    repo_path=repo_path,
                    env_var=env_var,
                    module_candidates=module_candidates,
                )
                self._refiner = ref_class(num_classes=num_classes)
            except (ImportError, FileNotFoundError):
                self._refiner = None

        if self._refiner is not None and hasattr(self._refiner, "estimate_bias"):
            self._bias_estimator = getattr(self._refiner, "estimate_bias")

    def _apply_refiner(self, logits: torch.Tensor) -> torch.Tensor:
        if self._refiner is None:
            return logits - self.bias_offset
        if callable(self._refiner):
            try:
                return self._refiner(logits, self.bias_offset)
            except TypeError:
                return self._refiner(logits)
        if hasattr(self._refiner, "refine_logits"):
            return self._refiner.refine_logits(logits, self.bias_offset)
        if hasattr(self._refiner, "forward"):
            return self._refiner.forward(logits)
        return logits - self.bias_offset

    @torch.no_grad()
    def _estimate_bias(self, input_shape: Tuple[int, ...], device: torch.device) -> None:
        if callable(self._bias_estimator):
            try:
                bias = self._bias_estimator(input_shape=input_shape, device=device)
                if bias is not None:
                    self.bias_offset.copy_(torch.as_tensor(bias, device=device))
                    return
            except TypeError:
                pass
        blank = torch.zeros((1,) + input_shape, device=device)
        logits = self(blank)
        self.bias_offset.copy_(logits.squeeze(0))

    def ssl_loss(self, batch_l, batch_u, epoch):
        x_l, y_l = batch_l
        x_u, _ = batch_u
        logits_l = self(x_l)
        loss_sup = F.cross_entropy(logits_l, y_l)

        if torch.allclose(self.bias_offset, torch.zeros_like(self.bias_offset)):
            self._estimate_bias(tuple(x_u.shape[1:]), x_u.device)

        with torch.no_grad():
            logits_w = self(self._weak(x_u))
            logits_w = self._apply_refiner(logits_w)
            probs = torch.softmax(logits_w, dim=1)
            confidence, pseudo = probs.max(dim=1)
            mask = (confidence >= self.threshold).float()

        logits_s = self(self._strong(x_u))
        loss_unsup = (F.cross_entropy(logits_s, pseudo, reduction="none") * mask).mean()
        weight = self.unsup_weight * _cosine_rampup(epoch, self.rampup)
        return loss_sup, weight * loss_unsup, {"mask_rate": mask.mean().item(), "weight": weight}
