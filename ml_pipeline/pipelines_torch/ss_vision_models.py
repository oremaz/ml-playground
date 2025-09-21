"""Semi-supervised vision algorithms with optional hooks to official research repositories."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from third_party import load_class, load_function
from .ssl_algorithms import (
    MeanTeacher as _MeanTeacher,
    PiModel as _PiModel,
    PseudoLabel as _PseudoLabel,
    SemiSupervisedClassifier,
    cosine_rampup as _cosine_rampup,
)

__all__ = [
    "SSLImageClassifier",
    "PseudoLabel",
    "PiModel",
    "MeanTeacher",
    "CDMADDebiased",
]

PseudoLabel = _PseudoLabel
PiModel = _PiModel
MeanTeacher = _MeanTeacher


class SSLImageClassifier(SemiSupervisedClassifier):
    """Base wrapper exposing a torch.nn.Module compatible interface."""

    def __init__(self, backbone: nn.Module, *, weak=None, strong=None) -> None:
        super().__init__(backbone, weak=weak, strong=strong)


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
