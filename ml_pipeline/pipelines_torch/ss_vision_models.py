"""Semi-supervised vision algorithms with optional hooks to official research repositories."""
from __future__ import annotations

import torch.nn as nn

from .ssl_algorithms import (
    MeanTeacher as _MeanTeacher,
    PiModel as _PiModel,
    PseudoLabel as _PseudoLabel,
    SemiSupervisedClassifier,
)

__all__ = [
    "SSLImageClassifier",
    "PseudoLabel",
    "PiModel",
    "MeanTeacher",
]

PseudoLabel = _PseudoLabel
PiModel = _PiModel
MeanTeacher = _MeanTeacher


class SSLImageClassifier(SemiSupervisedClassifier):
    """Abstract base for vision SSL classifiers.

    Subclass this and implement ``ssl_loss`` using one of the concrete
    algorithms (``MeanTeacher``, ``PseudoLabel``, ``PiModel``), or compose
    directly with those classes instead.
    """

    def __init__(self, backbone: nn.Module, *, weak=None, strong=None) -> None:
        super().__init__(backbone, weak=weak, strong=strong)


