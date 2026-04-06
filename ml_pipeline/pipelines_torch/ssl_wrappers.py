"""
ssl_wrappers.py
~~~~~~~~~~~~~~~~
Reusable SSL wrappers for the BenchmarkRunner.

Usage (in a notebook):
    from ml_pipeline.pipelines_torch.ssl_wrappers import SSLVisionWrapper, SSLTabularWrapper
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from ml_pipeline.pipelines_torch.vision_models import get_model
from ml_pipeline.pipelines_torch.ssl_algorithms import MeanTeacher, PseudoLabel, PiModel
from ml_pipeline.pipelines_torch.models import TorchMLPClassifier
from ml_pipeline.pipelines_torch.ss_models import SemiSupervisedTabular

__all__ = ["SSLVisionWrapper", "SSLTabularWrapper"]


# ---------------------------------------------------------------------------
# SSLVisionWrapper
# ---------------------------------------------------------------------------

class SSLVisionWrapper(nn.Module):
    """
    Wraps a vision SSL algorithm (MeanTeacher / PseudoLabel / PiModel) so it
    can be plugged into BenchmarkRunner via the custom ``fit`` interception.

    Parameters
    ----------
    ssl_algo : str
        One of ``"supervised"`` | ``"mean_teacher"`` | ``"pseudo_label"`` | ``"pi_model"``.
    num_classes : int
        Number of target classes.
    num_labeled : int
        How many samples are treated as labelled during ``fit``.
    """

    _VALID_ALGOS = {"supervised", "mean_teacher", "pseudo_label", "pi_model"}

    def __init__(
        self,
        ssl_algo: str = "mean_teacher",
        num_classes: int = 10,
        num_labeled: int = 2000,
    ) -> None:
        super().__init__()
        if ssl_algo not in self._VALID_ALGOS:
            raise ValueError(f"ssl_algo must be one of {self._VALID_ALGOS}, got {ssl_algo!r}")

        self.ssl_algo = ssl_algo
        self.num_labeled = num_labeled

        base_model = get_model("simple_cnn", num_classes=num_classes)

        def weak_aug(x):  return x + torch.randn_like(x) * 0.05
        def strong_aug(x): return x + torch.randn_like(x) * 0.15

        if ssl_algo == "mean_teacher":
            self.ssl_model = MeanTeacher(base_model, ema_decay=0.999, weak=weak_aug, strong=strong_aug)
        elif ssl_algo == "pseudo_label":
            self.ssl_model = PseudoLabel(base_model, threshold=0.95, weak=weak_aug, strong=strong_aug)
        elif ssl_algo == "pi_model":
            self.ssl_model = PiModel(base_model, weak=weak_aug, strong=strong_aug)
        else:  # supervised
            self.ssl_model = base_model

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ssl_model(x)

    # ------------------------------------------------------------------
    def fit(
        self,
        X,
        y,
        epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        **kwargs,
    ) -> list:
        dev = next(self.parameters()).device
        if torch.is_tensor(X): X = X.cpu().numpy()
        if torch.is_tensor(y): y = y.cpu().numpy()

        X_l, X_u, y_l, y_u = train_test_split(
            X, y, train_size=self.num_labeled, stratify=y, random_state=42
        )

        l_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_l), torch.LongTensor(y_l)),
            batch_size=batch_size, shuffle=True, drop_last=True,
        )

        u_loader = None
        if self.ssl_algo != "supervised":
            u_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_u), torch.LongTensor(y_u)),
                batch_size=batch_size * 2, shuffle=True, drop_last=True,
            )

        opt = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.train()
        global_step = 0
        history = []

        for epoch in range(epochs):
            iter_u = iter(u_loader) if u_loader is not None else None
            total_loss = 0.0

            for batch_l in l_loader:
                bl = (batch_l[0].to(dev), batch_l[1].to(dev))
                opt.zero_grad()

                if iter_u is not None:
                    try:
                        batch_u = next(iter_u)
                    except StopIteration:
                        iter_u = iter(u_loader)
                        batch_u = next(iter_u)
                    bu = (batch_u[0].to(dev), None)
                    loss_sup, loss_unsup, _ = self.ssl_model.ssl_loss(bl, bu, global_step)
                    loss = loss_sup + loss_unsup
                else:
                    logits = self.ssl_model(bl[0])
                    loss = F.cross_entropy(logits, bl[1])

                loss.backward()
                opt.step()

                if hasattr(self.ssl_model, "update_teacher"):
                    self.ssl_model.update_teacher()

                total_loss += loss.item()
                global_step += 1

            epoch_loss = total_loss / len(l_loader)
            print(f"[{self.ssl_algo}] Vision Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")
            history.append({"epoch": epoch + 1, "loss": epoch_loss})

        return history

    # ------------------------------------------------------------------
    def predict_proba(self, X) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            logits = self.forward(
                torch.FloatTensor(X).to(next(self.parameters()).device)
            )
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


# ---------------------------------------------------------------------------
# SSLTabularWrapper
# ---------------------------------------------------------------------------

class SSLTabularWrapper(nn.Module):
    """
    Wraps a tabular SSL algorithm so it can be plugged into BenchmarkRunner.

    Parameters
    ----------
    ssl_algo : str
        One of ``"supervised"`` | ``"mean_teacher"`` | ``"pseudo_label"``.
    input_dim : int
        Number of input features.
    num_classes : int
        Number of target classes.
    num_labeled : int
        How many samples are treated as labelled during ``fit``.
    hidden_dims : list[int]
        Hidden layer sizes for the MLP backbone.
    dropout : float
        Dropout probability for the backbone MLP.
    rampup : int
        Ramp-up length (epochs) for consistency loss.
    """

    _VALID_ALGOS = {"supervised", "mean_teacher", "pseudo_label"}

    def __init__(
        self,
        ssl_algo: str = "mean_teacher",
        input_dim: int = 54,
        num_classes: int = 7,
        num_labeled: int = 500,
        hidden_dims=None,
        dropout: float = 0.2,
        rampup: int = 50,
    ) -> None:
        super().__init__()
        if ssl_algo not in self._VALID_ALGOS:
            raise ValueError(f"ssl_algo must be one of {self._VALID_ALGOS}, got {ssl_algo!r}")

        self.ssl_algo = ssl_algo
        self.num_labeled = num_labeled

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.base_model = TorchMLPClassifier(
            input_dim, hidden_dims, num_classes, dropout=dropout, batchnorm=True
        )

        if ssl_algo != "supervised":
            self.ssl_model = SemiSupervisedTabular(
                self.base_model, num_classes,
                use_mean_teacher=(ssl_algo == "mean_teacher"),
                rampup=rampup,
            )
        else:
            self.ssl_model = self.base_model

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ssl_model(x)

    # ------------------------------------------------------------------
    def fit(
        self,
        X,
        y,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        **kwargs,
    ) -> list:
        dev = next(self.parameters()).device
        if torch.is_tensor(X): X = X.cpu().numpy()
        if torch.is_tensor(y): y = y.cpu().numpy()

        X_l, X_u, y_l, y_u = train_test_split(
            X, y, train_size=self.num_labeled, stratify=y, random_state=42
        )

        l_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_l), torch.LongTensor(y_l)),
            batch_size=batch_size, shuffle=True, drop_last=True,
        )

        u_loader = None
        if self.ssl_algo != "supervised":
            u_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_u), torch.LongTensor(y_u)),
                batch_size=batch_size * 4, shuffle=True, drop_last=True,
            )

        opt = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.train()
        global_step = 0
        history = []

        for epoch in range(epochs):
            iter_u = iter(u_loader) if u_loader is not None else None
            total_loss = 0.0

            for batch_l in l_loader:
                bl = (batch_l[0].to(dev), batch_l[1].to(dev))
                opt.zero_grad()

                if iter_u is not None:
                    try:
                        batch_u = next(iter_u)
                    except StopIteration:
                        iter_u = iter(u_loader)
                        batch_u = next(iter_u)
                    bu = (batch_u[0].to(dev), None)
                    loss, _ = self.ssl_model.step(bl, bu, global_step)
                else:
                    logits = self.ssl_model(bl[0])
                    loss = F.cross_entropy(logits, bl[1])

                loss.backward()
                opt.step()

                if hasattr(self.ssl_model, "post_step"):
                    self.ssl_model.post_step()

                total_loss += loss.item()
                global_step += 1

            epoch_loss = total_loss / len(l_loader)
            print(f"[{self.ssl_algo}] Tabular Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")
            history.append({"epoch": epoch + 1, "loss": epoch_loss})

        return history

    # ------------------------------------------------------------------
    def predict_proba(self, X) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            logits = self.forward(
                torch.FloatTensor(X).to(next(self.parameters()).device)
            )
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
