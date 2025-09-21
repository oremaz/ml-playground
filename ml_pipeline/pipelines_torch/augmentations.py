"""Tabular augmentation utilities with official research integrations."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from third_party import load_class


class MGS_GRF_Augmentor:
    """Wrapper around the official MGS-GRF implementation."""

    def __init__(
        self,
        *,
        repo_path: Optional[str] = None,
        env_var: str = "MGS_GRF_REPO",
        class_name: str = "MGS_GRF",
        module_candidates: Optional[Sequence[str]] = ("mgs_grf", "src.mgs_grf"),
        init_kwargs: Optional[dict] = None,
    ) -> None:
        self._cls = load_class(
            "mgs-grf",
            class_name,
            repo_path=repo_path,
            env_var=env_var,
            module_candidates=module_candidates,
        )
        self.impl = self._cls(**(init_kwargs or {}))

    def fit_resample(self, X: np.ndarray, y: np.ndarray, target_class: int, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.impl.fit_resample(X, y, target_class=target_class, n_samples=n_samples)


class TabEBMGenerator:
    """Wrapper around the official TabEBM generator (NeurIPS 2024)."""

    def __init__(
        self,
        *,
        repo_path: Optional[str] = None,
        env_var: str = "TABEBM_REPO",
        class_name: str = "TabEBM",
        module_candidates: Optional[Sequence[str]] = ("TabEBM", "tabe.TabEBM"),
        init_kwargs: Optional[dict] = None,
    ) -> None:
        self._cls = load_class(
            "TabEBM",
            class_name,
            repo_path=repo_path,
            env_var=env_var,
            module_candidates=module_candidates,
        )
        self.model = self._cls(**(init_kwargs or {}))

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "TabEBMGenerator":
        self.model.fit(X, y, **kwargs)
        return self

    def sample(self, n_samples: int, *, conditioned_on: Optional[int] = None) -> np.ndarray:
        if conditioned_on is None:
            return self.model.sample(n_samples)
        return self.model.sample_conditioned(n_samples, conditioned_on)

