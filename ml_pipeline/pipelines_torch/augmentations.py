"""Tabular augmentation utilities with official research integrations."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

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


class SimplicialSMOTE:
    """Implementation of Simplicial SMOTE (Kachan et al., 2025)."""

    def __init__(
        self,
        *,
        neighbors: int = 4,
        random_state: int = 42,
        noise: float = 0.0,
    ) -> None:
        if neighbors < 2:
            raise ValueError("neighbors must be >= 2 for simplicial interpolation")
        self.neighbors = neighbors
        self.random_state = np.random.RandomState(random_state)
        self.noise = noise

    def _sample_simplex(self, points: np.ndarray) -> np.ndarray:
        weights = self.random_state.dirichlet(np.ones(points.shape[0]))
        sample = weights @ points
        if self.noise > 0:
            sample = sample + self.random_state.normal(0, self.noise, size=sample.shape)
        return sample

    def sample(self, X: np.ndarray, y: np.ndarray, target_class: int, n_samples: int) -> np.ndarray:
        minority_idx = np.where(y == target_class)[0]
        if minority_idx.size == 0:
            raise ValueError("Target class has no samples to interpolate from.")
        nbrs = NearestNeighbors(n_neighbors=min(self.neighbors, minority_idx.size)).fit(X[minority_idx])
        synthetic = []
        for _ in range(n_samples):
            anchor_id = self.random_state.choice(minority_idx)
            anchor = X[anchor_id]
            _, indices = nbrs.kneighbors(anchor.reshape(1, -1), return_distance=True)
            neighbor_points = X[minority_idx][indices[0]]
            simplex_points = np.vstack([anchor, neighbor_points])
            synthetic.append(self._sample_simplex(simplex_points))
        return np.asarray(synthetic)


class MEBSMOTE:
    """Minimum enclosing ball SMOTE (Shangguan et al., 2024)."""

    def __init__(
        self,
        *,
        neighbors: int = 5,
        random_state: int = 42,
    ) -> None:
        if neighbors < 1:
            raise ValueError("neighbors must be >=1")
        self.neighbors = neighbors
        self.random_state = np.random.RandomState(random_state)

    def sample(self, X: np.ndarray, y: np.ndarray, target_class: int, n_samples: int) -> np.ndarray:
        minority_idx = np.where(y == target_class)[0]
        if minority_idx.size == 0:
            raise ValueError("Target class has no samples")
        nbrs = NearestNeighbors(n_neighbors=min(self.neighbors, minority_idx.size)).fit(X[minority_idx])
        synthetic = []
        for _ in range(n_samples):
            anchor_id = self.random_state.choice(minority_idx)
            anchor = X[anchor_id]
            _, indices = nbrs.kneighbors(anchor.reshape(1, -1), return_distance=True)
            neighbors_pts = X[minority_idx][indices[0]]
            centroid = neighbors_pts.mean(axis=0)
            radius = np.linalg.norm(neighbors_pts - centroid, axis=1).max()
            direction = centroid - anchor
            new_point = anchor + self.random_state.rand() * direction
            jitter = self.random_state.normal(0, radius * 0.05, size=new_point.shape)
            synthetic.append(new_point + jitter)
        return np.asarray(synthetic)
